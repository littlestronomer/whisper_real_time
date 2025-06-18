import argparse
import os
import numpy as np
import torch
from faster_whisper import WhisperModel
import speech_recognition as sr
import threading
from concurrent.futures import ThreadPoolExecutor

from datetime import datetime, timedelta, timezone
from queue import Queue
from time import sleep
from sys import platform

# pyannote imports
from pyannote.audio import Pipeline
from dotenv import load_dotenv
from huggingface_hub import login

import ctranslate2

# import cProfile
# import io
# import pstats

import torch.ao.quantization

# Load environment variables (HUGGINGFACE_TOKEN, etc.)
load_dotenv()

class TranscriptionManager:
    def __init__(self, args, audio_model, diarization_pipeline):
        self.args = args
        self.audio_model = audio_model
        self.diarization_pipeline = diarization_pipeline
        self.transcribe_queue = Queue()
        self.diarize_queue = Queue()
        self.utterances = []
        self.current_speaker = None
        self.current_text = ""
        self.current_start = None
        self.current_end = None
        self.last_diarization = None
        self.diarization_buffer = bytearray()
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.audio_chunks_processed = 0
        self.last_print_time = datetime.now()
        self.phrase_bytes = bytes()
        self.phrase_time = None
        self.diarization_window = 3  # seconds
        self.sliding_window = 1.5  # seconds
        self.last_diarization_time = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_diarizing = threading.Event()  # Used to check if diarization is already running
        # Add speaker mapping functionality
        self.speaker_mapping = {}  # Maps pyannote speaker IDs to consistent IDs
        self.next_speaker_id = 0
        print(f"Using device: {self.device}")


        

    def get_consistent_speaker_id(self, pyannote_speaker_id):
        """Maps pyannote speaker IDs to consistent IDs throughout the conversation"""
        if pyannote_speaker_id not in self.speaker_mapping:
            self.speaker_mapping[pyannote_speaker_id] = f"SPEAKER_{self.next_speaker_id:02d}"
            self.next_speaker_id += 1
        return self.speaker_mapping[pyannote_speaker_id]

    def process_transcription(self, audio_np):
        try:
            segments, info = self.audio_model.transcribe(
                audio_np,
                beam_size=2,                                        # can  be changed to 1 for faster inference
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )


            with self.lock:
                self.process_segments(segments)
                self.audio_chunks_processed += 1
                current_time = datetime.now()
                if (current_time - self.last_print_time).total_seconds() >= 1.0:
                    print(f"\rProcessing audio chunks: {self.audio_chunks_processed}", end="")
                    self.last_print_time = current_time
        except Exception as e:
            print(f"\nError in transcription: {str(e)}")

    def process_diarization(self):
        try:
            current_time = datetime.now()

            # Skip if not enough time has passed
            if (self.last_diarization_time is not None and
                (current_time - self.last_diarization_time).total_seconds() < self.sliding_window):
                return

            # Skip if already running
            if self.is_diarizing.is_set():
                return

            # Skip if not enough audio data
            required_samples = int(16000 * 2 * self.diarization_window)
            if len(self.diarization_buffer) < required_samples:
                return

            self.is_diarizing.set()  # Mark as running

            # Convert to numpy array and normalize
            waveform_np = np.frombuffer(self.diarization_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            waveform = torch.from_numpy(waveform_np).unsqueeze(0).to(self.device)

            with self.lock:
                if self.diarization_pipeline is not None:
                    self.last_diarization = self.diarization_pipeline(
                        {"waveform": waveform, "sample_rate": 16000},
                        min_speakers=1,
                        max_speakers=2
                    )

                    # Keep only the last sliding_window worth of audio
                    keep_samples = int(16000 * 2 * self.sliding_window)
                    self.diarization_buffer = self.diarization_buffer[-keep_samples:]
                else:
                    print("\nWarning: Diarization pipeline not initialized")

        except Exception as e:
            print(f"\nError in diarization: {str(e)}")
            import traceback
            traceback.print_exc()

        finally:
            self.is_diarizing.clear()  # Mark as done


    def process_segments(self, segments):
        flush_happened = False
        for seg in segments:
            start, end = seg.start, seg.end
            text = seg.text.strip()
            if not text:  # Skip empty segments
                continue

            midpoint = 0.5 * (start + end)
            spk = "unknown"
            
            if self.last_diarization:
                for turn, _, label in self.last_diarization.itertracks(yield_label=True):
                    if turn.start <= midpoint <= turn.end:
                        spk = self.get_consistent_speaker_id(label)  # Use consistent speaker ID
                        break

            if self.current_speaker and spk != self.current_speaker:
                self.utterances.append({
                    "speaker": self.current_speaker,
                    "text": self.current_text,
                    "start": self.current_start,
                    "end": self.current_end
                })
                print(f"\n{self.current_speaker}: {self.current_text}")
                self.current_text = ""
                flush_happened = True

            if not self.current_text:
                self.current_start = start
            self.current_text = (self.current_text + " " + text).strip()
            self.current_speaker = spk
            self.current_end = end

            if text.endswith((".", "?", "!")):
                self.utterances.append({
                    "speaker": self.current_speaker,
                    "text": self.current_text,
                    "start": self.current_start,
                    "end": self.current_end
                })
                print(f"\n{self.current_speaker}: {self.current_text}")
                self.current_text = ""
                flush_happened = True

        if not flush_happened and self.current_text:
            self.utterances.append({
                "speaker": self.current_speaker,
                "text": self.current_text,
                "start": self.current_start,
                "end": self.current_end
            })
            print(f"\n{self.current_speaker}: {self.current_text}")
            self.current_text = ""

    def get_speaker_mapping(self):
        """Returns the current speaker mapping dictionary"""
        return self.speaker_mapping

def main():
    parser = argparse.ArgumentParser(
        description="Real‑time Whisper transcription + Pyannote speaker diarization"
    )
    parser.add_argument(
        "--model",
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Model to use",
    )
    parser.add_argument(
        "--non_english",
        action="store_true",
        help="Don't use the english model.",
    )
    parser.add_argument(
        "--energy_threshold",
        type=int,
        default=1000,
        help="Energy level for mic to detect.",
    )
    parser.add_argument(
        "--record_timeout",
        type=float,
        default=2,
        help="How real time the recording is in seconds.",
    )
    parser.add_argument(
        "--phrase_timeout",
        type=float,
        default=3,
        help="How much empty space between recordings before we consider it a new line in the transcription.",
    )
    parser.add_argument(
        "--diarization_window",
        type=float,
        default=3,
        help="Window size in seconds for speaker diarization (smaller=faster, less accurate).",
    )
    parser.add_argument(
        "--sliding_window",
        type=float,
        default=1.5,
        help="Sliding window step in seconds for diarization (smaller=more frequent updates).",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Your Hugging Face access token (or set HUGGINGFACE_TOKEN env var)",
    )
    if "linux" in platform:
        parser.add_argument(
            "--default_microphone",
            type=str,
            default="pulse",
            help="Substring of microphone name to use (or 'list' to enumerate)",
        )

    args = parser.parse_args()

    # Determine HF token
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or args.hf_token
    if not hf_token:
        print("❌ Error: Hugging Face token not provided.")
        print("   Either set $HUGGINGFACE_TOKEN or pass --hf_token YOUR_TOKEN")
        return
    login(token=hf_token)

    # Setup recognizer and microphone
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = args.energy_threshold
    recognizer.dynamic_energy_threshold = False
    recognizer.pause_threshold = args.phrase_timeout

    print(f"\nUsing energy threshold: {recognizer.energy_threshold}")
    print("Adjusting for ambient noise...")

    if "linux" in platform:
        if args.default_microphone.lower() == "list":
            print("Available microphones:")
            for i, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"  [{i}] {name}")
            return
        for i, name in enumerate(sr.Microphone.list_microphone_names()):
            if args.default_microphone in name:
                source = sr.Microphone(sample_rate=16000, device_index=i)
                print(f"Using microphone: {name}")
                break
        else:
            raise RuntimeError(f"No microphone matching '{args.default_microphone}'")
    else:
        source = sr.Microphone(sample_rate=16000)
        print("Using default microphone")

    # Load Whisper model
    whisper_model = args.model + ("" if args.non_english else ".en")
    print(f"Loading Whisper model: {whisper_model}")
    audio_model = WhisperModel(
        whisper_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="int8_float16",
        download_root="models"
    )


    # Load Pyannote speaker‑diarization pipeline with optimized settings
    print("Loading Pyannote diarization pipeline...")
    try:
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",  # Using the latest model
            use_auth_token=hf_token
        )
        # diarization_pipeline._inferences["_segmentation"].model = torch.quantization.quantize_dynamic(
        #     diarization_pipeline._inferences["_segmentation"].model,
        #     qconfig_spec={torch.nn.Linear, torch.nn.LSTM}, # Example set of modules
        #     dtype=torch.qint8
        # )

        # use torch dynamo
        diarization_pipeline._inferences["_segmentation"].model = torch.compile(diarization_pipeline._inferences["_segmentation"].model, mode="max-autotune")
        # diarization_pipeline._inferences["_embedding"] = torch.compile(diarization_pipeline._inferences["_embedding"], mode="max-autotune") # does not work because the optimized computation graph does not take the same attributes of the original model.



        print("Diarization pipeline loaded successfully")
    except Exception as e:
        print(f"Error loading diarization pipeline: {str(e)}")
        print("Falling back to transcription only mode")
        diarization_pipeline = None

    # Initialize transcription manager
    manager = TranscriptionManager(args, audio_model, diarization_pipeline)
    manager.diarization_window = args.diarization_window
    manager.sliding_window = args.sliding_window

    def record_callback(_, audio: sr.AudioData):
        try:
            raw = audio.get_raw_data()
            manager.transcribe_queue.put(raw)
            if diarization_pipeline is not None:
                manager.diarize_queue.put(raw)
        except Exception as e:
            print(f"\nError in record callback: {str(e)}")

    # Warm up mic and start listening
    print("\nAdjusting for ambient noise...")
    with source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    print(f"Adjusted energy threshold: {recognizer.energy_threshold}")
    
    print("\nStarting background listening...")
    recognizer.listen_in_background(
        source, record_callback, phrase_time_limit=args.record_timeout
    )

    print("\n✅ Models loaded. Listening...\n")
    print("Speak into your microphone to begin transcription.")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            now = datetime.now(timezone.utc)

            # Process transcription
            if not manager.transcribe_queue.empty():
                if manager.phrase_time and (now - manager.phrase_time) > timedelta(seconds=args.phrase_timeout):
                    manager.phrase_bytes = bytes()
                manager.phrase_time = now

                chunk = b""
                while not manager.transcribe_queue.empty():
                    chunk += manager.transcribe_queue.get()
                manager.phrase_bytes += chunk

                if len(manager.phrase_bytes) > 0:
                    audio_np = np.frombuffer(manager.phrase_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    manager.executor.submit(manager.process_transcription, audio_np)

            # Process diarization only if pipeline is available
            if diarization_pipeline is not None and not manager.diarize_queue.empty():
                buffer = b""
                while not manager.diarize_queue.empty():
                    buffer += manager.diarize_queue.get()
                if len(buffer) > 0:
                    manager.diarization_buffer.extend(buffer)
                    manager.executor.submit(manager.process_diarization)

            sleep(0.05)

    except KeyboardInterrupt:
        # Flush final buffer
        if manager.current_text:
            manager.utterances.append({
                "speaker": manager.current_speaker,
                "text": manager.current_text,
                "start": manager.current_start,
                "end": manager.current_end
            })
            print(f"\n{manager.current_speaker}: {manager.current_text}")

        # Sort chronologically and display with updated labels
        print("\n\nInterrupted. Final transcript (chronological):\n")
        for utt in sorted(manager.utterances, key=lambda x: x["start"] or 0):
            print(f"{utt['speaker']}: {utt['text']}")
        
        # Print speaker mapping for reference
        print("\nSpeaker mapping (for reference):")
        for pyannote_id, consistent_id in manager.get_speaker_mapping().items():
            print(f"Pyannote ID: {pyannote_id} -> Consistent ID: {consistent_id}")
        
        print("\nGoodbye!")
        return

if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    # ps.print_stats()
    # print(s.getvalue())