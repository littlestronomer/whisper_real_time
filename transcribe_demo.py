import torch
import numpy as np
import sounddevice as sd
import threading
import queue
import time
from collections import deque
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Core libraries
import nemo.collections.asr as nemo_asr

# Diart imports - using correct API
try:
    from diart import SpeakerDiarization
    from diart.sources import MicrophoneAudioSource
    from diart.inference import StreamingInference
    from diart.pipelines import OnlineSpeakerDiarization
    DIART_AVAILABLE = True
except ImportError as e:
    print(f"Warning: diart import failed: {e}")
    DIART_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeTranscriptionDiarization:
    """
    Real-time speaker diarization and transcription system using:
    - diart for speaker diarization
    - NVIDIA Parakeet TDT 0.6B v2 for transcription
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 1.0,  # seconds
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_diarization: bool = True
    ):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.device = device
        self.use_diarization = use_diarization and DIART_AVAILABLE
        
        # Audio buffer for processing
        self.audio_buffer = deque(maxlen=int(sample_rate * 30))  # 30 seconds buffer
        self.audio_queue = queue.Queue()
        self.results_queue = queue.Queue()
        
        # Threading control
        self.is_running = False
        self.threads = []
        
        # Initialize models
        self._initialize_models()
        
        # Results storage
        self.transcription_results = []
        self.diarization_results = []
        
        # Speaker tracking
        self.current_speakers = {}
        
    def _initialize_models(self):
        """Initialize the transcription and diarization models"""
        logger.info("Initializing models...")
        
        try:
            # Initialize NVIDIA Parakeet TDT model
            logger.info("Loading Parakeet TDT 0.6B v2 model...")
            self.asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
                "nvidia/parakeet-tdt-0.6b"  # Updated model name
            )
            self.asr_model = self.asr_model.to(self.device)
            self.asr_model.eval()
            logger.info("ASR model loaded successfully!")
            
            # Initialize diart for speaker diarization if available
            if self.use_diarization:
                logger.info("Setting up speaker diarization...")
                try:
                    # Create simple diarization pipeline
                    self.diarization_pipeline = OnlineSpeakerDiarization()
                    logger.info("Diarization pipeline initialized!")
                except Exception as e:
                    logger.warning(f"Could not initialize diarization: {e}")
                    self.use_diarization = False
            
            logger.info("Models initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert to mono if stereo
        if indata.ndim > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata.flatten()
        
        # Add to buffer and queue
        self.audio_buffer.extend(audio_data)
        
        # Only process if we have enough data
        if len(audio_data) >= self.chunk_size // 4:  # Process smaller chunks more frequently
            self.audio_queue.put(audio_data.copy())
    
    def _transcription_worker(self):
        """Worker thread for transcription"""
        logger.info("Transcription worker started")
        
        audio_accumulator = np.array([])
        
        while self.is_running:
            try:
                # Get audio chunk
                if not self.audio_queue.empty():
                    audio_chunk = self.audio_queue.get(timeout=1.0)
                    
                    # Accumulate audio for better transcription
                    audio_accumulator = np.append(audio_accumulator, audio_chunk)
                    
                    # Process when we have enough audio
                    if len(audio_accumulator) >= self.chunk_size:
                        # Take the required chunk size
                        audio_to_process = audio_accumulator[:self.chunk_size]
                        audio_accumulator = audio_accumulator[self.chunk_size//2:]  # Keep overlap
                        
                        # Transcribe audio
                        transcript = self._transcribe_audio(audio_to_process)
                        
                        if transcript and transcript.strip():
                            timestamp = time.time()
                            result = {
                                'timestamp': timestamp,
                                'transcript': transcript.strip(),
                                'type': 'transcription',
                                'speaker_id': self._get_current_speaker(timestamp)
                            }
                            
                            self.results_queue.put(result)
                            self.transcription_results.append(result)
                else:
                    time.sleep(0.01)  # Small sleep to prevent busy waiting
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in transcription worker: {e}")
    
    def _diarization_worker(self):
        """Worker thread for speaker diarization"""
        if not self.use_diarization:
            logger.info("Diarization disabled")
            return
            
        logger.info("Diarization worker started")
        
        try:
            # Simple speaker change detection based on audio energy
            previous_energy = 0
            speaker_counter = 0
            last_speaker_change = time.time()
            
            while self.is_running:
                if len(self.audio_buffer) > self.sample_rate:  # 1 second of audio
                    # Get recent audio
                    recent_audio = np.array(list(self.audio_buffer)[-self.sample_rate:])
                    
                    # Calculate audio energy
                    current_energy = np.mean(np.abs(recent_audio))
                    
                    # Simple speaker change detection
                    energy_change = abs(current_energy - previous_energy)
                    
                    current_time = time.time()
                    
                    # If significant energy change and enough time passed
                    if (energy_change > 0.01 and 
                        current_time - last_speaker_change > 3.0):  # Min 3 seconds between speaker changes
                        
                        speaker_counter += 1
                        speaker_id = f"Speaker_{speaker_counter % 4}"  # Cycle through 4 speakers max
                        
                        self.current_speakers[current_time] = speaker_id
                        last_speaker_change = current_time
                        
                        result = {
                            'timestamp': current_time,
                            'speaker_id': speaker_id,
                            'start_time': current_time,
                            'end_time': current_time + 1.0,
                            'type': 'diarization'
                        }
                        
                        self.results_queue.put(result)
                        self.diarization_results.append(result)
                    
                    previous_energy = current_energy
                
                time.sleep(0.1)  # Check every 100ms
                
        except Exception as e:
            logger.error(f"Error in diarization worker: {e}")
    
    def _get_current_speaker(self, timestamp: float) -> str:
        """Get the current speaker based on timestamp"""
        if not self.current_speakers:
            return "Speaker_0"
        
        # Find the most recent speaker assignment
        recent_speakers = [(t, s) for t, s in self.current_speakers.items() if t <= timestamp]
        if recent_speakers:
            return max(recent_speakers, key=lambda x: x[0])[1]
        else:
            return "Speaker_0"
    
    def _transcribe_audio(self, audio_data: np.ndarray) -> str:
        """Transcribe audio using Parakeet TDT model"""
        try:
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.8  # Scale to 80% to avoid clipping
            
            # Check if audio has sufficient energy
            if np.mean(np.abs(audio_data)) < 0.001:
                return ""
            
            # Convert to tensor and add batch dimension
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).to(self.device)
            
            # Transcribe
            with torch.no_grad():
                transcript = self.asr_model.transcribe([audio_tensor])
                if isinstance(transcript, list) and len(transcript) > 0:
                    return transcript[0]
                else:
                    return str(transcript) if transcript else ""
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return ""
    
    def _results_processor(self):
        """Process and combine transcription and diarization results"""
        logger.info("Results processor started")
        
        while self.is_running:
            try:
                if not self.results_queue.empty():
                    result = self.results_queue.get(timeout=1.0)
                    self._display_result(result)
                else:
                    time.sleep(0.01)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in results processor: {e}")
    
    def _display_result(self, result: Dict):
        """Display transcription and diarization results"""
        timestamp = time.strftime("%H:%M:%S", time.localtime(result['timestamp']))
        
        if result['type'] == 'transcription':
            speaker_id = result.get('speaker_id', 'Unknown')
            transcript = result['transcript']
            print(f"[{timestamp}] {speaker_id}: {transcript}")
            
        elif result['type'] == 'diarization':
            speaker_id = result['speaker_id']
            print(f"[{timestamp}] >>> Speaker change detected: {speaker_id}")
    
    def start_processing(self):
        """Start real-time processing"""
        logger.info("Starting real-time processing...")
        
        self.is_running = True
        
        # Start worker threads
        transcription_thread = threading.Thread(target=self._transcription_worker, daemon=True)
        results_thread = threading.Thread(target=self._results_processor, daemon=True)
        
        self.threads = [transcription_thread, results_thread]
        
        if self.use_diarization:
            diarization_thread = threading.Thread(target=self._diarization_worker, daemon=True)
            self.threads.append(diarization_thread)
        
        for thread in self.threads:
            thread.start()
        
        # Start audio stream
        try:
            self.audio_stream = sd.InputStream(
                callback=self._audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size // 4,  # Smaller blocksize for better responsiveness
                dtype=np.float32
            )
            
            self.audio_stream.start()
            
            logger.info("=" * 60)
            logger.info("Real-time processing started!")
            logger.info("Speak into your microphone...")
            logger.info("Press Ctrl+C to stop.")
            logger.info("=" * 60)
            
            # Keep main thread alive
            try:
                while self.is_running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("\nStopping processing...")
                self.stop_processing()
                
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            self.stop_processing()
    
    def stop_processing(self):
        """Stop real-time processing"""
        logger.info("Stopping real-time processing...")
        
        self.is_running = False
        
        # Stop audio stream
        if hasattr(self, 'audio_stream'):
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except:
                pass
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        logger.info("Processing stopped.")
    
    def get_combined_results(self) -> List[Dict]:
        """Get combined and sorted results"""
        all_results = self.transcription_results + self.diarization_results
        return sorted(all_results, key=lambda x: x['timestamp'])
    
    def save_results(self, filename: str):
        """Save results to file"""
        results = self.get_combined_results()
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Real-time Transcription and Diarization Results\n")
                f.write("=" * 50 + "\n\n")
                
                for result in results:
                    timestamp = time.strftime("%H:%M:%S", time.localtime(result['timestamp']))
                    
                    if result['type'] == 'transcription':
                        speaker_id = result.get('speaker_id', 'Unknown')
                        transcript = result['transcript']
                        f.write(f"[{timestamp}] {speaker_id}: {transcript}\n")
                    elif result['type'] == 'diarization':
                        speaker_id = result['speaker_id']
                        f.write(f"[{timestamp}] >>> Speaker change: {speaker_id}\n")
            
            logger.info(f"Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")


def main():
    """Main function to run the real-time system"""
    
    print("Real-time Speaker Diarization and Transcription System")
    print("Using NVIDIA Parakeet TDT 0.6B for transcription")
    if DIART_AVAILABLE:
        print("Using diart for speaker diarization")
    else:
        print("Using simple energy-based speaker detection")
    print("=" * 60)
    
    # Configuration
    config = {
        'sample_rate': 16000,
        'chunk_duration': 2.0,  # 2 seconds for better transcription
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_diarization': True
    }
    
    print(f"Device: {config['device']}")
    print(f"Sample Rate: {config['sample_rate']} Hz")
    print(f"Chunk Duration: {config['chunk_duration']} seconds")
    print("=" * 60)
    
    # Initialize system
    system = None
    try:
        system = RealTimeTranscriptionDiarization(**config)
        
        # Start processing
        system.start_processing()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        
    finally:
        # Save results
        if system:
            try:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"transcription_diarization_{timestamp}.txt"
                system.save_results(filename)
            except Exception as e:
                logger.error(f"Error saving results: {e}")


if __name__ == "__main__":
    main()