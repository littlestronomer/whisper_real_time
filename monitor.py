import time
import psutil
import pynvml

pynvml.nvmlInit()
gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

while True:
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().used / 1024**2
    gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(gpu).used / 1024**2
    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu).gpu

    print(f"CPU: {cpu:.1f}% | RAM: {mem:.1f} MB | GPU: {gpu_util}% | GPU Mem: {gpu_mem:.1f} MB")
    time.sleep(.1)
