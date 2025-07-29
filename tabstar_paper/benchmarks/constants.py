import os

CPU_CORES = 8

EXISTING_CORES = os.cpu_count()
if EXISTING_CORES < CPU_CORES:
    print(f"❗ Warning: {CPU_CORES} CPU devices requested, but only {EXISTING_CORES} available.")
    CPU_CORES = EXISTING_CORES