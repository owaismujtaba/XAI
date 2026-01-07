
import h5py
import sys
import numpy as np

def inspect_hdf5(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            if 'trial_0000' in f:
                ids = f['trial_0000']['seq_class_ids'][:]
                # trimming trailing zeros
                trimmed = np.trim_zeros(ids, 'b')
                print(f"Trimmed length: {len(trimmed)}")
                print(f"Contains 0 in valid part? {0 in trimmed}")
                print(f"Trimmed sample: {trimmed}")
                
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_hdf5(sys.argv[1])
    else:
        print("Please provide a filepath")
