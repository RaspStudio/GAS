import numpy as np
import os
import sys
import time
import struct

def generate_and_save_bmeta(output_file, num_vectors=1000000, value_range=(1, 1000)):
    
    try:
        print(f"Generating {num_vectors:,} Bmeta...")
        
        min_val, max_val = value_range
        data = np.random.randint(min_val, max_val+1, size=(num_vectors), dtype=np.int32)
        
        print(f"Writing {output_file}...")
        
        with open(output_file, 'wb') as f:
            for i in range(num_vectors):
                f.write(struct.pack('i', data[i]))
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"Usage: python {sys.argv[0]} <num_vectors> <output_path> <value_min> <value_max>")
        print(f"Example: python {sys.argv[0]} 1000000 meta.bmeta 0 1000")
        sys.exit(1)
        
    num_vectors = int(sys.argv[1])
    output_file = sys.argv[2]
    value_min = int(sys.argv[3])
    value_max = int(sys.argv[4])
    
    generate_and_save_bmeta(output_file, num_vectors, (value_min, value_max))
    
    print(f"Finished: {output_file}")    