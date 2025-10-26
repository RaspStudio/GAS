import numpy as np
import os
import sys
import time
import random

def generate_range(meta_min, meta_max, selectivity):
    length = int((meta_max - meta_min) * selectivity / 100)
    range_start = np.random.randint(meta_min, meta_max-length+1, dtype=np.int32)
    return (int(range_start), int(range_start + length))

def range_overlap(r1, r2) -> float:
    start1, end1 = r1
    start2, end2 = r2
    
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    if overlap_start >= overlap_end:
        return 0.0
    
    overlap_length = overlap_end - overlap_start
    
    smaller_length = min(end1 - start1, end2 - start2)
    
    return overlap_length / smaller_length

def generate_query_meta_range(output_file, n_query, selectivity, meta_min, meta_max, variety, min_overlap: float = 0.0):
    
    try:
        print(f"Generating {n_query:,} Query Range...")
        
        range_list = []
        while len(range_list) < (variety if variety != 0 else n_query):
            new_range = generate_range(meta_min, meta_max, selectivity)
            if min_overlap == 0 or all(range_overlap(new_range, existing_range) >= min_overlap for existing_range in range_list):
                range_list.append(new_range)

        while len(range_list) < n_query:
            range_list += range_list

        print(f"Writing {output_file}...")
        
        with open(output_file, 'w') as f:
            for i in range(n_query):
                f.write(str(range_list[i])+'\n')
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def generate_query_meta_set(output_file, n_query, selectivity, meta_min, meta_max, variety):
    
    try:
        print(f"Generating {n_query:,} Query Set...")
        accepted_vals_size = int((meta_max - meta_min) * selectivity / 100)
        set_list = [set(random.sample(range(meta_min, meta_max + 1), accepted_vals_size)) for _ in range(variety if variety != 0 else n_query)]

        while len(set_list) < n_query:
            set_list += set_list
        
        print(f"Writing {output_file}...")
        
        with open(output_file, 'w') as f:
            for i in range(n_query):
                f.write(str(set_list[i])+'\n')
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":

    if (len(sys.argv) > 0) and (len(sys.argv) < 7 or len(sys.argv) > 9):
        print(f"Usage: python {sys.argv[0]} <num_query> <output_path> <range/set> <selectivity*100> <range_min> <range_max> [variety] [min_overlap]")
        print(f"Example: python {sys.argv[0]} 10000 query_meta.txt set 10 0 1000 [1]")
        sys.exit(1)
        
    n_query = int(sys.argv[1])
    output_file = sys.argv[2]
    filter_type = sys.argv[3]
    selectivity = int(sys.argv[4])
    meta_min = int(sys.argv[5])
    meta_max = int(sys.argv[6]) 
    variety = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    overlap = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0
    
    if filter_type == 'range':
        generate_query_meta_range(output_file, n_query, selectivity, meta_min, meta_max, variety, overlap)
    elif filter_type == 'set':
        generate_query_meta_set(output_file, n_query, selectivity, meta_min, meta_max, variety)
    else:
        print(f"Unsupported filter type: {filter_type}")
        sys.exit(1)
    
    print(f"Finished: {output_file}")    