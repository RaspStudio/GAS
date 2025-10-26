import ast
import sys
import struct


def process_line(line):
    try:
        data = ast.literal_eval(line.strip())
        
        if isinstance(data, tuple) and len(data) >= 2:
            return [-1, data[0], data[1]]
        
        elif isinstance(data, set):
            elements = sorted(data)
            return [len(elements)] + elements
        
        else:
            raise ValueError(f"Unsupported: {type(data)}")
            
    except Exception as e:
        raise ValueError(f"Error on line: {line.strip()}, Detail: {str(e)}")

def convert_to_ivecs_binary(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'wb') as f_out:
            
            line_num = 0
            for line in f_in:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                
                try:
                    result = process_line(line)
                    result = [int(x) for x in result]
                    
                    for num in result:
                        f_out.write(struct.pack('i', num))
                        
                except ValueError as e:
                    print(f"Warning:  {line_num} line failed - {e}")
                except struct.error as e:
                    print(f"Warning:  {line_num} line pack failed - {e}")

        print(f"Success to binary: {output_file}")
        
    except FileNotFoundError:
        print(f"No such file: {input_file}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input text> <output ivecs>")
        print(f"Example: python {sys.argv[0]} input.txt output.ivecs")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_to_ivecs_binary(input_file, output_file)
    