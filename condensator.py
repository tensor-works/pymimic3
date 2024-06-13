import os
from pathlib import Path

def write_codebase_to_file(root_dir, output_file):
    with open(output_file, 'w') as outfile:
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(subdir, file)
                    print(file_path)
                    outfile.write(f"### FILE: {file_path} ###\n")
                    with open(file_path, 'r') as infile:
                        outfile.write(infile.read())
                        outfile.write("\n\n")

if __name__ == "__main__":
    root_directory = '.'  # Replace with the path to your codebase
    output_filename = 'condensed_codebase.txt'  # Replace with your desired output file name
    write_codebase_to_file(root_directory, output_filename)
    print(f"Codebase has been written to {output_filename}")