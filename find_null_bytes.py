"""
Script to find null bytes in Python files
"""
import os
import sys

def check_file_for_null_bytes(file_path):
    try:
        print(f"Checking {file_path}...")
        with open(file_path, 'rb') as f:
            content = f.read()
            if b'\x00' in content:
                print(f"NULL BYTES FOUND in {file_path}")
                # Find position of null bytes
                positions = [i for i, byte in enumerate(content) if byte == 0]
                print(f"Null bytes at positions: {positions}")
                return True
    except Exception as e:
        print(f"Error checking {file_path}: {e}")
    return False

def scan_directory(directory):
    found = False
    file_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if check_file_for_null_bytes(file_path):
                    found = True
                file_count += 1
    print(f"Scanned {file_count} Python files")
    return found

if __name__ == "__main__":
    directory = "."
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    
    print(f"Scanning directory: {directory}")
    if not scan_directory(directory):
        print("No files with NULL bytes found")
