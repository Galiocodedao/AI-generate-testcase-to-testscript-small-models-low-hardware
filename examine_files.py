"""
Final verification script with detailed error reporting
"""
import os
import sys

def examine_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            null_count = content.count(b'\x00')
            
            print(f"File: {file_path}")
            print(f"Size: {len(content)} bytes")
            print(f"Null bytes: {null_count}")
            print(f"First 30 bytes (hex): {content[:30].hex(' ')}")
            print("-" * 50)
            
            return null_count == 0
    except Exception as e:
        print(f"ERROR examining {file_path}: {str(e)}")
        print("-" * 50)
        return False

def main():
    base_dir = r'c:\Users\svphu\OneDrive\Documents\GitHub\AI-generate-testcase-to-testscript-small-models-low-hardware'
    files_to_check = [
        os.path.join(base_dir, 'src', '__init__.py'),
        os.path.join(base_dir, 'src', 'data', '__init__.py'),
        os.path.join(base_dir, 'src', 'models', '__init__.py'),
        os.path.join(base_dir, 'src', 'templates', '__init__.py'),
        os.path.join(base_dir, 'src', 'utils', '__init__.py'),
    ]
    
    print("DETAILED FILE EXAMINATION REPORT")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 50)
    
    all_clean = True
    for file_path in files_to_check:
        result = examine_file(file_path)
        all_clean = all_clean and result
    
    print("=" * 50)
    if all_clean:
        print("FINAL RESULT: ✓ All files are clean of null bytes!")
    else:
        print("FINAL RESULT: ✗ Some files may still have issues.")
    print("=" * 50)

if __name__ == "__main__":
    main()
