"""
Script to verify that the null bytes have been removed from all identified files.
"""
import os

def verify_null_bytes_removed():
    base_dir = r'c:\Users\svphu\OneDrive\Documents\GitHub\AI-generate-testcase-to-testscript-small-models-low-hardware'
    files_to_check = [
        os.path.join(base_dir, 'src', '__init__.py'),
        os.path.join(base_dir, 'src', 'data', '__init__.py'),
        os.path.join(base_dir, 'src', 'models', '__init__.py'),
        os.path.join(base_dir, 'src', 'templates', '__init__.py'),
        os.path.join(base_dir, 'src', 'utils', '__init__.py'),
    ]
    
    print("Checking files for null bytes after fix...")
    all_clean = True
    
    for file_path in files_to_check:
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                null_count = content.count(b'\x00')
                
                if null_count > 0:
                    print(f"✗ {file_path} still contains {null_count} null bytes!")
                    all_clean = False
                else:
                    print(f"✓ {file_path} is clean - no null bytes found")
                    
                # Show file content (first 20 bytes in hex for inspection)
                print(f"   First 20 bytes: {content[:20].hex(' ')}")
                
        except Exception as e:
            print(f"✗ Error checking {file_path}: {str(e)}")
            all_clean = False
    
    if all_clean:
        print("\n✓ SUCCESS: All files are clean of null bytes!")
    else:
        print("\n✗ WARNING: Some files may still contain null bytes.")

if __name__ == "__main__":
    verify_null_bytes_removed()
