import os

def fix_null_bytes(file_path):
    """
    Remove null bytes from a file and save the clean content back to the same file.
    
    Args:
        file_path: Path to the file to fix
        
    Returns:
        tuple: (success, message) where success is a boolean and message is a descriptive string
    """
    try:
        # Read the file in binary mode
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Count null bytes
        null_count = content.count(b'\x00')
        
        if null_count == 0:
            return True, f"No null bytes found in {file_path}"
        
        # Remove null bytes
        clean_content = content.replace(b'\x00', b'')
        
        # Write the cleaned content back to the file
        with open(file_path, 'wb') as f:
            f.write(clean_content)
            
        return True, f"Fixed {file_path}: Removed {null_count} null bytes"
    
    except Exception as e:
        return False, f"Error fixing {file_path}: {str(e)}"

def main():
    # Using absolute paths for the files to fix
    base_dir = r'c:\Users\svphu\OneDrive\Documents\GitHub\AI-generate-testcase-to-testscript-small-models-low-hardware'
    files_to_fix = [
        os.path.join(base_dir, 'src', '__init__.py'),
        os.path.join(base_dir, 'src', 'data', '__init__.py'),
        os.path.join(base_dir, 'src', 'models', '__init__.py'),
        os.path.join(base_dir, 'src', 'templates', '__init__.py'),
        os.path.join(base_dir, 'src', 'utils', '__init__.py'),
    ]
    
    absolute_paths = files_to_fix
    
    print(f"Starting to fix {len(files_to_fix)} files with null bytes...")
    
    for file_path in absolute_paths:
        success, message = fix_null_bytes(file_path)
        print(message)
    
    print("\nAll files processed. Please verify that your imports work correctly now.")
    print("To verify, try importing modules that were previously failing.")

if __name__ == "__main__":
    main()
