"""
Script to verify that the imports work correctly after fixing the null bytes.
"""

def verify_imports():
    print("Attempting to import modules that had null bytes in their __init__.py files...")
    
    try:
        # Import the main module
        import src
        print("✓ Successfully imported 'src'")
        
        # Import submodules
        try:
            import src.data
            print("✓ Successfully imported 'src.data'")
        except ImportError as e:
            print(f"✗ Failed to import 'src.data': {e}")
        
        try:
            import src.models
            print("✓ Successfully imported 'src.models'")
        except ImportError as e:
            print(f"✗ Failed to import 'src.models': {e}")
        
        try:
            import src.templates
            print("✓ Successfully imported 'src.templates'")
        except ImportError as e:
            print(f"✗ Failed to import 'src.templates': {e}")
        
        try:
            import src.utils
            print("✓ Successfully imported 'src.utils'")
        except ImportError as e:
            print(f"✗ Failed to import 'src.utils': {e}")
            
        print("\nAll imports have been verified.")
        
    except ImportError as e:
        print(f"✗ Failed to import 'src': {e}")

if __name__ == "__main__":
    verify_imports()
