"""
Demo script for the AI Test Script Generator
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to system path for imports
parent_dir = str(Path(__file__).parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from src.models.generator import TestScriptGenerator
    print("Successfully imported TestScriptGenerator")
except Exception as e:
    print(f"Error importing TestScriptGenerator: {str(e)}")
    sys.exit(1)

def main():
    """Main function for the demo"""
    print("Starting AI Test Script Generator demo...")
    
    # Get the current directory
    current_dir = Path(__file__).parent
    
    # Load test cases from the examples directory
    test_case_path = current_dir / "examples" / "test_cases.json"
    
    print(f"Looking for test cases at: {test_case_path}")
    
    if not test_case_path.exists():
        print(f"Test case file not found: {test_case_path}")
        return

    try:
        # Load test cases
        with open(test_case_path, 'r', encoding='utf-8') as f:
            test_cases_data = json.load(f)
        
        print("Successfully loaded test cases")
        
        # Get the test cases
        if isinstance(test_cases_data, dict) and 'testCases' in test_cases_data:
            test_cases = test_cases_data['testCases']
        else:
            test_cases = test_cases_data
        
        print(f"Found {len(test_cases)} test cases")
        
        # Initialize generator
        print("Initializing generator...")
        generator = TestScriptGenerator()
        
        # Create output directory if it doesn't exist
        output_dir = current_dir / "examples" / "generated"
        output_dir.mkdir(exist_ok=True)
        
        # Generate test scripts for each test case
        for i, test_case in enumerate(test_cases):
            test_name = test_case.get('name', f'Test{i+1}')
            print(f"Generating script for test case: {test_name}")
            
            # Generate the script
            script = generator.generate(test_case)
            
            # Save the script
            output_path = output_dir / f"{test_name}.java"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(script)
            
            print(f"Generated script saved to {output_path}")
        
        print("All test scripts generated successfully.")
    
    except Exception as e:
        print(f"Error generating scripts: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
