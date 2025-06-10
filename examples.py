"""
Example of using the AI Test Script Generator
"""

import json
import logging
import os
import traceback
from pathlib import Path

from src.models.generator import TestScriptGenerator
from src.utils.logger import setup_logger

# Set up logging
setup_logger(logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function for the example"""
    # Set current directory to the script's directory
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
      # Load test cases from the examples directory
    test_case_path = Path('examples/test_cases.json')
    
    if not test_case_path.exists():
        logger.error(f"Test case file not found: {test_case_path}")
        return
    
    try:
        # Load test cases
        with open(test_case_path, 'r', encoding='utf-8') as f:
            test_cases_data = json.load(f)
        
        # Get the test cases
        if isinstance(test_cases_data, dict) and 'testCases' in test_cases_data:
            test_cases = test_cases_data['testCases']
        else:
            test_cases = test_cases_data
        
        # Initialize generator
        generator = TestScriptGenerator()
          # Create output directory if it doesn't exist
        output_dir = Path('examples/generated')
        output_dir.mkdir(exist_ok=True)
        
        # Generate test scripts for each test case
        for i, test_case in enumerate(test_cases):
            logger.info(f"Generating script for test case: {test_case.get('name', f'Test{i+1}')}")
            
            # Generate the script
            script = generator.generate(test_case)
            
            # Save the script
            output_path = output_dir / f"{test_case.get('name', f'Test{i+1}')}.java"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(script)
            
            logger.info(f"Generated script saved to {output_path}")
        
        logger.info("All test scripts generated successfully.")
    
    except Exception as e:
        logger.error(f"Error generating scripts: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
