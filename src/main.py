# -*- coding: utf-8 -*-

"""
Main entry point for the AI Test Script Generator
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.models.generator import TestScriptGenerator
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="AI Test Script Generator for Eclipse SWTBot")
    parser.add_argument("--input", type=str, required=True, help="Path to input test case file (JSON)")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory for generated scripts")
    parser.add_argument("--model", type=str, default="test-script-generator-small", 
                       help="Model to use for generation (default: test-script-generator-small)")
    parser.add_argument("--template", type=str, help="Custom template file to use")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(log_level)
    
    logger.info("Starting AI Test Script Generator")
    
    try:
        # Check if input file exists
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {args.input}")
            sys.exit(1)
        
        # Create output directory if it doesn't exist
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load test cases
        with open(input_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        
        # Initialize generator
        generator = TestScriptGenerator(model_name=args.model, template_path=args.template)
        
        # Generate test scripts
        for i, test_case in enumerate(test_cases):
            logger.info(f"Generating script for test case {i+1}/{len(test_cases)}")
            test_script = generator.generate(test_case)
            
            # Save test script
            test_name = test_case.get('name', f'TestCase{i+1}')
            output_file = output_path / f"{test_name}.java"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(test_script)
            
            logger.info(f"Generated script saved to {output_file}")
        
        logger.info(f"Successfully generated {len(test_cases)} test scripts")
    
    except Exception as e:
        logger.error(f"Error during script generation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
