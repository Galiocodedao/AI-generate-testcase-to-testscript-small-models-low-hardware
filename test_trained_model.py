"""
Test script to verify the trained model works correctly
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.generator import TestScriptGenerator
from src.models.adapter import ModelAdapter
import json

def test_trained_model():
    """Test the trained model with sample test cases"""
    
    print("Testing the trained model...")
    
    # Load the trained model
    model_path = "C:/Users/svphu/OneDrive/Documents/GitHub/models/swtbot-fine-tuned"
    print(f"Loading model from: {model_path}")
    
    try:
        # Create model adapter with the trained model
        model = ModelAdapter(model_name=model_path)
        print("✓ Model loaded successfully")
        
        # Create test script generator with the trained model
        generator = TestScriptGenerator(model=model)
        print("✓ Generator initialized successfully")
        
        # Load test cases
        test_cases_path = "examples/test_cases.json"
        with open(test_cases_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        test_cases = test_data['testCases'][:1]  # Test with just one case
        
        # Generate a test script
        for test_case in test_cases:
            print(f"\n=== Generating script for: {test_case['name']} ===")
            script = generator.generate(test_case)
            
            # Save the generated script
            output_path = f"test_output_{test_case['name']}.java"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(script)
            
            print(f"✓ Script generated and saved to {output_path}")
            print(f"Script preview (first 300 chars):")
            print("-" * 50)
            print(script[:300] + "...")
            print("-" * 50)
        
        print("\n✅ Trained model test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_trained_model()
    sys.exit(0 if success else 1)
