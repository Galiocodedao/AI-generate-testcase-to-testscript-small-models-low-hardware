"""
Unit tests for the Test Script Generator
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import pytest

from src.models.generator import TestScriptGenerator
from src.data.parser import TestCaseParser

class TestScriptGeneratorTests(unittest.TestCase):
    """Test cases for the TestScriptGenerator class"""
    
    def setUp(self):
        """Set up the test environment"""
        # Create a test generator
        self.generator = TestScriptGenerator()
        
        # Create test data
        self.test_case = {
            "name": "SimpleTest",
            "description": "A simple test case",
            "steps": [
                {
                    "id": 1,
                    "description": "Click on the 'OK' button"
                },
                {
                    "id": 2,
                    "description": "Enter 'Hello World' in the text field"
                }
            ]
        }
    
    def test_extract_ui_elements(self):
        """Test extracting UI elements from test steps"""
        ui_elements = self.generator._extract_ui_elements(self.test_case)
        
        # There should be elements extracted
        self.assertTrue(len(ui_elements) > 0)
        
        # Check button element
        button_elements = [e for e in ui_elements if e.get('type') == 'button']
        self.assertTrue(len(button_elements) > 0)
        button_element = button_elements[0]
        self.assertEqual(button_element.get('action'), 'click')
        self.assertEqual(button_element.get('name'), 'OK')
        
        # Check text element
        text_elements = [e for e in ui_elements if e.get('type') == 'text']
        self.assertTrue(len(text_elements) > 0)
        text_element = text_elements[0]
        self.assertEqual(text_element.get('action'), 'enter')
        self.assertEqual(text_element.get('value'), 'Hello World')
    
    def test_generate_code(self):
        """Test generating code from UI elements"""
        ui_elements = [
            {
                "type": "button",
                "name": "OK",
                "action": "click"
            },
            {
                "type": "text",
                "value": "Hello World",
                "action": "enter"
            }
        ]
        
        code_lines = self.generator._generate_code_from_elements(ui_elements)
        
        # There should be code lines generated
        self.assertTrue(len(code_lines) > 0)
        
        # Check button code
        button_code = [line for line in code_lines if 'button' in line and 'click' in line]
        self.assertTrue(len(button_code) > 0)
        
        # Check text code
        text_code = [line for line in code_lines if 'text' in line and 'setText' in line]
        self.assertTrue(len(text_code) > 0)
    
    def test_generate_script(self):
        """Test generating a full test script"""
        script = self.generator.generate(self.test_case)
        
        # The script should be a non-empty string
        self.assertTrue(isinstance(script, str))
        self.assertTrue(len(script) > 0)
        
        # The script should contain basic Java elements
        self.assertIn('public class SimpleTest', script)
        self.assertIn('@Test', script)
        
        # The script should contain the button and text code
        self.assertIn('button', script.lower())
        self.assertIn('text', script.lower())
    
    def test_parser_json(self):
        """Test parsing a JSON test case file"""
        # Create a temporary test case file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp.write(json.dumps({"testCases": [self.test_case]}).encode('utf-8'))
            tmp_path = tmp.name
        
        try:
            # Parse the test cases
            test_cases = TestCaseParser.parse_file(tmp_path)
            
            # There should be one test case
            self.assertEqual(len(test_cases), 1)
            
            # The test case should match our test data
            test_case = test_cases[0]
            self.assertEqual(test_case['name'], self.test_case['name'])
        finally:
            # Clean up
            os.unlink(tmp_path)


if __name__ == '__main__':
    unittest.main()
