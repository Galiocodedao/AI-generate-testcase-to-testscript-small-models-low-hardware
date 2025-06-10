"""
Test case parser module for handling different test case formats
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

import yaml

logger = logging.getLogger(__name__)

class TestCaseParser:
    """
    Parser for test cases in different formats (JSON, YAML, etc.)
    """
    
    @staticmethod
    def parse_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Parse a test case file and return the test cases
        
        Args:
            file_path: Path to the test case file
            
        Returns:
            List of test case dictionaries
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Test case file not found: {file_path}")
        
        # Determine file format based on extension
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.json':
            return TestCaseParser._parse_json(file_path)
        elif file_ext in ['.yaml', '.yml']:
            return TestCaseParser._parse_yaml(file_path)
        else:
            raise ValueError(f"Unsupported test case file format: {file_ext}")
    
    @staticmethod
    def _parse_json(file_path: Path) -> List[Dict[str, Any]]:
        """
        Parse a JSON test case file
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of test case dictionaries
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # List of test cases
                return data
            elif isinstance(data, dict):
                # Dictionary with test cases as values
                if 'testCases' in data:
                    return data['testCases']
                elif 'tests' in data:
                    return data['tests']
                else:
                    # Assume the dict itself is a single test case
                    return [data]
            else:
                raise ValueError(f"Invalid JSON test case format in {file_path}")
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def _parse_yaml(file_path: Path) -> List[Dict[str, Any]]:
        """
        Parse a YAML test case file
        
        Args:
            file_path: Path to the YAML file
            
        Returns:
            List of test case dictionaries
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Handle different YAML structures (similar to JSON)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                if 'testCases' in data:
                    return data['testCases']
                elif 'tests' in data:
                    return data['tests']
                else:
                    return [data]
            else:
                raise ValueError(f"Invalid YAML test case format in {file_path}")
        
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML file {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def from_text(text: str) -> List[Dict[str, Any]]:
        """
        Parse test cases from plain text (e.g., natural language descriptions)
        
        Args:
            text: Test case descriptions in plain text
            
        Returns:
            List of test case dictionaries
        """
        # This is a simple implementation for demonstration
        # In a real system, this would use more sophisticated NLP
        
        # Split by double newline to separate test cases
        test_case_texts = [tc.strip() for tc in text.split('\n\n') if tc.strip()]
        test_cases = []
        
        for i, test_text in enumerate(test_case_texts):
            lines = test_text.split('\n')
            
            # Assume first line is the title
            title = lines[0]
            
            # Extract description if present (lines that don't start with a number)
            description_lines = []
            step_lines = []
            
            for line in lines[1:]:
                line = line.strip()
                if line and not line[0].isdigit():
                    description_lines.append(line)
                elif line:
                    step_lines.append(line)
            
            description = ' '.join(description_lines)
            
            # Parse steps
            steps = []
            for step_line in step_lines:
                # Try to extract step number
                try:
                    step_id, step_text = step_line.split('.', 1)
                    step_id = int(step_id.strip())
                    step_text = step_text.strip()
                except (ValueError, IndexError):
                    # If parsing fails, just use the whole line
                    step_id = len(steps) + 1
                    step_text = step_line.strip()
                
                steps.append({
                    "id": step_id,
                    "description": step_text
                })
            
            # Create test case dictionary
            test_case = {
                "name": f"Test{i+1}_{title.replace(' ', '')}",
                "description": description or title,
                "steps": steps
            }
            
            test_cases.append(test_case)
        
        return test_cases
