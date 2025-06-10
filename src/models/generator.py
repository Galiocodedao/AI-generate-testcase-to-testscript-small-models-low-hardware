"""
Test Script Generator model for converting test cases to SWTBot scripts
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import torch
from jinja2 import Environment, FileSystemLoader

from src import config
from src.models.adapter import ModelAdapter
from src.utils.swtbot_utils import load_swtbot_components

logger = logging.getLogger(__name__)

class TestScriptGenerator:
    """
    A class for generating SWTBot test scripts from test case descriptions
    """
    
    def __init__(self, model_name: str = config.DEFAULT_MODEL_NAME, 
                 template_path: Optional[str] = None,
                 custom_components_path: Optional[str] = None):
        """
        Initialize the test script generator
        
        Args:
            model_name: Name of the model to use (must be defined in config.MODEL_CONFIG)
            template_path: Path to a custom template file (if None, use default)
            custom_components_path: Path to custom SWTBot components file
        """
        self.model_name = model_name
        self.model_config = config.MODEL_CONFIG.get(model_name)
        
        if not self.model_config:
            raise ValueError(f"Model {model_name} not found in configuration")
        
        # Load the model using the adapter
        try:
            logger.info(f"Loading model: {self.model_config['model_name']}")
            self.model = ModelAdapter(self.model_config)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
          # Set up template environment
        template_dir = config.TEMPLATE_DIR
        self.template_name = template_path or config.DEFAULT_TEMPLATE
        
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Load the predefined SWTBot actions/components for matching
        self.swtbot_components = load_swtbot_components(custom_components_path)
        
        logger.info("Test Script Generator initialized successfully")
    
    def _create_action_patterns(self) -> Dict[str, List[Dict]]:
        """
        Create regex patterns for identifying actions in test steps
        
        Returns:
            Dictionary of patterns for different UI actions
        """
        return {
            "button": [
                {
                    "pattern": r"(?i)click\s+(?:on\s+)?(?:the\s+)?(?:'|\")([^'\"]+)(?:'|\")\s+button",
                    "action": "click",
                    "extract": lambda m: {"name": m.group(1)}
                },
                {
                    "pattern": r"(?i)press\s+(?:the\s+)?(?:'|\")([^'\"]+)(?:'|\")\s+button",
                    "action": "click",
                    "extract": lambda m: {"name": m.group(1)}
                }
            ],
            "text": [
                {
                    "pattern": r"(?i)(?:enter|type)\s+(?:'|\")([^'\"]+)(?:'|\")\s+(?:in(?:to)?)\s+(?:the\s+)?(?:text(?:\s+field|\s+box)?|input)(?:\s+with\s+label\s+(?:'|\")([^'\"]+)(?:'|\"))?",
                    "action": "enter",
                    "extract": lambda m: {"value": m.group(1), "label": m.group(2) if m.group(2) else None}
                },
                {
                    "pattern": r"(?i)clear\s+(?:the\s+)?(?:text(?:\s+field|\s+box)?|input)",
                    "action": "clear",
                    "extract": lambda m: {}
                }
            ],
            "menu": [
                {
                    "pattern": r"(?i)click\s+(?:on\s+)?(?:the\s+)?(?:'|\")([^'\"]+)(?:'|\")\s+menu",
                    "action": "click",
                    "extract": lambda m: {"name": m.group(1)}
                },
                {
                    "pattern": r"(?i)select\s+(?:'|\")([^'\"]+)(?:'|\")\s+from\s+(?:the\s+)?(?:'|\")([^'\"]+)(?:'|\")\s+menu",
                    "action": "clickPath",
                    "extract": lambda m: {"submenuName": m.group(1), "menuName": m.group(2)}
                }
            ],
            "combobox": [
                {
                    "pattern": r"(?i)select\s+(?:'|\")([^'\"]+)(?:'|\")\s+from\s+(?:the\s+)?combo(?:\s+box)?(?:\s+with\s+label\s+(?:'|\")([^'\"]+)(?:'|\"))?",
                    "action": "select",
                    "extract": lambda m: {"value": m.group(1), "label": m.group(2) if m.group(2) else None}
                }
            ],
            "tree": [
                {
                    "pattern": r"(?i)select\s+(?:'|\")([^'\"]+)(?:'|\")\s+(?:in|from)\s+(?:the\s+)?tree",
                    "action": "select",
                    "extract": lambda m: {"path": m.group(1)}
                },
                {
                    "pattern": r"(?i)expand\s+(?:the\s+node\s+)?(?:'|\")([^'\"]+)(?:'|\")\s+in\s+(?:the\s+)?tree",
                    "action": "expand",
                    "extract": lambda m: {"path": m.group(1)}
                }
            ],
            "check": [
                {
                    "pattern": r"(?i)verify\s+(?:that\s+)?(?:the\s+)?(?:'|\")([^'\"]+)(?:'|\")\s+(?:is|appears|exists)",
                    "action": "verify",
                    "extract": lambda m: {"text": m.group(1)}
                }
            ],
            "wait": [
                {
                    "pattern": r"(?i)wait\s+(?:for\s+)?(\d+)\s+(?:second|seconds)",
                    "action": "sleep",
                    "extract": lambda m: {"timeoutMs": int(m.group(1)) * 1000}
                }
            ]
        }
    
    def _extract_ui_elements(self, test_case: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract UI elements and actions from a test case
        
        Args:
            test_case: Test case description
            
        Returns:
            List of UI elements and their actions
        """
        steps = test_case.get('steps', [])
        ui_elements = []
        
        for step in steps:
            step_text = step.get('description', '')
            elements = self._analyze_step(step_text)
            ui_elements.extend(elements)
        
        return ui_elements
      def _analyze_step(self, step_text: str) -> List[Dict[str, Any]]:
        """
        Analyze a test step to extract UI elements and actions
        
        Args:
            step_text: Description of the test step
            
        Returns:
            List of UI elements and their actions
        """
        elements = []
        
        # Create patterns for matching if not already created
        if not hasattr(self, 'action_patterns'):
            self.action_patterns = self._create_action_patterns()
        
        # Try to match step text with known patterns
        matched = False
        
        for element_type, patterns in self.action_patterns.items():
            for pattern_info in patterns:
                match = re.search(pattern_info['pattern'], step_text)
                if match:
                    # Extract data based on the pattern
                    extracted_data = pattern_info['extract'](match)
                    
                    # Create element info
                    element_info = {
                        "type": element_type,
                        "action": pattern_info['action'],
                        **extracted_data
                    }
                    
                    elements.append(element_info)
                    matched = True
                    # Don't break here to allow multiple matches in one step
        
        # If no pattern matched, try semantic analysis using the model
        if not matched and hasattr(self, 'model'):
            semantic_elements = self._analyze_step_semantically(step_text)
            elements.extend(semantic_elements)
        
        return elements
        
    def _analyze_step_semantically(self, step_text: str) -> List[Dict[str, Any]]:
        """
        Analyze a step using semantic matching when pattern matching fails
        
        Args:
            step_text: Description of the test step
            
        Returns:
            List of UI elements and their actions
        """
        elements = []
        
        # This is a simplified example of semantic matching
        # In a real system, this would be more sophisticated
        
        # Create action templates for semantic matching if not already created
        if not hasattr(self, 'action_templates'):
            self.action_templates = [
                {"text": "Click on a button", "type": "button", "action": "click"},
                {"text": "Press a button", "type": "button", "action": "click"},
                {"text": "Enter text in a field", "type": "text", "action": "enter"},
                {"text": "Select an item from dropdown", "type": "combobox", "action": "select"},
                {"text": "Check a checkbox", "type": "checkbox", "action": "check"},
                {"text": "Select an item from tree", "type": "tree", "action": "select"},
                {"text": "Click on a menu item", "type": "menu", "action": "click"},
                {"text": "Wait for a condition", "type": "wait", "action": "condition"},
                {"text": "Verify that element exists", "type": "check", "action": "verify"},
            ]
        
        # Find text between quotes which might be element names/values
        names_values = re.findall(r"['\"]([^'\"]+)['\"]", step_text)
        
        # Calculate similarity between step and action templates
        template_texts = [template["text"] for template in self.action_templates]
        similarities = self.model.batch_similarity(step_text, template_texts)
        
        # Find best matching template
        max_sim_idx = similarities.index(max(similarities))
        best_template = self.action_templates[max_sim_idx]
        
        # Only use semantic match if similarity is above threshold
        if similarities[max_sim_idx] > 0.5:
            element = {
                "type": best_template["type"],
                "action": best_template["action"]
            }
            
            # Add name/value from extracted quotes if available
            if names_values:
                if best_template["type"] == "button" and best_template["action"] == "click":
                    element["name"] = names_values[0]
                elif best_template["type"] == "text" and best_template["action"] == "enter":
                    element["value"] = names_values[0]
                elif best_template["type"] == "combobox" and best_template["action"] == "select":
                    element["value"] = names_values[0]
                elif best_template["type"] == "menu" and best_template["action"] == "click":
                    element["name"] = names_values[0]
            
            elements.append(element)
        
        return elements
      def _generate_code_from_elements(self, ui_elements: List[Dict[str, Any]]) -> List[str]:
        """
        Generate SWTBot code from extracted UI elements
        
        Args:
            ui_elements: List of UI elements and their actions
            
        Returns:
            List of code lines
        """
        code_lines = []
        
        for element in ui_elements:
            element_type = element.get('type')
            action = element.get('action')
            
            # Find matching SWTBot component and action
            if element_type in self.swtbot_components:
                component_actions = self.swtbot_components[element_type]
                matching_action = next((ca for ca in component_actions if ca['action'] == action), None)
                
                if matching_action:
                    # Replace placeholders in the method template
                    method = matching_action['method']
                    
                    # Check if all required parameters are available
                    all_params_available = True
                    missing_params = []
                    
                    # Extract required parameters from the method template
                    required_params = re.findall(r"\{(\w+)\}", method)
                    
                    for param in required_params:
                        if param not in element and param != 'index':
                            all_params_available = False
                            missing_params.append(param)
                    
                    # If parameters are missing, try to fill defaults
                    if not all_params_available:
                        # Add default index if needed
                        if 'index' in required_params and 'index' not in element:
                            element['index'] = '0'  # Default to first element
                        
                        # If still missing params, skip this element
                        if missing_params and any(p != 'index' for p in missing_params):
                            logger.warning(f"Missing parameters {missing_params} for element {element}")
                            continue
                    
                    # Replace placeholders in the method template
                    for key, value in element.items():
                        if key not in ['type', 'action'] and value is not None:
                            method = method.replace(f"{{{key}}}", str(value))
                    
                    # Default index if not specified
                    if '{index}' in method:
                        method = method.replace('{index}', '0')
                    
                    # Add comments for clarity
                    comment = f"// {element.get('action', '').capitalize()} {element_type}"
                    if 'name' in element:
                        comment += f" '{element['name']}'"
                    elif 'value' in element:
                        comment += f" with value '{element['value']}'"
                    
                    code_lines.append(comment)
                    code_lines.append(method)
            else:
                logger.warning(f"Unknown element type: {element_type}")
        
        return code_lines
    
    def generate(self, test_case: Dict[str, Any]) -> str:
        """
        Generate a SWTBot test script from a test case description
        
        Args:
            test_case: Test case description
            
        Returns:
            Generated test script as a string
        """
        logger.info(f"Generating test script for: {test_case.get('name', 'Unknown test')}")
        
        # Extract test case metadata
        test_name = test_case.get('name', 'GeneratedTest')
        test_description = test_case.get('description', '')
        
        # Extract UI elements from test steps
        ui_elements = self._extract_ui_elements(test_case)
        
        # Generate code for each UI element
        code_lines = self._generate_code_from_elements(ui_elements)
        
        # Load the template
        try:
            template = self.jinja_env.get_template(self.template_name)
        except Exception as e:
            logger.error(f"Error loading template {self.template_name}: {str(e)}")
            # Use a basic fallback template
            from jinja2 import Template
            template_content = """
package org.eclipse.swtbot.test;

import static org.junit.Assert.*;
import org.eclipse.swtbot.eclipse.finder.SWTBotEclipseTestCase;
import org.eclipse.swtbot.swt.finder.junit.SWTBotJunit4ClassRunner;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * {{ test_description }}
 */
@RunWith(SWTBotJunit4ClassRunner.class)
public class {{ test_name }} extends SWTBotEclipseTestCase {
    
    @Test
    public void {{ test_method_name }}() {
        {% for code_line in code_lines %}
        {{ code_line }};
        {% endfor %}
    }
}
"""
            template = Template(template_content)
        
        # Render the template with the test data
        test_method_name = test_name.replace("Test", "").lower()
        rendered_script = template.render(
            test_name=test_name,
            test_description=test_description,
            test_method_name=test_method_name,
            code_lines=code_lines
        )
        
        return rendered_script
