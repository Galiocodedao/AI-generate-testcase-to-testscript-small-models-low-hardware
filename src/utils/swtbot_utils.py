"""
SWTBot component utilities for test script generation
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from src import config

logger = logging.getLogger(__name__)

# Default SWTBot component definitions
DEFAULT_SWTBOT_COMPONENTS = {
    "button": [
        {"action": "click", "method": "bot.button(\"{name}\").click()", "priority": 1},
        {"action": "check", "method": "assertTrue(bot.button(\"{name}\").isEnabled())", "priority": 2},
    ],
    "text": [
        {"action": "enter", "method": "bot.text({index}).setText(\"{value}\")", "priority": 1},
        {"action": "clear", "method": "bot.text({index}).setText(\"\")", "priority": 2},
        {"action": "enterByLabel", "method": "bot.textWithLabel(\"{label}\").setText(\"{value}\")", "priority": 1},
    ],
    "checkbox": [
        {"action": "check", "method": "bot.checkBox(\"{name}\").select()", "priority": 1},
        {"action": "uncheck", "method": "bot.checkBox(\"{name}\").deselect()", "priority": 2},
        {"action": "toggle", "method": "bot.checkBox(\"{name}\").toggle()", "priority": 3},
    ],
    "combobox": [
        {"action": "select", "method": "bot.comboBox({index}).setSelection(\"{value}\")", "priority": 1},
        {"action": "selectByName", "method": "bot.comboBoxWithLabel(\"{label}\").setSelection(\"{value}\")", "priority": 1},
        {"action": "get", "method": "String value = bot.comboBox({index}).selection()", "priority": 2},
    ],
    "tree": [
        {"action": "select", "method": "bot.tree().select(\"{path}\")", "priority": 1},
        {"action": "expand", "method": "bot.tree().expandNode(\"{path}\")", "priority": 2},
        {"action": "collapse", "method": "bot.tree().collapseNode(\"{path}\")", "priority": 3},
        {"action": "doubleClick", "method": "bot.tree().select(\"{path}\").doubleClick()", "priority": 4},
    ],
    "menu": [
        {"action": "click", "method": "bot.menu(\"{name}\").click()", "priority": 1},
        {"action": "clickPath", "method": "bot.menu(\"{menuName}\").menu(\"{submenuName}\").click()", "priority": 2},
    ],
    "shell": [
        {"action": "close", "method": "bot.shell(\"{name}\").close()", "priority": 1},
        {"action": "activate", "method": "bot.shell(\"{name}\").activate()", "priority": 2},
    ],
    "view": [
        {"action": "open", "method": "bot.viewByTitle(\"{title}\").show()", "priority": 1},
        {"action": "close", "method": "bot.viewByTitle(\"{title}\").close()", "priority": 2},
    ],
    "editor": [
        {"action": "open", "method": "bot.editorByTitle(\"{title}\").show()", "priority": 1},
        {"action": "close", "method": "bot.editorByTitle(\"{title}\").close()", "priority": 2},
        {"action": "save", "method": "bot.editorByTitle(\"{title}\").save()", "priority": 3},
    ],
    "label": [
        {"action": "getText", "method": "String text = bot.label(\"{text}\").getText()", "priority": 1},
        {"action": "verify", "method": "assertEquals(\"{text}\", bot.label({index}).getText())", "priority": 2},
    ],
    "wait": [
        {"action": "condition", "method": "bot.waitUntil(new DefaultCondition() {\n\t@Override\n\tpublic boolean test() {\n\t\treturn {condition};\n\t}\n})", "priority": 1},
        {"action": "timeout", "method": "bot.waitUntil(condition, {timeout})", "priority": 2},
        {"action": "sleep", "method": "bot.sleep({timeoutMs})", "priority": 3},
    ],
}

def load_swtbot_components(custom_file_path: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load SWTBot component definitions from file or use defaults
    
    Args:
        custom_file_path: Path to custom component definitions JSON file
        
    Returns:
        Dictionary of SWTBot components and their actions
    """
    components = DEFAULT_SWTBOT_COMPONENTS.copy()
    
    # If custom file path is provided, try to load and merge with defaults
    if custom_file_path:
        try:
            with open(custom_file_path, 'r', encoding='utf-8') as f:
                custom_components = json.load(f)
            
            # Merge with defaults (custom overrides defaults)
            for component, actions in custom_components.items():
                components[component] = actions
            
            logger.info(f"Loaded custom SWTBot components from {custom_file_path}")
        except Exception as e:
            logger.warning(f"Failed to load custom SWTBot components: {str(e)}")
    
    return components

def save_swtbot_components(components: Dict[str, List[Dict[str, Any]]], file_path: str) -> None:
    """
    Save SWTBot component definitions to a file
    
    Args:
        components: Dictionary of SWTBot components
        file_path: Path to save the components to
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(components, f, indent=2)
        
        logger.info(f"Saved SWTBot components to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save SWTBot components: {str(e)}")
        raise

def get_action_for_component(component_type: str, action: str, components: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """
    Get the action definition for a component type and action
    
    Args:
        component_type: Type of UI component
        action: Action to perform
        components: Dictionary of SWTBot components
        
    Returns:
        Action definition or None if not found
    """
    if component_type in components:
        for action_def in components[component_type]:
            if action_def['action'] == action:
                return action_def
    
    return None
