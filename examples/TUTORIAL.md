# AI Test Script Generator: Tutorial

This tutorial will guide you through using the AI Test Script Generator to convert test cases into SWTBot test scripts for Eclipse UI testing.

## Prerequisites

Before you start, make sure you have:

1. Eclipse IDE with SWTBot plugins installed
2. Python 3.8 or higher
3. This project cloned to your local machine

## Installation

1. Create a virtual environment and activate it:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the pre-trained models:

```bash
python -m src.utils.download_models
```

## Creating Test Cases

The AI Test Script Generator accepts test cases in JSON or YAML format. Here's a simple example:

```json
{
  "testCases": [
    {
      "name": "LoginTest",
      "description": "Test the login functionality",
      "steps": [
        {
          "id": 1,
          "description": "Click on the 'Login' button"
        },
        {
          "id": 2,
          "description": "Enter 'admin' in the username field"
        },
        {
          "id": 3,
          "description": "Enter 'password123' in the password field"
        },
        {
          "id": 4,
          "description": "Click on the 'Submit' button"
        },
        {
          "id": 5,
          "description": "Verify that the welcome message is displayed"
        }
      ]
    }
  ]
}
```

Save this file as `test_cases.json` in your project directory.

## Generating Test Scripts

To generate test scripts from your test cases:

```bash
python -m src.main --input path/to/test_cases.json --output path/to/output
```

This will create Java files in the output directory, one for each test case in your input file.

## Customizing SWTBot Components

You can customize the SWTBot components by creating a JSON file with your custom component definitions:

```json
{
  "custom_component": [
    {"action": "customAction", "method": "bot.customMethod(\"{param}\")", "priority": 1}
  ]
}
```

Then, pass the path to this file when initializing the generator:

```python
from src.models.generator import TestScriptGenerator

generator = TestScriptGenerator(custom_components_path="path/to/custom_components.json")
```

## Integration with Eclipse

1. Import the generated test scripts into your Eclipse project:
   - Right-click on the project > Import > General > File System
   - Select the directory containing the generated scripts
   - Click Finish

2. Add SWTBot dependencies to your project:
   - Right-click on the project > Properties > Java Build Path > Libraries
   - Add the SWTBot libraries

3. Run the tests:
   - Right-click on the test class > Run As > JUnit Test

## Example

Here's a complete example of how to use the AI Test Script Generator in your code:

```python
from src.models.generator import TestScriptGenerator
from src.data.parser import TestCaseParser

# Parse test cases
test_cases = TestCaseParser.parse_file("path/to/test_cases.json")

# Initialize generator
generator = TestScriptGenerator()

# Generate test scripts for each test case
for test_case in test_cases:
    script = generator.generate(test_case)
    
    # Save the script
    with open(f"{test_case['name']}.java", "w") as f:
        f.write(script)
```

For more examples, see the `src/examples.py` file in this repository.
