#!/usr/bin/env python3
"""
AI Test Script Generator - Simplified Web Application
GTX 1660 Optimized Version - Working Demo

A Flask web application that provides a user-friendly interface for generating
SWTBot test scripts from test case descriptions.

Author: AI Assistant
License: MIT
"""

from flask import Flask, render_template, request, jsonify, flash
from flask_cors import CORS
import json
import os
import time
import traceback
from datetime import datetime
import logging

# Import our simple generator
from simple_generator import SWTBotGenerator

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
_generator = None
_model_load_time = None
_generation_stats = {
    'total_requests': 0,
    'successful_generations': 0,
    'average_time': 0.0,
    'fastest_time': float('inf'),
    'slowest_time': 0.0
}

def get_generator():
    """Get or initialize the generator instance."""
    global _generator, _model_load_time
    
    if _generator is None:
        logger.info("Loading SWTBot generator...")
        start_time = time.time()
        
        try:
            _generator = SWTBotGenerator()
            _model_load_time = time.time() - start_time
            logger.info(f"Generator loaded successfully in {_model_load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to load generator: {str(e)}")
            raise
    
    return _generator

@app.route('/')
def index():
    """Main page with test case input form."""
    return render_template('index.html')

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    """Generate SWTBot test script from test case description."""
    if request.method == 'GET':
        return render_template('generate.html')
    
    try:
        # Get form data
        test_case_description = request.form.get('test_case', '').strip()
        test_name = request.form.get('test_name', '').strip()
        
        if not test_case_description:
            flash('Please provide a test case description.', 'error')
            return render_template('generate.html')
        
        # Update stats
        global _generation_stats
        _generation_stats['total_requests'] += 1
        
        # Generate test script
        start_time = time.time()
        generator = get_generator()
        
        # Generate with performance tracking
        logger.info(f"Generating test script for: {test_case_description[:50]}...")
        result = generator.generate_test_script(
            description=test_case_description,
            test_name=test_name or "GeneratedTest"
        )
        
        generation_time = time.time() - start_time
        
        # Update performance stats
        _generation_stats['successful_generations'] += 1
        total_time = _generation_stats['average_time'] * (_generation_stats['successful_generations'] - 1)
        _generation_stats['average_time'] = (total_time + generation_time) / _generation_stats['successful_generations']
        _generation_stats['fastest_time'] = min(_generation_stats['fastest_time'], generation_time)
        _generation_stats['slowest_time'] = max(_generation_stats['slowest_time'], generation_time)
        
        logger.info(f"Test script generated in {generation_time:.3f} seconds")
        
        # Prepare response data
        response_data = {
            'test_script': result,
            'generation_time': generation_time,
            'test_name': test_name or "GeneratedTest",
            'description': test_case_description,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return render_template('generate.html', **response_data)
        
    except Exception as e:
        logger.error(f"Error generating test script: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f'Error generating test script: {str(e)}', 'error')
        return render_template('generate.html')

@app.route('/api/generate', methods=['POST'])
def api_generate():
    """API endpoint for generating test scripts."""
    try:
        data = request.get_json()
        
        if not data or 'description' not in data:
            return jsonify({'error': 'Missing test case description'}), 400
        
        description = data['description'].strip()
        test_name = data.get('test_name', 'GeneratedTest').strip()
        
        if not description:
            return jsonify({'error': 'Test case description cannot be empty'}), 400
        
        # Generate test script
        start_time = time.time()
        generator = get_generator()
        
        result = generator.generate_test_script(
            description=description,
            test_name=test_name
        )
        
        generation_time = time.time() - start_time
        
        # Update stats
        global _generation_stats
        _generation_stats['total_requests'] += 1
        _generation_stats['successful_generations'] += 1
        
        return jsonify({
            'test_script': result,
            'generation_time': generation_time,
            'test_name': test_name,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def api_stats():
    """Get performance statistics."""
    try:        
        stats = {
            'model_load_time': _model_load_time,
            'generation_stats': _generation_stats.copy(),
            'model_info': {
                'type': 'GTX 1660 Optimized (Demo)',
                'base_model': 'Template-based Generator',
                'optimization': 'Memory efficient, fast generation'
            },
            'system_info': {
                'gpu_available': False,  # Using CPU for demo
                'device': 'cpu'
            }
        }
        
        # Clean up infinite values for JSON serialization
        if stats['generation_stats']['fastest_time'] == float('inf'):
            stats['generation_stats']['fastest_time'] = 0.0
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/examples')
def examples():
    """Show example test cases and generated scripts."""
    try:
        # Load example test cases
        examples_path = os.path.join(os.path.dirname(__file__), 'examples', 'test_cases.json')
        examples_data = []
        
        if os.path.exists(examples_path):
            with open(examples_path, 'r', encoding='utf-8') as f:
                examples_data = json.load(f)
        
        return render_template('examples.html', examples=examples_data)
        
    except Exception as e:
        logger.error(f"Error loading examples: {str(e)}")
        flash('Error loading examples', 'error')
        return render_template('examples.html', examples=[])

@app.route('/about')
def about():
    """About page with project information."""
    return render_template('about.html')

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {str(error)}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    print(f"""
    🚀 AI Test Script Generator - GTX 1660 Optimized (Demo)
    
    📊 Features:
    • Template-based test script generation
    • Real-time SWTBot test script creation
    • Modern, responsive web interface
    • Performance monitoring and stats
    • Memory efficient processing
    
    🌐 Starting server at http://127.0.0.1:5000
    """)
    
    # Run Flask app
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,
        threaded=True
    )
