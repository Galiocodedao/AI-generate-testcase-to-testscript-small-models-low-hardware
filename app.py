#!/usr/bin/env python3
"""
AI Test Script Generator - Web Application
GTX 1660 Optimized Version

A Flask web application that provides a user-friendly interface for generating
SWTBot test scripts from test case descriptions using AI models optimized for
GTX 1660 hardware.

Features:
- GTX 1660 optimized model inference
- Real-time test script generation
- Modern, responsive UI
- Performance monitoring
- Memory efficient processing

Author: AI Assistant
License: MIT
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_cors import CORS
import json
import os
import sys
import time
import traceback
from datetime import datetime
import logging

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.generator import SWTBotGenerator
from utils.logger import setup_logger
from config import Config

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
CORS(app)

# Setup logging
logger = setup_logger(__name__)

# Global model instance (loaded on first use for faster startup)
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
    """Get or initialize the GTX 1660 optimized generator instance."""
    global _generator, _model_load_time
    
    if _generator is None:
        logger.info("Loading GTX 1660 optimized model...")
        start_time = time.time()
        
        try:
            _generator = SWTBotGenerator()
            # Try to load GTX 1660 optimized model first
            gtx_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'swtbot-gtx1660-optimized')
            if os.path.exists(gtx_model_path):
                _generator.load_model(gtx_model_path)
                logger.info(f"Loaded GTX 1660 optimized model from {gtx_model_path}")
            else:
                # Fallback to fine-tuned model
                default_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'swtbot-fine-tuned')
                if os.path.exists(default_model_path):
                    _generator.load_model(default_model_path)
                    logger.info(f"Loaded fallback model from {default_model_path}")
                else:
                    logger.info("Using default pre-trained model")
            
            _model_load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {_model_load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
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
        generator = get_generator()
        
        stats = {
            'model_load_time': _model_load_time,
            'generation_stats': _generation_stats.copy(),
            'model_info': {
                'type': 'GTX 1660 Optimized',
                'base_model': 'paraphrase-MiniLM-L3-v2',
                'optimization': 'Memory efficient, CUDA accelerated'
            },
            'system_info': {
                'gpu_available': generator.model.device.type == 'cuda' if hasattr(generator, 'model') else False,
                'device': str(generator.model.device) if hasattr(generator, 'model') else 'cpu'
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
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get configuration
    config = Config()
    
    print(f"""
    🚀 AI Test Script Generator - GTX 1660 Optimized
    
    📊 Features:
    • GTX 1660 optimized model inference
    • Real-time SWTBot test script generation
    • Modern, responsive web interface
    • Performance monitoring and stats
    • Memory efficient processing
    
    🌐 Starting server...
    """)
    
    # Run Flask app
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        threaded=True
    )