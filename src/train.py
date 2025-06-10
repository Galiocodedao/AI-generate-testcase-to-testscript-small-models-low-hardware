# -*- coding: utf-8 -*-
"""
Model training module for AI Test Script Generator
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

from src import config
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Trainer for fine-tuning models to generate test scripts
    """
    
    def __init__(self, model_name: str = "paraphrase-MiniLM-L3-v2", 
                 output_model_name: str = "test-script-generator-fine-tuned",
                 device: str = "cpu"):
        """
        Initialize the model trainer
        
        Args:
            model_name: Base model to fine-tune (default: paraphrase-MiniLM-L3-v2)
            output_model_name: Name to save the fine-tuned model
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.output_model_name = output_model_name
        self.device = device
        self.model = None
        self.train_examples = []
        self.eval_examples = []
        
        # Create output directory
        self.output_dir = Path(config.MODELS_DIR) / output_model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ModelTrainer initialized with base model {model_name}")
    
    def load_base_model(self):
        """Load the base model for fine-tuning"""
        try:
            logger.info(f"Loading base model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)
            logger.info(f"Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise
    
    def prepare_training_data(self, training_data_path: Union[str, Path], 
                             train_ratio: float = 0.8):
        """
        Prepare training data from JSON file
        
        Args:
            training_data_path: Path to training data file
            train_ratio: Ratio of training to evaluation data
        """
        logger.info(f"Loading training data from {training_data_path}")
        
        try:
            with open(training_data_path, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
            
            pairs = []
            # Extract training pairs from the data
            if "pairs" in training_data:
                pairs = training_data["pairs"]
            elif "examples" in training_data:
                pairs = training_data["examples"]
            
            # Shuffle the data
            import random
            random.shuffle(pairs)
            
            # Split into train and eval sets
            split_idx = int(len(pairs) * train_ratio)
            train_pairs = pairs[:split_idx]
            eval_pairs = pairs[split_idx:]
            
            # Create train examples
            for pair in train_pairs:
                if "text1" in pair and "text2" in pair and "similarity" in pair:
                    self.train_examples.append(
                        InputExample(texts=[pair["text1"], pair["text2"]], label=pair["similarity"])
                    )
            
            # Create evaluation examples
            for pair in eval_pairs:
                if "text1" in pair and "text2" in pair and "similarity" in pair:
                    self.eval_examples.append(
                        InputExample(texts=[pair["text1"], pair["text2"]], label=pair["similarity"])
                    )
            
            logger.info(f"Created {len(self.train_examples)} training examples and {len(self.eval_examples)} evaluation examples")
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise
    
    def train(self, epochs: int = 4, batch_size: int = 16, 
             learning_rate: float = 2e-5, evaluation_steps: int = 1000):
        """
        Train the model on the prepared data
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            evaluation_steps: How often to evaluate the model
        """
        if self.model is None:
            logger.warning("Model not loaded. Loading base model first.")
            self.load_base_model()
        
        if not self.train_examples:
            raise ValueError("No training examples available. Call prepare_training_data first.")
        
        logger.info(f"Starting training with {len(self.train_examples)} examples for {epochs} epochs")
        
        # Configure training
        train_dataloader = DataLoader(self.train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Configure evaluation
        if self.eval_examples:
            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(self.eval_examples, name='eval')
        else:
            evaluator = None
        
        # Warmup steps
        warmup_steps = int(len(train_dataloader) * 0.1) 
        
        # Train the model
        try:
            start_time = time.time()
            
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                evaluator=evaluator,
                evaluation_steps=evaluation_steps,
                warmup_steps=warmup_steps,
                output_path=str(self.output_dir),
                optimizer_params={'lr': learning_rate}
            )
            
            end_time = time.time()
            training_time = end_time - start_time
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Model saved to {self.output_dir}")
            
            # Save training configuration
            with open(self.output_dir / "training_config.json", 'w') as f:
                json.dump({
                    "base_model": self.model_name,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "training_examples": len(self.train_examples),
                    "evaluation_examples": len(self.eval_examples),
                    "training_time_seconds": training_time,
                    "date_trained": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def create_training_data_from_examples(self, 
                                         test_cases_path: Union[str, Path],
                                         swtbot_references_path: Union[str, Path],
                                         output_path: Union[str, Path]):
        """
        Create training data from test cases and SWTBot references
        
        Args:
            test_cases_path: Path to test cases file
            swtbot_references_path: Path to SWTBot reference file
            output_path: Path to save the training data
        """
        logger.info("Creating training data from examples")
        
        try:
            # Load test cases
            with open(test_cases_path, 'r', encoding='utf-8') as f:
                test_cases_data = json.load(f)
            
            # Load SWTBot references
            with open(swtbot_references_path, 'r', encoding='utf-8') as f:
                swtbot_references = json.load(f)
            
            # Extract pairs
            pairs = []
            
            # Process test cases
            test_cases = []
            if "testCases" in test_cases_data:
                test_cases = test_cases_data["testCases"]
            else:
                test_cases = test_cases_data
            
            for test_case in test_cases:
                if "steps" in test_case:
                    for step in test_case["steps"]:
                        description = step.get("description", "")
                        
                        # Find matching SWTBot actions
                        for component_type, actions in swtbot_references.items():
                            for action in actions:
                                # Create keywords to search for
                                keywords = [component_type]
                                if "keywords" in action:
                                    keywords.extend(action["keywords"])
                                
                                # Check if any keyword is in the description
                                if any(keyword.lower() in description.lower() for keyword in keywords):
                                    # Create a pair
                                    pairs.append({
                                        "text1": description,
                                        "text2": action.get("description", ""),
                                        "similarity": 0.9 
                                    })
                                else:
                                    # Create negative example (low similarity)
                                    if np.random.random() < 0.1:  # Only add some negative examples
                                        pairs.append({
                                            "text1": description,
                                            "text2": action.get("description", ""),
                                            "similarity": 0.1
                                        })
            
            # Save the pairs
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"pairs": pairs}, f, indent=2)
            
            logger.info(f"Created {len(pairs)} training pairs and saved to {output_path}")
        
        except Exception as e:
            logger.error(f"Error creating training data: {str(e)}")
            raise

def main():
    """Main function for training"""
    # Set up logging
    setup_logger(logging.INFO)
    
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Train AI Test Script Generator model")
    parser.add_argument("--data", type=str, required=True, help="Path to training data file")
    parser.add_argument("--model", type=str, default="paraphrase-MiniLM-L3-v2", help="Base model name")
    parser.add_argument("--output", type=str, default="test-script-generator-fine-tuned", help="Output model name")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--create-data", action="store_true", help="Create training data from examples")
    parser.add_argument("--test-cases", type=str, help="Path to test cases file (for --create-data)")
    parser.add_argument("--swtbot-refs", type=str, help="Path to SWTBot references file (for --create-data)")
    args = parser.parse_args()
    
    # Create trainer
    trainer = ModelTrainer(model_name=args.model, output_model_name=args.output)
    
    # Create training data if requested
    if args.create_data:
        if not args.test_cases or not args.swtbot_refs:
            logger.error("--test-cases and --swtbot-refs required with --create-data")
            return 1
        
        trainer.create_training_data_from_examples(
            test_cases_path=args.test_cases,
            swtbot_references_path=args.swtbot_refs,
            output_path=args.data
        )
        return 0
    
    # Train model
    trainer.load_base_model()
    trainer.prepare_training_data(args.data)
    trainer.train(epochs=args.epochs, batch_size=args.batch_size)
    
    logger.info("Training completed successfully")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
