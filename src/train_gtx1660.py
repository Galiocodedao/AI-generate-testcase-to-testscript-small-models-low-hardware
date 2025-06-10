"""
Optimized training script for GTX 1660 GPU
"""

import torch
import logging
import json
import time
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

from src import config
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)

class GTX1660Trainer:
    """
    Optimized trainer for GTX 1660 GPU (6GB VRAM)
    """
    
    def __init__(self, 
                 model_name: str = "paraphrase-MiniLM-L3-v2",
                 output_model_name: str = "swtbot-gtx1660-optimized",
                 use_gpu: bool = True):
        """
        Initialize trainer optimized for GTX 1660
        
        Args:
            model_name: Base model to fine-tune
            output_model_name: Name for the fine-tuned model
            use_gpu: Whether to use GPU if available
        """
        self.model_name = model_name
        self.output_model_name = output_model_name
        
        # GPU optimization for GTX 1660
        self.device = self._setup_device(use_gpu)
        self._setup_memory_optimization()
        
        self.model = None
        self.train_examples = []
        self.eval_examples = []
        
        # Create output directory
        self.output_dir = Path(config.MODELS_DIR) / output_model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"GTX1660Trainer initialized - Device: {self.device}")
        logger.info(f"GPU Memory Available: {self._get_gpu_memory()}")
    
    def _setup_device(self, use_gpu: bool) -> str:
        """Setup optimal device for GTX 1660"""
        if use_gpu and torch.cuda.is_available():
            device = "cuda"
            # Check if it's GTX 1660 series
            gpu_name = torch.cuda.get_device_name(0)
            if "1660" in gpu_name:
                logger.info(f"Detected GTX 1660: {gpu_name}")
            else:
                logger.info(f"GPU detected: {gpu_name}")
            return device
        else:
            logger.info("Using CPU (GPU not available or disabled)")
            return "cpu"
    
    def _setup_memory_optimization(self):
        """Setup memory optimization for GTX 1660"""
        if self.device == "cuda":
            # Enable memory growth
            torch.cuda.empty_cache()
            
            # Set memory fraction (use ~80% of 6GB = ~4.8GB)
            torch.cuda.set_per_process_memory_fraction(0.8)
            
            # Enable mixed precision training
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            logger.info("GPU memory optimization enabled for GTX 1660")
    
    def _get_gpu_memory(self) -> str:
        """Get GPU memory information"""
        if self.device == "cuda":
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            return f"Total: {total_memory:.1f}GB, Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB"
        return "CPU mode"
    
    def load_base_model(self):
        """Load and optimize model for GTX 1660"""
        try:
            logger.info(f"Loading base model: {self.model_name}")
              # Clear any existing GPU memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Memory after loading: {self._get_gpu_memory()}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            if "out of memory" in str(e).lower():
                logger.error("GPU out of memory! Try reducing batch size or using CPU")
            raise
    
    def prepare_training_data(self, training_data_path: Union[str, Path], 
                             train_ratio: float = 0.8):
        """Prepare training data with memory optimization"""
        logger.info(f"Loading training data from {training_data_path}")
        
        try:
            with open(training_data_path, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
            
            pairs = training_data.get("pairs", [])
            
            # Shuffle and limit data if needed for GTX 1660
            import random
            random.shuffle(pairs)
            
            # Limit training data if GPU memory is low
            max_examples = 1000  # Adjust based on available memory
            if len(pairs) > max_examples:
                logger.warning(f"Limiting training data to {max_examples} examples for GTX 1660")
                pairs = pairs[:max_examples]
            
            # Split data
            split_idx = int(len(pairs) * train_ratio)
            train_pairs = pairs[:split_idx]
            eval_pairs = pairs[split_idx:]
            
            # Create examples
            self.train_examples = [
                InputExample(texts=[p["text1"], p["text2"]], label=float(p["similarity"]))
                for p in train_pairs if all(k in p for k in ["text1", "text2", "similarity"])
            ]
            
            self.eval_examples = [
                InputExample(texts=[p["text1"], p["text2"]], label=float(p["similarity"]))
                for p in eval_pairs if all(k in p for k in ["text1", "text2", "similarity"])
            ]
            
            logger.info(f"Created {len(self.train_examples)} training and {len(self.eval_examples)} eval examples")
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise
    
    def train_optimized(self, 
                       epochs: int = 3,
                       batch_size: int = 16,  # Optimized for GTX 1660
                       learning_rate: float = 2e-5,
                       warmup_steps_ratio: float = 0.1,
                       save_steps: int = 500,
                       eval_steps: int = 250):
        """
        Optimized training for GTX 1660
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size (16 is optimal for GTX 1660)
            learning_rate: Learning rate
            warmup_steps_ratio: Ratio of total steps for warmup
            save_steps: Steps between model saves
            eval_steps: Steps between evaluations
        """
        if self.model is None:
            self.load_base_model()
        
        if not self.train_examples:
            raise ValueError("No training examples. Call prepare_training_data first.")
        
        # Adjust batch size based on available memory
        if self.device == "cuda":
            available_memory = torch.cuda.get_device_properties(0).total_memory
            if available_memory < 6 * 1024**3:  # Less than 6GB
                batch_size = min(batch_size, 8)
                logger.warning(f"Reduced batch size to {batch_size} due to limited GPU memory")
        
        logger.info(f"Starting optimized training for GTX 1660:")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Learning rate: {learning_rate}")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - Training examples: {len(self.train_examples)}")
        
        try:
            # Clear memory before training
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
              # Setup data loader with optimized settings for Windows
            train_dataloader = DataLoader(
                self.train_examples, 
                shuffle=True, 
                batch_size=batch_size,
                num_workers=0,  # Use 0 workers for Windows compatibility
                pin_memory=True if self.device == "cuda" else False
            )
            
            # Setup loss function
            train_loss = losses.CosineSimilarityLoss(self.model)
            
            # Setup evaluator
            evaluator = None
            if self.eval_examples:
                evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                    self.eval_examples, 
                    name='gtx1660_eval'
                )
            
            # Calculate warmup steps
            total_steps = len(train_dataloader) * epochs
            warmup_steps = int(total_steps * warmup_steps_ratio)
            
            logger.info(f"Training configuration:")
            logger.info(f"  - Total steps: {total_steps}")
            logger.info(f"  - Warmup steps: {warmup_steps}")
            logger.info(f"  - Memory before training: {self._get_gpu_memory()}")
            
            start_time = time.time()
            
            # Train with optimized settings
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                evaluator=evaluator,
                evaluation_steps=eval_steps,
                warmup_steps=warmup_steps,
                output_path=str(self.output_dir),
                optimizer_params={'lr': learning_rate},
                save_best_model=True,
                checkpoint_path=str(self.output_dir / "checkpoints"),
                checkpoint_save_steps=save_steps,
                use_amp=True if self.device == "cuda" else False,  # Mixed precision for GPU
            )
            
            end_time = time.time()
            training_time = end_time - start_time
            
            logger.info(f"Training completed successfully!")
            logger.info(f"  - Training time: {training_time:.2f} seconds")
            logger.info(f"  - Model saved to: {self.output_dir}")
            logger.info(f"  - Final memory usage: {self._get_gpu_memory()}")
            
            # Save training configuration
            config_data = {
                "model_info": {
                    "base_model": self.model_name,
                    "device": self.device,
                    "gpu_name": torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU"
                },
                "training_params": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "warmup_steps": warmup_steps,
                    "total_steps": total_steps
                },
                "data_info": {
                    "training_examples": len(self.train_examples),
                    "evaluation_examples": len(self.eval_examples)
                },
                "results": {
                    "training_time_seconds": training_time,
                    "date_trained": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
            with open(self.output_dir / "gtx1660_training_config.json", 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info("Training configuration saved")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            if "out of memory" in str(e).lower():
                logger.error("GPU out of memory! Try:")
                logger.error("  - Reducing batch size (--batch-size 8)")
                logger.error("  - Using CPU (--device cpu)")
                logger.error("  - Reducing training data size")
            raise
        finally:
            # Clean up GPU memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()


def main():
    """Main function for GTX 1660 optimized training"""
    setup_logger(logging.INFO)
    
    import argparse
    parser = argparse.ArgumentParser(description="GTX 1660 Optimized Training")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--model", type=str, default="paraphrase-MiniLM-L3-v2", help="Base model")
    parser.add_argument("--output", type=str, default="swtbot-gtx1660-optimized", help="Output model name")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (16 optimal for GTX 1660)")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="Device to use")
    parser.add_argument("--create-data", action="store_true", help="Create training data first")
    parser.add_argument("--test-cases", type=str, help="Test cases file for data creation")
    parser.add_argument("--swtbot-refs", type=str, help="SWTBot references file for data creation")
    
    args = parser.parse_args()
    
    # Determine device
    use_gpu = True
    if args.device == "cpu":
        use_gpu = False
    elif args.device == "cuda":
        use_gpu = True
    # auto will be determined by the trainer
    
    # Create trainer
    trainer = GTX1660Trainer(
        model_name=args.model,
        output_model_name=args.output,
        use_gpu=use_gpu
    )
    
    # Create training data if requested
    if args.create_data:
        if not args.test_cases or not args.swtbot_refs:
            logger.error("--test-cases and --swtbot-refs required with --create-data")
            return 1
        
        from src.train import ModelTrainer
        temp_trainer = ModelTrainer()
        temp_trainer.create_training_data_from_examples(
            test_cases_path=args.test_cases,
            swtbot_references_path=args.swtbot_refs,
            output_path=args.data
        )
        logger.info(f"Training data created at {args.data}")
    
    # Train model
    trainer.prepare_training_data(args.data)
    trainer.train_optimized(
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    logger.info("GTX 1660 optimized training completed successfully!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
