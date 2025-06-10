"""
Model adapter module for supporting different types of models
"""
import logging
from typing import Dict, Any, List, Optional, Union

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

class ModelAdapter:
    """
    Adapter for different types of models to provide a consistent interface
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the model adapter
        
        Args:
            model_config: Configuration dictionary for the model
        """
        self.model_config = model_config
        self.model_type = model_config.get('model_type', 'sentence-transformer')
        self.model_name = model_config.get('model_name')
        self.model = None
        self.tokenizer = None
        
        # Load the appropriate model
        if self.model_type == 'sentence-transformer':
            self._load_sentence_transformer()
        elif self.model_type == 'huggingface':
            self._load_huggingface_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        logger.info(f"Loaded model: {self.model_name} ({self.model_type})")
    
    def _load_sentence_transformer(self):
        """Load a SentenceTransformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            
            # Set device (CPU for low-spec hardware)
            device = self.model_config.get('device', 'cpu')
            self.model.to(device)
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {str(e)}")
            raise
    
    def _load_huggingface_model(self):
        """Load a HuggingFace model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Set device (CPU for low-spec hardware)
            device = self.model_config.get('device', 'cpu')
            self.model.to(device)
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {str(e)}")
            raise
    
    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts into vectors
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Tensor of encoded vectors
        """
        if self.model_type == 'sentence-transformer':
            return self.model.encode(texts, convert_to_tensor=True)
        
        elif self.model_type == 'huggingface':
            # Basic implementation for HuggingFace models
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use mean pooling of last hidden state
            attention_mask = inputs['attention_mask']
            last_hidden = outputs.last_hidden_state
            
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        
        else:
            raise ValueError(f"Unsupported model type for encoding: {self.model_type}")
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Encode texts
        embeddings = self.encode([text1, text2])
        
        # Calculate cosine similarity
        emb1 = embeddings[0]
        emb2 = embeddings[1]
        
        # Normalize embeddings
        emb1 = emb1 / torch.norm(emb1)
        emb2 = emb2 / torch.norm(emb2)
        
        # Cosine similarity
        return torch.dot(emb1, emb2).item()
    
    def batch_similarity(self, text: str, candidates: List[str]) -> List[float]:
        """
        Calculate similarity between a text and multiple candidates
        
        Args:
            text: The query text
            candidates: List of candidate texts
            
        Returns:
            List of similarity scores
        """
        # Encode text and candidates
        text_embedding = self.encode([text])[0]
        candidate_embeddings = self.encode(candidates)
        
        # Normalize embeddings
        text_embedding = text_embedding / torch.norm(text_embedding)
        normalized_candidates = candidate_embeddings / torch.norm(candidate_embeddings, dim=1, keepdim=True)
        
        # Calculate cosine similarities
        similarities = torch.matmul(normalized_candidates, text_embedding).tolist()
        
        return similarities
