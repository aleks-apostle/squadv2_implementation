from typing import List, Tuple, Dict, Optional
import torch
import numpy as np
from transformers import DistilBertTokenizerFast
from tqdm import tqdm
import logging
from pathlib import Path

from config import QAConfig

class Predictor:
    """
    A comprehensive predictor for question answering that handles both
    answerable and unanswerable questions.
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional['QAConfig'] = None,
        tokenizer: Optional[DistilBertTokenizerFast] = None,
        impossible_threshold: float = 0.5,
        max_answer_length: int = 30,
        n_best_size: int = 20,
        max_seq_length: int = 384,
        doc_stride: int = 128
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the saved model checkpoint
            config: Optional QAConfig object for configuration
            tokenizer: Optional pre-initialized tokenizer
            impossible_threshold: Threshold for considering a question unanswerable
            max_answer_length: Maximum length of predicted answer
            n_best_size: Number of best answers to consider
            max_seq_length: Maximum sequence length for input
            doc_stride: Stride for sliding window over long contexts
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Store configuration
        self.config = config
        self.impossible_threshold = impossible_threshold
        self.max_answer_length = max_answer_length
        self.n_best_size = n_best_size
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() 
                                 else 'mps' if torch.backends.mps.is_available() 
                                 else 'cpu')
        
        # Initialize tokenizer
        self.tokenizer = tokenizer or DistilBertTokenizerFast.from_pretrained(
            'distilbert-base-uncased',
            use_fast=True
        )
        
        # Load model
        self.model = self._load_model(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.logger.info(f"Predictor initialized with device: {self.device}")
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """
        Load the model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            
        Returns:
            Loaded model
        """
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize model
            from bert import BertForQAV2
            model = BertForQAV2(model_name="distilbert-base-uncased")
            
            # Load state dict
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            
            self.logger.info(f"Model loaded successfully from {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _get_best_spans(
        self,
        start_logits: np.ndarray,
        end_logits: np.ndarray,
        input_ids: torch.Tensor,
        n_best: int
    ) -> List[Dict]:
        """
        Get the best answer spans from logits.
        
        Args:
            start_logits: Start position logits
            end_logits: End position logits
            input_ids: Input token IDs
            n_best: Number of best spans to return
            
        Returns:
            List of dictionaries containing span information
        """
        best_spans = []
        
        # Get best start indices
        start_indices = np.argsort(start_logits)[-n_best:]
        
        # Get best end indices
        end_indices = np.argsort(end_logits)[-n_best:]
        
        for start_idx in start_indices:
            for end_idx in end_indices:
                # Skip invalid spans
                if end_idx < start_idx:
                    continue
                
                length = end_idx - start_idx + 1
                if length > self.max_answer_length:
                    continue
                
                # Get answer text
                answer_tokens = input_ids[start_idx:end_idx + 1]
                answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                if not answer_text.strip():
                    continue
                
                # Calculate score (average of start and end logits, normalized by length)
                score = (start_logits[start_idx] + end_logits[end_idx]) / length
                
                best_spans.append({
                    'text': answer_text,
                    'score': score,
                    'start_index': start_idx,
                    'end_index': end_idx,
                    'length': length
                })
        
        return sorted(best_spans, key=lambda x: x['score'], reverse=True)
    
    def predict(
        self,
        question: str,
        context: str,
        return_all_spans: bool = False
    ) -> Tuple[str, float, bool]:
        """
        Make a prediction for a single question-context pair.
        
        Args:
            question: Question text
            context: Context text
            return_all_spans: Whether to return all candidate spans
            
        Returns:
            Tuple of (answer_text, confidence_score, is_answerable)
        """
        # Tokenize input
        encoding = self.tokenizer(
            question,
            context,
            max_length=self.max_seq_length,
            truncation='only_second',
            stride=self.doc_stride,
            return_tensors='pt',
            padding='max_length'
        )
        
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        try:
            with torch.no_grad():
                # Get model outputs
                outputs = self.model(**encoding)
                start_logits = outputs[0]
                end_logits = outputs[1]
                answerable_logits = outputs[2]
                
                # Check if question is answerable
                answerable_probs = torch.softmax(answerable_logits, dim=-1)
                is_impossible_prob = answerable_probs[0][1].item()
                
                if is_impossible_prob > self.impossible_threshold:
                    if return_all_spans:
                        return "", is_impossible_prob, False, []
                    return "", is_impossible_prob, False
                
                # Get best spans
                best_spans = self._get_best_spans(
                    start_logits[0].cpu().numpy(),
                    end_logits[0].cpu().numpy(),
                    encoding['input_ids'][0],
                    self.n_best_size
                )
                
                if not best_spans:
                    if return_all_spans:
                        return "", 0.0, False, []
                    return "", 0.0, False
                
                # Get best answer
                best_answer = best_spans[0]
                confidence = (1 - is_impossible_prob) * best_answer['score']
                
                if return_all_spans:
                    return best_answer['text'], confidence, True, best_spans
                return best_answer['text'], confidence, True
                
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            if return_all_spans:
                return "", 0.0, False, []
            return "", 0.0, False
    

def create_predictor(
    model_path: str,
    config: Optional['QAConfig'] = None,
    **kwargs
) -> Predictor:
    """
    Helper function to create a predictor instance.
    
    Args:
        model_path: Path to model checkpoint
        config: Optional configuration object
        **kwargs: Additional arguments for Predictor
        
    Returns:
        Initialized Predictor
    """
    return Predictor(model_path=model_path, config=config, **kwargs)