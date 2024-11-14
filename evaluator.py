import string
import re
from typing import Dict, List
import numpy as np
from collections import defaultdict
from Levenshtein import ratio

class QAEvaluator:
    """
    Enhanced evaluator for Question Answering with better metrics 
    and handling of impossible questions.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        case_sensitive: bool = False,
        ignore_punctuation: bool = True,
        ignore_articles: bool = True
    ):
        """
        Initialize evaluator with customizable text matching settings.
        
        Args:
            similarity_threshold: Threshold for considering answers similar (0-1)
            case_sensitive: Whether to consider case in matching
            ignore_punctuation: Whether to ignore punctuation in matching
            ignore_articles: Whether to ignore articles (a, an, the) in matching
        """
        self.similarity_threshold = similarity_threshold
        self.case_sensitive = case_sensitive
        self.ignore_punctuation = ignore_punctuation
        self.ignore_articles = ignore_articles
        self.predictions = []
        self.error_types = defaultdict(int)
    
    def normalize_answer(self, text: str) -> str:
        """
        Normalize answer text based on evaluator settings.
        
        Args:
            text: Answer text to normalize
            
        Returns:
            Normalized answer text
        """
        if not text:
            return ""
            
        # Convert to string if needed
        text = str(text)
        
        # Apply case normalization
        if not self.case_sensitive:
            text = text.lower()
        
        # Remove articles if specified
        if self.ignore_articles:
            text = re.sub(r'\b(a|an|the)\b', '', text, flags=re.IGNORECASE)
        
        # Remove punctuation if specified
        if self.ignore_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))
        
        # Normalize whitespace
        text = " ".join(text.split())
        
        return text
    
    def compute_answer_similarity(self, prediction: str, ground_truth: str) -> float:
        """
        Compute similarity between prediction and ground truth using Levenshtein distance.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            Similarity score between 0 and 1
        """
        pred_norm = self.normalize_answer(prediction)
        truth_norm = self.normalize_answer(ground_truth)
        
        if not pred_norm and not truth_norm:
            return 1.0
        if not pred_norm or not truth_norm:
            return 0.0
            
        return ratio(pred_norm, truth_norm)
    
    def categorize_prediction(
        self,
        prediction: str,
        ground_truth: str,
        is_impossible: bool,
        predicted_impossible: bool,
        similarity: float
    ) -> str:
        """
        Categorize the prediction result for detailed error analysis.
        
        Args:
            prediction: Predicted answer text
            ground_truth: Ground truth answer text
            is_impossible: Whether question is actually impossible
            predicted_impossible: Whether model predicted impossible
            similarity: Similarity score between prediction and ground truth
            
        Returns:
            Category string describing the prediction result
        """
        if is_impossible:
            if predicted_impossible:
                return "correct_impossible"
            return "false_positive"  # Predicted answer for impossible question
        
        if predicted_impossible:
            return "false_negative"  # Failed to predict answer for possible question
        
        if similarity >= self.similarity_threshold:
            return "correct_answer"
        
        if similarity >= 0.5:
            return "partial_match"
        
        if not prediction.strip():
            return "empty_prediction"
            
        return "wrong_answer"
    
    def add_prediction(
        self,
        prediction: str,
        ground_truth: str,
        is_impossible: bool,
        predicted_impossible: bool,
        confidence: float,
        question: str = "",
        context: str = ""
    ):
        """
        Add a prediction for evaluation.
        
        Args:
            prediction: Predicted answer text
            ground_truth: Ground truth answer text
            is_impossible: Whether question is actually impossible
            predicted_impossible: Whether model predicted impossible
            confidence: Model's confidence in prediction
            question: Optional question text for analysis
            context: Optional context text for analysis
        """
        # Compute similarity for answerable questions
        similarity = 0.0
        if not is_impossible and not predicted_impossible:
            similarity = self.compute_answer_similarity(prediction, ground_truth)
        
        # Determine if prediction is correct
        is_correct = False
        if is_impossible:
            is_correct = predicted_impossible
        else:
            is_correct = similarity >= self.similarity_threshold
        
        # Categorize prediction
        error_type = self.categorize_prediction(
            prediction,
            ground_truth,
            is_impossible,
            predicted_impossible,
            similarity
        )
        
        # Store prediction details
        self.predictions.append({
            'question': question,
            'context': context,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'is_impossible': is_impossible,
            'predicted_impossible': predicted_impossible,
            'confidence': float(confidence),
            'similarity': float(similarity),
            'is_correct': is_correct,
            'error_type': error_type
        })
        
        # Update error type counts
        self.error_types[error_type] += 1
    
    def get_metrics(self) -> Dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Returns:
            Dictionary containing various evaluation metrics
        """
        if not self.predictions:
            return {
                'total_examples': 0,
                'accuracy': 0.0,
                'impossible_accuracy': 0.0,
                'answerable_accuracy': 0.0
            }
        
        total = len(self.predictions)
        impossible_preds = [p for p in self.predictions if p['is_impossible']]
        answerable_preds = [p for p in self.predictions if not p['is_impossible']]
        
        metrics = {
            'total_examples': total,
            'total_impossible': len(impossible_preds),
            'total_answerable': len(answerable_preds),
            
            # Overall metrics
            'accuracy': sum(1 for p in self.predictions if p['is_correct']) / total,
            
            # Impossible question metrics
            'impossible_accuracy': (
                sum(1 for p in impossible_preds if p['is_correct']) / len(impossible_preds)
                if impossible_preds else 0.0
            ),
            
            # Answerable question metrics
            'answerable_accuracy': (
                sum(1 for p in answerable_preds if p['is_correct']) / len(answerable_preds)
                if answerable_preds else 0.0
            ),
            
            # Average similarity for answerable questions
            'average_similarity': (
                np.mean([p['similarity'] for p in answerable_preds])
                if answerable_preds else 0.0
            ),
            
            # Confidence analysis
            'average_confidence': np.mean([p['confidence'] for p in self.predictions]),
            'confidence_correct': (
                np.mean([p['confidence'] for p in self.predictions if p['is_correct']])
                if any(p['is_correct'] for p in self.predictions) else 0.0
            ),
            'confidence_incorrect': (
                np.mean([p['confidence'] for p in self.predictions if not p['is_correct']])
                if any(not p['is_correct'] for p in self.predictions) else 0.0
            ),
            
            # Error distribution
            'error_distribution': {
                error_type: count / total
                for error_type, count in self.error_types.items()
            }
        }
        
        return metrics
    
    def get_error_examples(self, n_examples: int = 5) -> Dict:
        """
        Get examples of different error types for analysis.
        
        Args:
            n_examples: Number of examples to get for each category
            
        Returns:
            Dictionary containing examples of different error types
        """
        error_examples = defaultdict(list)
        
        for pred in self.predictions:
            if len(error_examples[pred['error_type']]) < n_examples:
                error_examples[pred['error_type']].append({
                    'question': pred.get('question', ''),
                    'context': pred.get('context', ''),
                    'prediction': pred['prediction'],
                    'ground_truth': pred['ground_truth'],
                    'confidence': pred['confidence'],
                    'similarity': pred['similarity']
                })
        
        return dict(error_examples)
    
    def print_evaluation_report(self):
        """Print a detailed evaluation report."""
        metrics = self.get_metrics()
        error_examples = self.get_error_examples(n_examples=3)
        
        print("\nEvaluation Report")
        print("=" * 50)
        
        print("\nOverall Metrics:")
        print(f"Total examples: {metrics['total_examples']}")
        print(f"Overall accuracy: {metrics['accuracy']:.2%}")
        print(f"Answerable accuracy: {metrics['answerable_accuracy']:.2%}")
        print(f"Impossible accuracy: {metrics['impossible_accuracy']:.2%}")
        print(f"Average similarity: {metrics['average_similarity']:.2%}")
        
        print("\nConfidence Analysis:")
        print(f"Average confidence: {metrics['average_confidence']:.2%}")
        print(f"Confidence when correct: {metrics['confidence_correct']:.2%}")
        print(f"Confidence when incorrect: {metrics['confidence_incorrect']:.2%}")
        
        print("\nError Distribution:")
        for error_type, percentage in metrics['error_distribution'].items():
            print(f"{error_type}: {percentage:.2%}")
        
        print("\nError Examples:")
        for error_type, examples in error_examples.items():
            if examples:
                print(f"\n{error_type.replace('_', ' ').title()}:")
                for i, example in enumerate(examples, 1):
                    print(f"\nExample {i}:")
                    if example['question']:
                        print(f"Question: {example['question']}")
                    print(f"Prediction: {example['prediction']}")
                    print(f"Ground truth: {example['ground_truth']}")
                    print(f"Confidence: {example['confidence']:.2%}")
                    if 'similarity' in example:
                        print(f"Similarity: {example['similarity']:.2%}")
    
    def reset(self):
        """Reset the evaluator state."""
        self.predictions = []
        self.error_types.clear()
