import torch
from transformers import DistilBertTokenizerFast
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List

from config import QAConfig
from dataloader import create_squad_dataset
from bert import BertForQAV2
from evaluator import QAEvaluator

class ModelTester:
    def __init__(
        self,
        model_path: str,
        test_data_path: str,
        config: QAConfig,
        output_dir: str = "evaluation_results"
    ):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._setup_components()
    
    def _setup_components(self):
        """Initialize model, tokenizer, and evaluator."""
        self.logger.info("Initializing components...")
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            self.config.model.model_name,
            use_fast=True
        )
        
        # Initialize model
        self.model = BertForQAV2(model_name=self.config.model.model_name)
        
        # Load model weights
        self.logger.info(f"Loading model from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() 
                                 else 'mps' if torch.backends.mps.is_available() 
                                 else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize evaluator
        self.evaluator = QAEvaluator(
            similarity_threshold=0.85,
            case_sensitive=False,
            ignore_punctuation=True,
            ignore_articles=True
        )
        
        self.logger.info(f"Using device: {self.device}")
    
    def _create_dataloader(self) -> DataLoader:
        """Create dataloader for test data."""
        test_dataset = create_squad_dataset(
            self.test_data_path,
            self.tokenizer,
            max_seq_length=self.config.model.max_seq_length,
            doc_stride=self.config.model.doc_stride,
            max_query_length=self.config.model.max_query_length,
            is_training=True  # Keep True to get answer positions
        )
        
        return DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=4
        )
    
    def evaluate(self):
        """Run full evaluation of the model."""
        self.logger.info("Starting evaluation...")
        start_time = datetime.now()
        
        # Create dataloader
        dataloader = self._create_dataloader()
        
        # Track predictions
        all_predictions = []
        
        # Evaluate batches
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                    
                    start_logits = outputs[0]
                    end_logits = outputs[1]
                    answerable_logits = outputs[2]
                    
                    # Process each example in batch
                    for i in range(start_logits.size(0)):
                        # Get answerable probability
                        answerable_probs = torch.softmax(answerable_logits[i], dim=0)
                        is_impossible_prob = answerable_probs[1].item()
                        predicted_impossible = is_impossible_prob > self.config.prediction.impossible_threshold
                        
                        # Get answer if not predicted impossible
                        if not predicted_impossible:
                            start_idx = torch.argmax(start_logits[i]).item()
                            end_idx = torch.argmax(end_logits[i]).item()
                            
                            if end_idx >= start_idx:
                                answer_tokens = batch['input_ids'][i][start_idx:end_idx + 1]
                                predicted_answer = self.tokenizer.decode(
                                    answer_tokens, 
                                    skip_special_tokens=True
                                )
                            else:
                                predicted_answer = ""
                        else:
                            predicted_answer = ""
                        
                        # Add to evaluator
                        self.evaluator.add_prediction(
                            prediction=predicted_answer,
                            ground_truth=batch['answer_text'][i],
                            is_impossible=batch['is_impossible'][i].item(),
                            predicted_impossible=predicted_impossible,
                            confidence=1 - is_impossible_prob,
                            question=batch.get('question', [''])[i],
                            context=batch.get('context', [''])[i]
                        )
                        
                        # Store prediction
                        prediction_info = {
                            'id': batch['example_id'][i],
                            'question': batch.get('question', [''])[i],
                            'context': batch.get('context', [''])[i],
                            'predicted_answer': predicted_answer,
                            'true_answer': batch['answer_text'][i],
                            'is_impossible': batch['is_impossible'][i].item(),
                            'predicted_impossible': predicted_impossible,
                            'confidence': 1 - is_impossible_prob
                        }
                        all_predictions.append(prediction_info)
                
                except Exception as e:
                    self.logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                    continue
        
        # Calculate metrics
        metrics = self.evaluator.get_metrics()
        error_examples = self.evaluator.get_error_examples(n_examples=5)
        
        # Save results
        self._save_results(metrics, error_examples, all_predictions)
        
        # Log summary
        duration = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"\nEvaluation completed in {duration:.2f} seconds")
        self.logger.info(f"Results saved to {self.output_dir}")
        
        return metrics
    
    def _save_results(self, metrics: Dict, error_examples: Dict, predictions: List[Dict]):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_file = self.output_dir / f'metrics_{timestamp}.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save error examples
        error_file = self.output_dir / f'error_analysis_{timestamp}.json'
        with open(error_file, 'w') as f:
            json.dump(error_examples, f, indent=2)
        
        # Save all predictions
        predictions_file = self.output_dir / f'predictions_{timestamp}.json'
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Create Excel report
        df = pd.DataFrame(predictions)
        excel_file = self.output_dir / f'evaluation_report_{timestamp}.xlsx'
        with pd.ExcelWriter(excel_file) as writer:
            df.to_excel(writer, sheet_name='Predictions', index=False)
            
            # Add metrics sheet
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
            # Add error examples sheet
            error_df = pd.DataFrame([
                {
                    'error_type': error_type,
                    'examples': json.dumps(examples, indent=2)
                }
                for error_type, examples in error_examples.items()
            ])
            error_df.to_excel(writer, sheet_name='Error Analysis', index=False)

def main():
    # Load configuration
    config = QAConfig.create_default()
    
    # Initialize tester
    tester = ModelTester(
        model_path="checkpoints/best_model.pt",  # Path to your saved model
        test_data_path="data/dev-v2.0.json",    # Path to test data
        config=config,
        output_dir="evaluation_results"
    )
    
    # Run evaluation
    metrics = tester.evaluate()
    
    # Print summary
    print("\nEvaluation Summary:")
    print("=" * 50)
    print(f"Overall Accuracy: {metrics['accuracy']:.2%}")
    print(f"Answerable Accuracy: {metrics['answerable_accuracy']:.2%}")
    print(f"Impossible Accuracy: {metrics['impossible_accuracy']:.2%}")
    print(f"Average Similarity: {metrics['average_similarity']:.2%}")
    print("\nError Distribution:")
    for error_type, percentage in metrics['error_distribution'].items():
        print(f"{error_type}: {percentage:.2%}")

if __name__ == "__main__":
    main()
