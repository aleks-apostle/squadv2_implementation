from datetime import datetime
import logging
from pathlib import Path
import string
from typing import Dict

import numpy as np
import wandb
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertTokenizerFast, get_linear_schedule_with_warmup


class QATrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        tokenizer: DistilBertTokenizerFast,
        learning_rate: float = 3e-5,
        warmup_ratio: float = 0.1,
        num_epochs: int = 3,
        max_grad_norm: float = 1.0,
        use_wandb: bool = True
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.use_wandb = use_wandb

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() 
                                 else 'mps' if torch.backends.mps.is_available() 
                                 else 'cpu')
        self.model = self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Calculate total steps for scheduler
        total_steps = len(train_dataloader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)

        # Initialize scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _prepare_batch_inputs(self, batch: Dict) -> Dict:
        """Prepare inputs for the model by filtering out non-model inputs."""
        model_inputs = {
            'input_ids': batch['input_ids'].to(self.device),
            'attention_mask': batch['attention_mask'].to(self.device),
        }
        
        # Add training-specific inputs
        if 'start_positions' in batch:
            model_inputs['start_positions'] = batch['start_positions'].to(self.device)
        if 'end_positions' in batch:
            model_inputs['end_positions'] = batch['end_positions'].to(self.device)
        if 'is_impossible' in batch:
            model_inputs['is_impossible'] = batch['is_impossible'].to(self.device)
            
        return model_inputs

    def train(self):
        """Main training loop with comprehensive logging and evaluation."""
        best_f1 = 0
        best_model_path = None
        patience = 3
        patience_counter = 0

        # Initialize wandb
        if self.use_wandb:
            wandb.init(project="squad_qa", name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            wandb.watch(self.model)

        for epoch in range(self.num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.evaluate()

            # Log metrics
            metrics = {**train_metrics, **val_metrics}
            self.log_metrics(metrics, epoch)

            # Save best model based on answerable_accuracy instead of f1
            if val_metrics['answerable_accuracy'] > best_f1:
                best_f1 = val_metrics['answerable_accuracy']
                patience_counter = 0
                best_model_path = self.save_checkpoint(epoch, val_metrics, is_best=True)
                self.logger.info(f"New best accuracy: {best_f1:.4f}")
            else:
                patience_counter += 1
                self.logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

            # Early stopping
            if patience_counter >= patience:
                self.logger.info(f"No improvement for {patience} epochs. Stopping early.")
                break

            # Regular checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best=False)

            self.logger.info(f"Epoch {epoch + 1} metrics:")
            self.logger.info(f"Train loss: {train_metrics['train_loss']:.4f}")
            self.logger.info(f"Validation metrics:")
            for k, v in val_metrics.items():
                self.logger.info(f"{k}: {v:.4f}")

        return best_model_path, best_f1

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        answerability_preds = []
        answerability_true = []

        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for step, batch in enumerate(progress_bar):
            # Prepare inputs (removing extra fields like 'weight')
            model_inputs = self._prepare_batch_inputs(batch)
            
            # Get sample weights if available
            sample_weights = batch.get('weight', None)
            if sample_weights is not None:
                sample_weights = sample_weights.to(self.device)

            # Forward pass
            outputs = self.model(**model_inputs)
            loss = outputs[0]

            # Apply sample weighting if available
            if sample_weights is not None:
                loss = loss * sample_weights.view(-1)
                loss = loss.mean()

            # Track answerability predictions
            answerable_logits = outputs[2]  # Third output is answerable_logits
            answerability_preds.extend(torch.argmax(answerable_logits, dim=1).cpu().numpy())
            answerability_true.extend(batch['is_impossible'].cpu().numpy())

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.max_grad_norm
            )

            # Optimization step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # Update progress
            total_loss += loss.item()
            progress_bar.set_postfix({
                'loss': total_loss / (step + 1),
                'lr': self.scheduler.get_last_lr()[0]
            })

        # Calculate answerability metrics
        answerability_report = classification_report(
            answerability_true, 
            answerability_preds, 
            output_dict=True
        )

        return {
            'train_loss': total_loss / len(self.train_dataloader),
            'train_answerability_f1': answerability_report['macro avg']['f1-score'],
            'train_impossible_f1': answerability_report.get('1', {}).get('f1-score', 0.0)
        }

    def evaluate(self):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                # Prepare inputs
                model_inputs = self._prepare_batch_inputs(batch)
                
                # Forward pass
                outputs = self.model(**model_inputs)
                loss = outputs[0]
                total_loss += loss.item()

                # Get predictions
                start_logits = outputs[1]
                end_logits = outputs[2]
                answerable_logits = outputs[3]

                # Process predictions
                for i in range(start_logits.size(0)):
                    # Get answerable probability
                    is_impossible = torch.argmax(answerable_logits[i]).item()
                    
                    if not is_impossible:
                        start_idx = torch.argmax(start_logits[i]).item()
                        end_idx = torch.argmax(end_logits[i]).item()
                        
                        # Get answer text
                        answer_tokens = batch['input_ids'][i][start_idx:end_idx + 1]
                        predicted_answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                    else:
                        predicted_answer = ""

                    all_predictions.append({
                        'predicted_answer': predicted_answer,
                        'true_answer': batch['answer_text'][i],
                        'is_impossible': batch['is_impossible'][i].item(),
                        'predicted_impossible': is_impossible
                    })

        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions)
        metrics['val_loss'] = total_loss / len(self.val_dataloader)

        return metrics

    def _calculate_metrics(self, predictions):
        """
        Calculate comprehensive evaluation metrics.
        """
        total = len(predictions)
        if total == 0:
            return {
                'exact_match': 0.0,
                'answerable_accuracy': 0.0,
                'impossible_accuracy': 0.0,
                'overall_accuracy': 0.0
            }

        # Count different types of predictions
        correct_impossible = 0
        correct_answerable = 0
        total_impossible = 0
        total_answerable = 0

        for pred in predictions:
            if pred['is_impossible']:
                total_impossible += 1
                if pred['predicted_impossible']:
                    correct_impossible += 1
            else:
                total_answerable += 1
                if not pred['predicted_impossible']:
                    if self._normalize_answer(pred['predicted_answer']) == self._normalize_answer(pred['true_answer']):
                        correct_answerable += 1

        # Calculate metrics
        metrics = {
            'exact_match': (correct_impossible + correct_answerable) / total,
            'answerable_accuracy': correct_answerable / total_answerable if total_answerable > 0 else 0.0,
            'impossible_accuracy': correct_impossible / total_impossible if total_impossible > 0 else 0.0,
            'overall_accuracy': (correct_impossible + correct_answerable) / total
        }

        # Add precision and recall for impossible questions
        predicted_impossible = sum(1 for p in predictions if p['predicted_impossible'])
        if predicted_impossible > 0:
            metrics['impossible_precision'] = correct_impossible / predicted_impossible
        else:
            metrics['impossible_precision'] = 0.0

        if total_impossible > 0:
            metrics['impossible_recall'] = correct_impossible / total_impossible
        else:
            metrics['impossible_recall'] = 0.0

        # Calculate F1 scores
        metrics['impossible_f1'] = self._calculate_f1(
            metrics['impossible_precision'],
            metrics['impossible_recall']
        )

        return metrics

    def _calculate_f1(self, precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _normalize_answer(self, s: str) -> str:
        """Normalize answer string for comparison."""
        if not s:
            return ""
        
        # Convert to lowercase
        s = s.lower()
        
        # Remove punctuation
        s = ''.join(c for c in s if c not in string.punctuation)
        
        # Remove extra whitespace
        s = ' '.join(s.split())
        
        return s

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)

        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'epoch': epoch
        }

        # Save checkpoint
        if is_best:
            path = checkpoint_dir / 'best_model.pt'
        else:
            path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'

        torch.save(checkpoint, path)
        return path

    def log_metrics(self, metrics: dict, epoch: int):
        """Log metrics to console and wandb if enabled."""
        self.logger.info("\nMetrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"{key}: {value:.4f}")
            else:
                self.logger.info(f"{key}: {value}")

        if self.use_wandb:
            wandb.log(metrics, step=epoch)

