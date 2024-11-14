import torch
import torch.nn as nn
from transformers import DistilBertModel
from typing import Tuple, Optional


class BertForQAV2(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", dropout_rate: float = 0.1):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.distilbert.config.hidden_size
        
        # Span prediction head
        self.qa_outputs = nn.Linear(hidden_size, 2)
        
        # Improved answerable classification head
        self.answerable_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)  # Binary classification: answerable/unanswerable
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the QA heads."""
        for module in [self.qa_outputs, self.answerable_classifier]:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        is_impossible: Optional[torch.Tensor] = None
    ):
        # Get BERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        
        sequence_output = self.dropout(outputs[0])
        pooled_output = sequence_output[:, 0, :]  # Use CLS token for classification
        
        # Get span logits
        span_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = span_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        # Get answerable logits
        answerable_logits = self.answerable_classifier(pooled_output)
        
        outputs = (start_logits, end_logits, answerable_logits)
        
        if start_positions is not None and end_positions is not None and is_impossible is not None:
            # Convert is_impossible to boolean mask
            is_impossible_mask = is_impossible.bool()
            
            # Calculate losses
            loss_fct = nn.CrossEntropyLoss()
            
            # Answerable loss with class weighting
            answerable_weight = torch.tensor([1.0, 2.0]).to(answerable_logits.device)
            answerable_loss = nn.CrossEntropyLoss(weight=answerable_weight)(
                answerable_logits,
                is_impossible
            )
            
            # Adjust start/end positions for impossible questions
            start_positions = torch.where(
                is_impossible_mask,
                torch.zeros_like(start_positions),
                start_positions
            )
            end_positions = torch.where(
                is_impossible_mask,
                torch.zeros_like(end_positions),
                end_positions
            )
            
            # Calculate span loss
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            span_loss = (start_loss + end_loss) / 2
            
            # Total loss with higher weight for answerability
            total_loss = span_loss + (2.0 * answerable_loss)
            
            outputs = (total_loss,) + outputs
        
        return outputs

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        impossible_threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with the model.
        
        Returns:
            Tuple of (start_logits, end_logits, is_impossible_prob)
        """
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        start_logits, end_logits, answerable_logits = outputs
        
        # Get probability of being impossible
        impossible_probs = torch.softmax(answerable_logits, dim=-1)[:, 1]
        
        return start_logits, end_logits, impossible_probs