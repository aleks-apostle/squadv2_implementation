from dataclasses import dataclass, field
from typing import Any

@dataclass
class ModelConfig:
    model_name: str = "distilbert-base-uncased"
    max_seq_length: int = 384
    doc_stride: int = 128
    max_query_length: int = 64
    dropout_rate: float = 0.1

@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 3e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    impossible_weight: float = 2.0
    early_stopping_patience: int = 3

@dataclass
class PredictionConfig:
    max_answer_length: int = 30
    impossible_threshold: float = 0.5
    n_best_size: int = 20

@dataclass
class QAConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)

    @classmethod
    def create_default(cls) -> 'QAConfig':
        """Create a QAConfig with default values."""
        return cls(
            model=ModelConfig(),
            training=TrainingConfig(),
            prediction=PredictionConfig()
        )