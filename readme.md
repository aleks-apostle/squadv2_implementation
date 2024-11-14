```markdown
# Question Answering System Documentation

## System Overview

This Question Answering system is built on DistilBERT and designed to handle both answerable and unanswerable questions using the SQuAD 2.0 dataset. The system consists of four main components: the dataset handler (SQuADDataset), the model architecture (BertForQAV2), the training manager (QATrainer), and the prediction system (Predictor).

## Core Components

### Data Processing (SQuADDataset)
The data processing system centers around the SQuADDataset class, which handles the loading and preprocessing of SQuAD 2.0 format data. Each example in the dataset is processed to handle both answerable and unanswerable questions. The class implements a sliding window approach for long contexts, ensuring that answers spanning multiple windows are properly handled.

Example usage of the dataset:
```python
# Initialize dataset
from dataloader import create_squad_dataset

train_dataset = create_squad_dataset(
    'data/train-v2.0.json',
    tokenizer,
    max_seq_length=config.model.max_seq_length,
    doc_stride=config.model.doc_stride,
    max_query_length=config.model.max_query_length,
    is_training=True,
    balance_dataset=True
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.training.batch_size,
    shuffle=True,
    num_workers=4
)
```

### Model Architecture (BertForQAV2)
The model's architecture extends DistilBERT with two specific prediction heads: one for span prediction (identifying the start and end positions of answers) and another for determining whether a question is answerable. This dual-head architecture enables the model to both locate answers and identify unanswerable questions.

Example of model initialization:
```python
# Initialize model
from bert import BertForQAV2

model = BertForQAV2(
    model_name=config.model.model_name
)


```

### Training System (QATrainer)
The training system manages the training process with support for validation, early stopping, and model checkpointing. It handles both the span prediction task and the answerable/unanswerable classification task, balancing the losses between these objectives.

Example of training setup:
```python
# Initialize trainer
from trainer import QATrainer

trainer = QATrainer(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    learning_rate=3e-5,
    num_epochs=3
)

# Start training
best_model_path, best_metrics = trainer.train()
```

### Prediction System (Predictor)
The prediction system handles inference on new questions, implementing the sliding window approach for long contexts and providing confidence scores for predictions. It includes special handling for unanswerable questions through a confidence threshold mechanism.

Example of prediction:
```python
# Initialize predictor
from predictor import Predictor

predictor = Predictor(
    model_path="best_model.pt",
    impossible_threshold=0.5
)

# Make prediction
answer, confidence, is_answerable = predictor.predict(
    question="What is the capital of France?",
    context="Paris is the capital of France."
)
```

### Configuration System (QAConfig)
The configuration system uses dataclasses to manage model and training parameters, ensuring type safety and providing default values for all components.

```python
# Create default config
config = QAConfig.create_default()
```

## Evaluation System (QAEvaluator)

The QAEvaluator class provides comprehensive evaluation metrics for the QA system. It implements:

1. Exact Match and F1 Score calculations
2. Specialized handling of impossible questions
3. Text normalization for answer comparison

Here's how the evaluator actually works based on the implemented code:

```python
# Initialize evaluator
from evaluator import QAEvaluator

evaluator = QAEvaluator(
    similarity_threshold=0.85,
    case_sensitive=False,
    ignore_punctuation=True,
    ignore_articles=True
)

# Add predictions for evaluation
evaluator.add_prediction(
    prediction="predicted answer",
    ground_truth="true answer",
    is_impossible=False,
    predicted_impossible=False,
    confidence=0.9,
    question="question text",  # optional
    context="context text"     # optional
)


# Get evaluation metrics
metrics = evaluator.get_metrics()
```

The evaluator provides these metrics based on the actual implementation:
- Overall accuracy
- Impossible question accuracy
- Answerable question accuracy
- Average similarity scores
- Error distribution

## Predictor System

The Predictor class handles model inference. From the actual implementation, it includes:

```python
class Predictor:
    def __init__(
        self,
        model_path: str,
        config: Optional[QAConfig] = None,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        impossible_threshold: float = 0.5
    ):
        # Initialization code for loading model and setup
        pass

    def predict(
        self, 
        question: str, 
        context: str,
        return_all_spans: bool = False
    ) -> Tuple[str, float, bool, List[Dict[str, Any]]]:
        # Makes prediction and returns:
        # - answer text
        # - confidence score
        # - whether question is answerable
        # - all predicted spans
        pass

```

Usage example based on the actual implementation:

```python
predictor = Predictor(
    model_path="path/to/model.pt",
    impossible_threshold=0.5
)

# Single prediction
answer, confidence, is_answerable = predictor.predict(
    question="What is the capital of France?",
    context="Paris is the capital of France."
)
```

## Training Process

The training process is managed by the QATrainer class. Based on the actual implementation, it handles:

```python
from trainer import QATrainer

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
       

    def train(self):
        """Main training loop with comprehensive logging and evaluation."""
        best_f1 = 0
        best_model_path = None
        patience = 3
        patience_counter = 0

        for epoch in range(self.num_epochs):
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.evaluate()

            
            # Model saving logic based on performance
```

The training process includes:
- Epoch-level training with metrics tracking
- Validation after each epoch
- Model checkpointing based on performance
- Early stopping when performance plateaus

## Model Configuration

The configuration system uses dataclasses for type safety and parameter management:

```python
from config import QAConfig, ModelConfig, TrainingConfig, PredictionConfig
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
```

Usage example based on the implementation:

```python
# Create default config
config = QAConfig.create_default()

# Or create with custom values
custom_config = QAConfig(
    model=ModelConfig(max_seq_length=512),
    training=TrainingConfig(batch_size=32),
    prediction=PredictionConfig(impossible_threshold=0.6)
)
```

The configuration system manages:
- Model architecture parameters
- Training hyperparameters
- Prediction settings
- Data processing parameters


## Complete Usage Workflows

Based on the implemented code, here are the complete workflows for training, evaluation, and prediction:

### Training Workflow

This workflow demonstrates how to train a new model using our implemented classes:

```python
from dataloader import create_squad_dataset
from bert import BertForQAV2
from trainer import QATrainer
from config import QAConfig
from transformers import DistilBertTokenizerFast

# 1. Initialize configuration
config = QAConfig.create_default()

# Initialize tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(config.model.model_name)

# Create datasets
train_dataset = create_squad_dataset(
    'data/train-v2.0.json',
    tokenizer,
    max_seq_length=config.model.max_seq_length,
    doc_stride=config.model.doc_stride,
    max_query_length=config.model.max_query_length,
    is_training=True,
    balance_dataset=True
)

val_dataset = create_squad_dataset(
    'data/dev-v2.0.json',
    tokenizer,
    max_seq_length=config.model.max_seq_length,
    doc_stride=config.model.doc_stride,
    max_query_length=config.model.max_query_length,
    is_training=True,
    balance_dataset=False
)

# Create dataloaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.training.batch_size,
    shuffle=True,
    num_workers=4
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=config.training.batch_size,
    shuffle=False,
    num_workers=4
)

# Initialize model
model = BertForQAV2(
    model_name=config.model.model_name,
    dropout_rate=config.model.dropout_rate
)

# Initialize trainer
trainer = QATrainer(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    tokenizer=tokenizer,
    learning_rate=config.training.learning_rate,
    warmup_ratio=config.training.warmup_ratio,
    num_epochs=config.training.num_epochs,
    max_grad_norm=config.training.max_grad_norm
)

# Train model
best_model_path, best_f1 = trainer.train()
```

### Evaluation Workflow

This workflow shows how to evaluate a trained model:

```python
# 1. Initialize evaluator
evaluator = QAEvaluator(
    similarity_threshold=0.85,
    case_sensitive=False,
    ignore_punctuation=True,
    ignore_articles=True
)

# 2. Initialize predictor with trained model
predictor = Predictor(
    model_path="path/to/best_model.pt",
    config=config
)

# 3. Process validation data and evaluate
for batch in val_dataloader:
    # Get predictions
    for i in range(len(batch['input_ids'])):
        question = batch['question'][i]
        context = batch['context'][i]
        
        # Make prediction
        answer, confidence, is_answerable = predictor.predict(
            question,
            context
        )
        
        # Add to evaluator
        evaluator.add_prediction(
            prediction=answer,
            ground_truth=batch['answer_text'][i],
            is_impossible=batch['is_impossible'][i],
            predicted_impossible=not is_answerable,
            confidence=confidence,
            question=question,
            context=context
        )

# 4. Get evaluation metrics
metrics = evaluator.get_metrics()
```

### Prediction Usage

This shows how to use the trained model for predictions:

```python
# 1. Initialize predictor
predictor = Predictor(
    model_path="path/to/best_model.pt",
    impossible_threshold=0.5,
    max_answer_length=30
)

# 2. Single prediction
answer, confidence, is_answerable = predictor.predict(
    question="What is the capital of France?",
    context="Paris is the capital of France."
)
```

Key Notes and Limitations

Based on the actual implementation:

1. The model expects input sequences no longer than the configured max_seq_length (default 384 tokens)
2. The sliding window approach is used for longer contexts with a configurable stride
3. Confidence scores are provided for all predictions
4. The system handles both answerable and unanswerable questions through the impossible_threshold parameter
