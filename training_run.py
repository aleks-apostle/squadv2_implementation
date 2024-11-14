from config import QAConfig
from dataloader import create_squad_dataset
from bert import BertForQAV2
from trainer import QATrainer
from transformers import DistilBertTokenizerFast
from torch.utils.data import DataLoader

def main():
    # Load configurations
    config = QAConfig()
    
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
    

if __name__ == "__main__":
    main()