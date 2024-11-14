import json
import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast


class SQuADDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 384,
        doc_stride: int = 128,
        max_query_length: int = 64,
        is_training: bool = True,
        balance_dataset: bool = True,
        impossible_weight: float = 2.0
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.is_training = is_training
        self.balance_dataset = balance_dataset
        self.impossible_weight = impossible_weight
        
        # Verify tokenizer type
        if not getattr(self.tokenizer, 'is_fast', False):
            raise ValueError(
                "This dataset seems to require a fast tokenizer."
            )
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)['data']
            
        self.examples = self._create_examples()
        self._log_statistics()
    
    def _create_examples(self) -> List[Dict]:
        examples = []
        answerable = []
        unanswerable = []
        
        for article in self.data:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                
                for qa in paragraph['qas']:
                    example = {
                        'id': qa['id'],
                        'question': qa['question'],
                        'context': context,
                        'is_impossible': qa.get('is_impossible', False),
                        'answer_text': "",
                        'answer_start': -1,
                        'answers': qa.get('answers', []),
                        'weight': 1.0
                    }
                    
                    if not example['is_impossible'] and example['answers']:
                        example['answer_text'] = example['answers'][0]['text']
                        example['answer_start'] = example['answers'][0]['answer_start']
                        answerable.append(example)
                    else:
                        example['weight'] = self.impossible_weight
                        unanswerable.append(example)
        
        if self.balance_dataset and self.is_training:
            n_answerable = len(answerable)
            n_unanswerable = len(unanswerable)
            target_size = max(n_answerable, n_unanswerable)
            
            if n_answerable < n_unanswerable:
                answerable = answerable * (n_unanswerable // n_answerable + 1)
                answerable = answerable[:n_unanswerable]
            elif n_unanswerable < n_answerable:
                unanswerable = unanswerable * (n_answerable // n_unanswerable + 1)
                unanswerable = unanswerable[:n_answerable]
            
            examples = answerable + unanswerable
            random.shuffle(examples)
        else:
            examples = answerable + unanswerable
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]
        
        # Tokenize question and context separately first
        question_encoding = self.tokenizer(
            example['question'],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_query_length
        )
        
        context_encoding = self.tokenizer(
            example['context'],
            add_special_tokens=False,
            truncation=False
        )
        
        # Calculate available length for context
        special_tokens_count = 3  # [CLS], [SEP], [SEP]
        max_context_length = self.max_seq_length - len(question_encoding['input_ids']) - special_tokens_count
        
        # Create the full encoding with truncation and sliding window
        encoding = self.tokenizer(
            example['question'],
            example['context'],
            truncation='only_second',
            max_length=self.max_seq_length,
            stride=self.doc_stride,
            padding='max_length',
            return_overflowing_tokens=True,
            return_offsets_mapping=True
        )
        
        # Get the first sequence (we'll handle overflow in a more sophisticated version)
        encoding = {k: v[0] for k, v in encoding.items()}
        
        # Convert to tensors
        encoding = {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'example_id': example['id'],
            'context': example['context'],
            'question': example['question'],
            'is_impossible': torch.tensor(int(example['is_impossible'])),
            'weight': torch.tensor(example['weight'])
        }
        
        if self.is_training:
            if not example['is_impossible']:
                # Handle answer positions without relying on offset mapping
                answer_text = example['answer_text']
                input_text = self.tokenizer.decode(encoding['input_ids'])
                
                # Find the answer span in tokenized input
                answer_tokens = self.tokenizer.encode(answer_text, add_special_tokens=False)
                input_ids = encoding['input_ids'].tolist()
                
                # Simple span finding (can be improved)
                start_position = 0
                end_position = 0
                
                for i in range(len(input_ids)):
                    if input_ids[i:i+len(answer_tokens)] == answer_tokens:
                        start_position = i
                        end_position = i + len(answer_tokens) - 1
                        break
                
                encoding['start_positions'] = torch.tensor(start_position)
                encoding['end_positions'] = torch.tensor(end_position)
            else:
                # For impossible questions, point to CLS token
                encoding['start_positions'] = torch.tensor(0)
                encoding['end_positions'] = torch.tensor(0)
            
            encoding['answer_text'] = example['answer_text']
        
        return encoding
    
    def _log_statistics(self):
        total = len(self.examples)
        impossible = sum(1 for ex in self.examples if ex['is_impossible'])
        answerable = total - impossible
        
        print(f"\nDataset Statistics:")
        print(f"Total examples: {total}")
        print(f"Answerable questions: {answerable} ({answerable/total:.1%})")
        print(f"Unanswerable questions: {impossible} ({impossible/total:.1%})")


def create_squad_dataset(
    data_path: str,
    tokenizer: DistilBertTokenizerFast,
    is_training: bool = True,
    max_seq_length: int = 384,
    doc_stride: int = 128,
    max_query_length: int = 64,
    balance_dataset: bool = True
) -> SQuADDataset:
    """
    Helper function to create a SQuAD dataset with proper configuration.
    """
    return SQuADDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=is_training,
        balance_dataset=balance_dataset
    )