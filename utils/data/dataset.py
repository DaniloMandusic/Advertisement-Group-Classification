import torch
from torch.utils.data import Dataset

class SequenceClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],       # (max_length,)
            "attention_mask": self.encodings["attention_mask"][idx],  # (max_length,)
            "labels":         self.labels[idx]                        # scalar
        }