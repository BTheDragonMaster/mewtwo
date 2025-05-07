import torch

from torch.utils.data import Dataset, DataLoader


class RegressionDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Tokenize the sequence
        encoding = self.tokenizer(sequence, truncation=True, padding="max_length", max_length=self.max_length,
                                  return_tensors="pt")
        input_ids = encoding['input_ids'].squeeze(0)  # Squeeze the batch dimension
        attention_mask = encoding['attention_mask'].squeeze(0)  # Squeeze the batch dimension

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.float)  # For regression, labels are continuous values
        }


# Example sequences and labels
sequences = ["ACGTACGT", "AGCTAGCT", "CGTACGTA"]
labels = [0.5, 1.2, 0.8]  # Continuous labels for regression

