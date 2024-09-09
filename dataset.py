import torch
from torch.utils.data import Dataset, DataLoader

class Workload_dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        history_data = self.data[index]
        future_data = self.labels[index]
        
        return history_data, future_data

