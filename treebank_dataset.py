from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence


class TreebankDataset(Dataset):
    def __init__(self, data, atoi, itoa, block_size=32):
        self.atoi = atoi
        self.itoa = itoa
        self.block_size = block_size
        self.data = self.convert(data)

    def convert(self, data):
        unknown_idx = self.atoi["[UNK]"]
        for i, ch in enumerate(data):
            if ch in self.atoi:
                data[i] = self.atoi[ch]
            else:
                data[i] = unknown_idx
        return torch.tensor(data)

    def __len__(self):
        return len(self.data) // self.block_size + 1

    def __getitem__(self, idx):
        index = idx * self.block_size
        if index + self.block_size < len(self.data):
            return self.data[index: index + self.block_size], self.data[index + 1: index + self.block_size + 1]
        data = self.data[index:]
        label = self.data[index + 1:]

        data_padding = (self.block_size - len(data)) * (len(self.atoi) - 1)
        label_padding = (self.block_size - len(label)) * (len(self.atoi) - 1)

        tensor_padding = torch.tensor(data_padding)
        tensor_label_padding = torch.tensor(label_padding)

        tensor_data = torch.cat((tensor_data, tensor_padding), dim=0)
        tensor_label = torch.cat((tensor_label,tensor_label_padding), dim=0)
        return tensor_data, tensor_label
