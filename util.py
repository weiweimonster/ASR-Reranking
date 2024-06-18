import torch

def build_character_vocab(data):
    chars = set(data)
    atoi = {ch: i for i, ch in enumerate(chars)}
    atoi["[UNK]"] = len(atoi)
    atoi["[PAD]"] = len(atoi)
    iota = {i: ch for i, ch in enumerate(chars)}
    iota[len(iota)] = "[UNK]"
    iota[len(iota)] = "[PAD]"
    return atoi, iota

def to_tokens(inputs, atoi):
    input_tokens = []
    for ch in inputs:
        if ch in atoi:
            input_tokens.append(atoi[ch])
        else:
            input_tokens.append(atoi["[UNK]"])
    target_tokens = input_tokens[1:] + [atoi["[PAD]"]]
    return torch.tensor(input_tokens), torch.tensor(target_tokens)