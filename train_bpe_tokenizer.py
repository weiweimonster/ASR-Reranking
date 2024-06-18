from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

files = ["data/lm_data/treebank-sentences-train.txt"]

# Train the tokenizer
tokenizer.train(files, trainer)

# Save the tokenizer to use it later
tokenizer_path = "trained_tokenizer.json"
tokenizer.save(tokenizer_path)
