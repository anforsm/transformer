import torch

class Tokenizer:
  def __init__(self, corpus):
    self.corpus = corpus

    chars = set()
    for char in corpus:
      chars.update(char)
    self.chars = sorted(list(chars))

    self.char_to_id = {c: i for i, c in enumerate(self.chars)}
    self.id_to_char = {i: c for i, c in enumerate(self.chars)}

    self.vocab_size = len(self.chars)
  
  def __call__(self, text):
    return self.encode(text)
    
  def encode(self, text) -> torch.Tensor:
    return torch.tensor([self.char_to_id[c] for c in text])
  
  def decode(self, ids):
    ids = ids.tolist()
    return ''.join([self.id_to_char[i] for i in ids])
  