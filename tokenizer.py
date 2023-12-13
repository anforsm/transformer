from collections import defaultdict
import torch
from tqdm import tqdm
import json

class Tokenizer:
  def __init__(self, corpus):
    self.corpus = corpus

    chars = set()
    for char in corpus:
      chars.update(char)
    self.chars = sorted(list(chars))
    self.chars.insert("<|pad|>")

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

class BPETokenizer:
  def __init__(self, corpus, vocab_size=100):
    self.corpus = corpus

    chars = set()
    for char in corpus:
      chars.update(char)
    chars = sorted(list(chars))
    #chars.insert(0, "<|pad|>")
    self.longest_token = 1
    self.vocab = set(chars)

    self.update_idx()

    pbar = tqdm(total=vocab_size)
    curr_vocab_size = len(self.vocab)
    pbar.update(curr_vocab_size)
    tokenized = self.encode(corpus)
    while curr_vocab_size < vocab_size:
      token_1, token_2 = self.get_most_common_bigram(tokenized)
      new_token = self.id_to_char[token_1] + self.id_to_char[token_2]
      new_token_id = len(self.vocab)


      if len(new_token) > self.longest_token:
        self.longest_token = len(new_token)

      self.char_to_id[new_token] = new_token_id
      self.id_to_char[new_token_id] = new_token

      self.vocab.add(new_token)
      curr_vocab_size = len(self.vocab)


      tokenized = self.encode(corpus)
      # fix tokeniation according to latest merge
      # replace_at_ids = []
      # prev = tokenized[0]
      # for i in range(1, len(tokenized)):
      #   curr = tokenized[i]
      #   if prev == token_1 and curr == token_2:
      #     replace_at_ids.append(i-1)

      #   prev = curr
      
      # for i in reversed(replace_at_ids):
      #   tokenized.pop(i+1)
      #   tokenized[i] = new_token_id

      pbar.update(1)


    self.vocab_size = len(self.vocab)
  
  def update_idx(self):
    self.char_to_id = {c: i for i, c in enumerate(self.vocab)}
    self.id_to_char = {i: c for i, c in enumerate(self.vocab)}
  
  def get_most_common_bigram(self, tokenized):

    #self.tokenized = self.encode(self.corpus)
    bigrams = defaultdict(int)
    for i in range(len(tokenized) - 1):
      bigrams[(tokenized[i], tokenized[i+1])] += 1

    most_common_bigram = None
    most_common_bigram_count = 0
    for bigram, count in bigrams.items():
      if count > most_common_bigram_count:
        most_common_bigram = bigram
        most_common_bigram_count = count
    
    return most_common_bigram
  
  def __call__(self, text):
    return self.encode(text)
  
  def encode(self, text):
    tokens = []
    i = 0

    while i < len(text):
      longest_token = ''
      for j in range(i, len(text)):
        token = text[i:j + 1]
        if token in self.vocab:
          longest_token = token
        if len(token) > self.longest_token:
          break
        
      tokens.append(self.char_to_id[longest_token])
      i += len(longest_token)

    return torch.tensor(tokens)
  
  def decode(self, tokens):
    tokens = tokens.tolist()
    return "".join([self.id_to_char[tok] for tok in tokens])
  
  def save(self, path="tokenizer.json"):
    with open(path, "w") as f:
      json.dump({
        "vocab": list(self.vocab),
        "longest_token": self.longest_token,
        "vocab_size": self.vocab_size,
        "char_to_id": self.char_to_id,
        "id_to_char": self.id_to_char,
      }, f)
  
  @staticmethod
  def load(path="tokenizer.json"):
    pass
    