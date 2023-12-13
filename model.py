import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

class OptimusPrime(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, seq_len=512):
    super().__init__()
    self.seq_len = seq_len
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.positional_encoding = PositionalEncoding(embedding_dim, seq_len)
    self.decoder = nn.TransformerEncoder(
      nn.TransformerEncoderLayer(
        d_model=embedding_dim,
        nhead=12,
        dim_feedforward=hidden_dim,
        dropout=0.1,
        activation='relu',
        batch_first=True,
      ),
      num_layers=12,
    )
    
    self.head = nn.Linear(embedding_dim, vocab_size)
  
  def to(self, device):
    super().to(device)
    self.device = device
    return self
    
  def forward(self, x):
    x = self.embedding(x)
    x = self.positional_encoding(x)
    x = self.decoder(
      x, 
      is_causal=True, 
      mask=nn.Transformer.generate_square_subsequent_mask(x.size(1)),
    )
    x = self.head(x)
    x = F.log_softmax(x, dim=-1)
    return x
  
  def generate(self, x, max_len=256):
    device = x.device
    x = x.squeeze(0)
    x = list(x)
    x = [14 for _ in range(self.seq_len)] + x
    x = torch.tensor(x).to(device).unsqueeze(0)
    for _ in range(max_len):
      out = self(x[:, -self.seq_len:])
      next_token = torch.multinomial(torch.exp(out[0, -1]), 1)
      #next_token = torch.argmax(out[:, -1], dim=-1)
      x = torch.cat((x, next_token.unsqueeze(-1)), dim=-1)
      #yield next_token
    return x
  
  def train(self, text, tokenizer):
    epochs = 5
    tokenized_train_text = tokenizer(text)

    def get_sample(batch_size=1):
      start_idx = torch.randint(0, len(tokenized_train_text) - self.seq_len, (batch_size,))
      x = torch.stack([torch.tensor(tokenized_train_text[idx:idx+self.seq_len]) for idx in start_idx]).to(self.device)
      y = torch.stack([torch.tensor(tokenized_train_text[idx+1:idx+self.seq_len+1]) for idx in start_idx]).to(self.device)
      return x, y
    
    training_steps = int(epochs * len(tokenized_train_text) / self.seq_len)

    optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    pbar = tqdm(range(training_steps))
    for step in pbar:
      x, y = get_sample(32)
      out = self(x)
      loss = F.nll_loss(out.reshape(-1, out.size(-1)), y.reshape(-1))
      loss.backward()
      pbar.set_description(f"Loss {loss.item():.2f}")
      optimizer.step()
      optimizer.zero_grad()


class PositionalEncoding(nn.Module):
  def __init__(self, embedding_dim, seq_len):
    super().__init__()

    self.embedding_dim = embedding_dim
    self.seq_len = seq_len
    
    self.emb = nn.Embedding(seq_len, embedding_dim)
  
  def forward(self, x):
    device = x.device
    pos = torch.arange(self.seq_len, device=device)
    pos = self.emb(pos)
    pos = pos.unsqueeze(0)
    return x + pos
    