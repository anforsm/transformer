from tokenizer import Tokenizer, BPETokenizer
from model import OptimusPrime

with open("data/01 - The Fellowship Of The Ring.txt", "r", encoding="iso-8859-1") as f:
  text = f.read()
with open("data/02 - The Two Towers.txt", "r", encoding="iso-8859-1") as f:
  text += f.read()
with open("data/03 - The Return Of The King.txt", "r", encoding="iso-8859-1") as f:
  text += f.read()

tok = BPETokenizer(text, 16384)
tok.save("tokenizer_trained.json")