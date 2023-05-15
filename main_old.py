import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k


class SelfAttention(nn.Module):
	def __init__(self, embed_size, heads):
		super(SelfAttention, self).__init__()
		self.embed_size = embed_size
		self.heads = heads
		self.head_dim = embed_size // heads

		assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"

		self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
		self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
		self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
		self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

	def forward(self, values, keys, query, mask):
		N = query.shape[0]
		value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

		# Split embedding into self.heads pieces
		values = values.reshape(N, value_len, self.heads, self.head_dim)
		keys = keys.reshape(N, key_len, self.heads, self.head_dim)
		queries = query.reshape(N, query_len, self.heads, self.head_dim)

		values = self.values(values)
		keys = self.keys(keys)
		queries = self.queries(queries)

		energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
		# queries shape: (N, query_len, heads, head_dim)
		# keys shape: (N, key_len, heads, head_dim)
		# energy shape: (N, heads, query_len, key_len)

		if mask is not None:
			energy = energy.masked_fill(mask==0, float("-1e20"))

		attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

		out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
			N, query_len, self.heads*self.head_dim
		)
		# attention shape: (N, heads, query_len, key_len)
		# values shape: (N, value_len, heads, head_dim)
		# (N, query_len, heads, head_dim)

		out = self.fc_out(out)
		return out

class TransformerBlock(nn.Module):
	def __init__(self, embed_size, heads, dropout, forward_expansion):
		super(TransformerBlock, self).__init__()
		self.attention = SelfAttention(embed_size, heads)
		self.norm1 = nn.LayerNorm(embed_size)
		self.norm2 = nn.LayerNorm(embed_size)

		self.feed_forward = nn.Sequential(
			nn.Linear(embed_size, forward_expansion*embed_size),
			nn.ReLU(),
			nn.Linear(forward_expansion*embed_size, embed_size)
		)
		self.dropout = nn.Dropout(dropout)

	def forward(self, value, key, query, mask):
		attention = self.attention(value, key, query, mask)

		x = self.dropout(self.norm1(attention + query))
		forward = self.feed_forward(x)

		out = self.dropout(self.norm2(forward + x))
		return out


# This is encoder
class Encoder(nn.Module):
	def __init__(self, src_vocab_size, embed_size, n_layers, heads, device, forward_expansion, dropout, max_length):
		super(Encoder, self).__init__()
		self.embed_size = embed_size
		self.device = device
		self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
		self.position_embedding = nn.Embedding(max_length, embed_size)
		self.layers = nn.ModuleList(
			[
				TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion)
			for _ in range(n_layers)]
		)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, mask):
		N, seq_length = x.shape
		positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
		out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

		for layer in self.layers:
			out = layer(out, out, out, mask)

		return out


class DecoderBlock(nn.Module):
	def __init__(self, embed_size, heads, forward_expansion, dropout, device):
		super(DecoderBlock, self).__init__()
		self.attention = SelfAttention(embed_size, heads)
		self.norm = nn.LayerNorm(embed_size)
		self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, value, key, src_mask, trg_mask):
		attention = self.attention(x, x, x, trg_mask)
		query = self.dropout(self.norm(attention + x))
		out = self.transformer_block(value, key, query, src_mask)
		return out


class Decoder(nn.Module):
	def __init__(self,
				trg_vocab_size,
				embed_size,
				num_layers,
				heads,
				forward_expansion,
				dropout,
				device,
				max_length,
		):
		super(Decoder, self).__init__()
		self.device = device
		self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
		self.position_embedding = nn.Embedding(max_length, embed_size)

		self.layers = nn.ModuleList(
			[DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
				for _ in range(num_layers)]
		)

		self.fc_out = nn.Linear(embed_size, trg_vocab_size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, enc_out, src_mask, trg_mask):
		N, seq_length = x.shape
		positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
		x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

		for layer in self.layers:
			x = layer(x, enc_out, enc_out, src_mask, trg_mask)

		out = self.fc_out(x)
		return out


class Transformer(nn.Module):
	def __init__(
			self,
			src_vocab_size,
			trg_vocab_size,
			src_pad_idx,
			trg_pad_idx,
			embed_size=256,
			num_layers=6,
			forward_expansion=4,
			heads=8,
			dropout=0,
			device="cpu",
			max_length=100,
	):
		super(Transformer, self).__init__()

		self.encoder = Encoder(
			src_vocab_size,
			embed_size,
			num_layers,
			heads,
			device,
			forward_expansion,
			dropout,
			max_length,
		)

		self.decoder = Decoder(
			trg_vocab_size,
			embed_size,
			num_layers,
			heads,
			forward_expansion,
			dropout,
			device,
			max_length,
		)

		self.src_pad_idx = src_pad_idx
		self.trg_pad_idx = trg_pad_idx
		self.device = device

	def make_src_mask(self, src):
		src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
		# (N, 1, 1, src_len)
		return src_mask.to(self.device)

	def make_trg_mask(self, trg):
		N, trg_len = trg.shape
		trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
			N, 1, trg_len, trg_len
		)
		return trg_mask.to(self.device)

	def forward(self, src, trg):
		src_mask = self.make_src_mask(src)
		trg_mask = self.make_trg_mask(trg)
		enc_src = self.encoder(src, src_mask)
		out = self.decoder(trg, enc_src, src_mask, trg_mask)
		return out

def train(model):
	loss_fn = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
	epoch_number = 0

	EPOCHS = 5

	best_vloss = 1_000_000.

	for epoch in range(EPOCHS):
		print('EPOCH {}:'.format(epoch_number + 1))

		# Make sure gradient tracking is on, and do a pass over the data
		model.train(True)
		avg_loss = train_one_epoch(epoch_number, writer)

		# We don't need gradients on to do reporting
		model.train(False)

		running_vloss = 0.0
		for i, vdata in enumerate(validation_loader):
			vinputs, vlabels = vdata
			voutputs = model(vinputs)
			vloss = loss_fn(voutputs, vlabels)
			running_vloss += vloss

		avg_vloss = running_vloss / (i + 1)
		print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

		# Log the running loss averaged per batch
		# for both training and validation
		writer.add_scalars('Training vs. Validation Loss',
						{ 'Training' : avg_loss, 'Validation' : avg_vloss },
						epoch_number + 1)
		writer.flush()

		# Track best performance, and save the model's state
		if avg_vloss < best_vloss:
			best_vloss = avg_vloss
			model_path = 'model_{}_{}'.format(timestamp, epoch_number)
			torch.save(model.state_dict(), model_path)

		epoch_number += 1

def train_one_epoch(epoch_index, tb_writer):
	training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
	running_loss = 0.
	last_loss = 0.

	# Here, we use enumerate(training_loader) instead of
	# iter(training_loader) so that we can track the batch
	# index and do some intra-epoch reporting
	for i, data in enumerate(training_loader):
		# Every data instance is an input + label pair
		inputs, labels = data

		# Zero your gradients for every batch!
		optimizer.zero_grad()

		# Make predictions for this batch
		outputs = model(inputs)

		# Compute the loss and its gradients
		loss = loss_fn(outputs, labels)
		loss.backward()

		# Adjust learning weights
		optimizer.step()

		# Gather data and report
		running_loss += loss.item()
		if i % 1000 == 999:
			last_loss = running_loss / 1000 # loss per batch
			print('  batch {} loss: {}'.format(i + 1, last_loss))
			tb_x = epoch_index * len(training_loader) + i + 1
			tb_writer.add_scalar('Loss/train', last_loss, tb_x)
			running_loss = 0.

	return last_loss

if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
		device
	)
	trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(
		device
	)

	src_pad_idx = 0
	trg_pad_idx = 0
	src_vocab_size = 10
	trg_vocab_size = 10
	model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
		device
	)
	out = model(x, trg[:, :-1])
	print(out.shape) # (N, trg_len - 1, trg_vocab_size)

	SRC_LANGUAGE = 'de'
	TGT_LANGUAGE = 'en'
	# load multi30k dataset
	train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
	val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
	test_iter = Multi30k(split='test', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

	# print some examples
	src_language_index = train_iter.dataset.fields[SRC_LANGUAGE].vocab.stoi[SRC_LANGUAGE]
	tgt_language_index = train_iter.dataset.fields[TGT_LANGUAGE].vocab.stoi[TGT_LANGUAGE]
	print(f"Index of {SRC_LANGUAGE} language: {src_language_index}")
	print(f"Index of {TGT_LANGUAGE} language: {tgt_language_index}")
	print("First training example:")
	print(vars(train_iter.dataset.examples[0]))