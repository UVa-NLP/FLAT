import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from gensim.models import KeyedVectors
import textattack


SMALL = 1e-08


class MASK_BERN(nn.Module):
	def __init__(self, args):
		super(MASK_BERN, self).__init__()

		self.device = args.device
		self.mask_hidden_dim = args.mask_hidden_dim
		self.activations = {'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'relu': torch.relu, 'leaky_relu': F.leaky_relu}
		self.activation = self.activations[args.activation]
		self.embed_dim = args.embed_dim

		# ============= Covariance matrix & Mean vector ================
		self.linear_layer = nn.Linear(self.embed_dim, self.mask_hidden_dim)

		self.hidden2p = nn.Linear(self.mask_hidden_dim, 2)
		

	def forward_sent_batch(self, embeds):

		temps = self.activation(self.linear_layer(embeds))
		p = self.hidden2p(temps)  # seqlen, bsz, dim
		return p

	def forward(self, x, p):
		if self.training:
			r = F.gumbel_softmax(p,hard=True,dim=2)[:,:,1:2]
			x_prime = r * x
			return x_prime
		else:
			probs = F.softmax(p,dim=2)[:,:,1:2] #select the probs of being 1
			x_prime = probs * x			
			return x_prime

	def get_statistics_batch(self, embeds):
		p = self.forward_sent_batch(embeds)
		return p


class CNN(nn.Module):
	def __init__(self, args, kernel_num, embed_dim, dropout, max_seq_length, embed_num, wordvocab, num_labels, kernel_sizes):
		super(CNN, self).__init__()
		
		self.kernel_num = kernel_num
		self.embed_dim = embed_dim
		self.dropout = dropout
		self.max_seq_length = max_seq_length
		self.embed_num = embed_num
		self.num_labels = num_labels
		self.kernel_sizes = kernel_sizes
		self.word_emb = nn.Embedding(embed_num, embed_dim, padding_idx=0)

		# initialize word embedding with pretrained word2vec
		emb_matrix = self.getVectors(embed_dim, wordvocab)
		self.word_emb.weight.data.copy_(torch.from_numpy(emb_matrix))

		self.word_emb.weight.requires_grad = False
		self.in_channels = 1

		# <unk> vectors is randomly initialized
		nn.init.uniform_(self.word_emb.weight.data[1], -0.05, 0.05)

		# <pad> vector is initialized as zero padding
		nn.init.constant_(self.word_emb.weight.data[0], 0)

		for filter_size in self.kernel_sizes:
			conv = nn.Conv1d(self.in_channels, kernel_num, embed_dim * filter_size, stride=embed_dim)
			setattr(self, 'conv_' + str(filter_size), conv)

		self.fc = nn.Linear(len(self.kernel_sizes) * kernel_num, num_labels)

		word2id = {}
		for i, w in enumerate(wordvocab):
			word2id[w] = i
		self.word2id = word2id

		self.tokenizer = textattack.models.tokenizers.GloveTokenizer(
			word_id_map=self.word2id,
			unk_token_id=1,
			pad_token_id=0,
			max_length=max_seq_length,
		)
		
		
	def getVectors(self, embed_dim, wordvocab):
		vectors = []
		word2vec = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
		for i in range(len(wordvocab)):
			word = wordvocab[i]
			if word in word2vec.vocab:
				vectors.append(word2vec[word])
			else:
				vectors.append(np.random.uniform(-0.01, 0.01, embed_dim))
				
		return np.array(vectors)
		
		
	def forward(self, x):
		batch_size, seq_len, _ = x.shape
		conv_in = x.view(batch_size, 1, -1)
		conv_result = [
			F.max_pool1d(F.relu(getattr(self, 'conv_' + str(filter_size))(conv_in)), seq_len - filter_size + 1).view(-1,
																													 self.kernel_num)
			for filter_size in self.kernel_sizes]

		R_out = torch.cat(conv_result, 1)
		out = F.dropout(R_out, p=self.dropout, training=self.training)
		out = self.fc(out)
		out = F.softmax(out, 1)

		return out, R_out


class MyCNNVmask(nn.Module):
	def __init__(
		self,
		args,
		kernel_num=200,
		embed_dim=300,
		dropout=0.2,
		num_labels=2,
		max_seq_length=250,
		model_path=None,
		emb_layer_trainable=False,
		wordvocab = None,
		hidden_layer = 1,
	):
		super(MyCNNVmask, self).__init__()
		self.args = args
		embed_dim = args.embed_dim
		kernel_num = args.kernel_num
		kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
		dropout = args.dropout
		num_labels = args.num_labels
		max_seq_length = args.max_seq_length
		embed_num = len(wordvocab)
		self.blnk = MASK_BERN(args)
		self.cnnmodel = CNN(args, kernel_num, embed_dim, dropout, max_seq_length, embed_num, wordvocab, num_labels, kernel_sizes)


	def get_importance_score(self, batch):
		# embedding
		x = batch
		embed = self.cnnmodel.word_emb(x)
		# MASK
		p = self.blnk.get_statistics_batch(embed)
		probs = F.softmax(p,dim=2)[:,:,1:2] #select the probs of being 1
		return probs
	
	
	def forward(self, batch, do_explain=False, explainer=None):
		if not do_explain:
			embed = self.cnnmodel.word_emb(batch)
			x = embed
		else:
			x = batch
		# MASK
		p = self.blnk.get_statistics_batch(x)
		x_prime = self.blnk(x, p)
		output, R_out = self.cnnmodel(x_prime)

		probs_pos = F.softmax(p,dim=2)[:,:,1]
		probs_neg = F.softmax(p,dim=2)[:,:,0]
		self.infor_loss = torch.mean(probs_pos * torch.log(probs_pos+1e-8) + probs_neg*torch.log(probs_neg+1e-8))
		if explainer == 'lime':
			return output, R_out
		return output
