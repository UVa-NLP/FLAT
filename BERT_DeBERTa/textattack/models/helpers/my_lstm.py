import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from gensim.models import KeyedVectors
import textattack


class MyLSTM(nn.Module):
    def __init__(
        self,
        args,
        hidden_dim=200,
        embed_dim=300,
        dropout=0.2,
        num_labels=2,
        max_seq_length=250,
        model_path=None,
        emb_layer_trainable=False,
        wordvocab = None,
        hidden_layer = 1,
    ):
        super(MyLSTM, self).__init__()
        self.args = args
        hidden_dim = args.hidden_dim
        embed_dim = args.embed_dim
        dropout = args.dropout
        max_seq_length = args.max_seq_length
        embed_num = len(wordvocab)
        self.embed = nn.Embedding(embed_num, embed_dim, padding_idx=0)

		# initialize word embedding with pretrained word2vec
        emb_matrix = self.getVectors(embed_dim, wordvocab)
        self.embed.weight.data.copy_(torch.from_numpy(emb_matrix))

		# fix embedding
        if not emb_layer_trainable:
            self.embed.weight.requires_grad = False
        
        # <unk> vectors is randomly initialized
        nn.init.uniform_(self.embed.weight.data[1], -0.05, 0.05)
        
        # <pad> vector is initialized as zero padding
        nn.init.constant_(self.embed.weight.data[0], 0)
        
        # lstm
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=hidden_layer)
        
        # initial weight
        init.xavier_normal_(self.lstm.all_weights[0][0], gain=np.sqrt(6.0))
        init.xavier_normal_(self.lstm.all_weights[0][1], gain=np.sqrt(6.0))
        
        # linear
        self.hidden2label = nn.Linear(hidden_dim, num_labels)
        
        # dropout
        self.dropout = dropout

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
        
        
    def forward(self, batch, do_explain=False, explainer=None):
        if not do_explain:
            x = batch.t()
            embed = self.embed(x)
            embed = F.dropout(embed, p=self.dropout, training=self.training)
            x = embed.view(len(x), embed.size(1), -1)
        else:
            x = batch.transpose(0, 1)
        # lstm
        lstm_out, _ = self.lstm(x)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        # pooling
        lstm_out = torch.tanh(lstm_out)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        R_out = torch.tanh(lstm_out)
        lstm_out = F.dropout(R_out, p=self.args.dropout, training=self.training)
        # linear
        logit = self.hidden2label(lstm_out)
        out = F.softmax(logit, 1)

        if explainer == 'lime':
            return out, R_out
        return out
