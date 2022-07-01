import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from gensim.models import KeyedVectors
import textattack


class MyCNN(nn.Module):
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
        super(MyCNN, self).__init__()
        self.args = args
        embed_dim = args.embed_dim
        kernel_num = args.kernel_num
        self.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
        dropout = args.dropout
        max_seq_length = args.max_seq_length
        embed_num = len(wordvocab)
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
            conv = nn.Conv1d(self.in_channels, kernel_num, args.embed_dim * filter_size, stride=args.embed_dim)
            setattr(self, 'conv_' + str(filter_size), conv)

        self.fc = nn.Linear(len(self.kernel_sizes) * kernel_num, args.num_labels)

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
            x = batch
            batch_size, seq_len = x.size()
            conv_in = self.word_emb(x).view(batch_size, 1, -1)
        else:
            x = batch
            batch_size, seq_len, _ = x.shape
            conv_in = x.view(batch_size, 1, -1)
        
        conv_result = [
            F.max_pool1d(F.relu(getattr(self, 'conv_' + str(filter_size))(conv_in)), seq_len - filter_size + 1).view(-1,
                                                                                                                    self.args.kernel_num)
            for filter_size in self.kernel_sizes]

        R_out = torch.cat(conv_result, 1)
        out = F.dropout(R_out, p=self.args.dropout, training=self.training)
        out = self.fc(out)
        out = F.softmax(out, 1)
        if explainer == 'lime':
            return out, R_out
        return out
