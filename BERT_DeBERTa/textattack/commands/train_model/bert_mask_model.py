import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
import torch.optim as optim
from torch.distributions.beta import Beta as Ba
from torch.distributions.kl import kl_divergence as KL

SMALL = 1e-08


class MASK(nn.Module):
    def __init__(self, args):
        super(MASK, self).__init__()

        self.device = args.device
        self.mask_hidden_dim = args.mask_hidden_dim
        self.activations = {'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'relu': torch.relu,
                            'leaky_relu': F.leaky_relu}
        self.activation = self.activations[args.activation]
        self.embed_dim = args.embed_dim

        # ============= Covariance matrix & Mean vector ================
        self.linear_layer = nn.Linear(self.embed_dim, self.mask_hidden_dim)

        self.hidden2p = nn.Linear(self.mask_hidden_dim, 2)

    def forward_sent_batch(self, embeds):

        temps = self.activation(self.linear_layer(embeds))
        p = self.hidden2p(temps)  # bsz, seqlen, dim
        return p

    def forward(self, x, p, atten):
        if self.training:
            r = F.gumbel_softmax(p, hard=True, dim=2)[:, :, 1:2]
            sel = r.to(torch.int64)
            x_prime = r * x
            mask_atten = r.squeeze(-1).long() * atten
            return x_prime, mask_atten
        else:
            probs = F.softmax(p, dim=2)[:, :, 1:2]  # select the probs of being 1
            x_prime = probs * x

            return x_prime, atten

    def get_statistics_batch(self, embeds):
        p = self.forward_sent_batch(embeds)
        return p

class MASK_BERT(nn.Module):

    def __init__(self, args, prebert):
        super(MASK_BERT, self).__init__()
        self.args = args
        self.maskmodel = MASK(args)
        self.bertmodel = prebert


    def get_importance_score(self, input_ids):
        # embedding
        token_type_ids = torch.zeros_like(input_ids)
        try:
            x = self.bertmodel.bert.embeddings(input_ids, token_type_ids)
        except:
            x = self.bertmodel.deberta.embeddings(input_ids, token_type_ids)
        # MASK
        p = self.maskmodel.get_statistics_batch(x)
        probs = F.softmax(p,dim=2)[:,:,1:2] #select the probs of being 1
        return probs
    
    
    def forward(self, inputs, in_emb=None, do_explain=False, explainer=None):
        if not do_explain:
            try:
                x = self.bertmodel.bert.embeddings(inputs['input_ids'], inputs['token_type_ids'])
            except:
                x = self.bertmodel.deberta.embeddings(inputs['input_ids'], inputs['token_type_ids'])
        else:
            x = in_emb

        # Mask
        p = self.maskmodel.get_statistics_batch(x)
        x_prime, mask_atten = self.maskmodel(x, p, inputs['attention_mask'])

        output = self.bertmodel(input_ids=inputs['input_ids'], attention_mask=mask_atten, token_type_ids=inputs['token_type_ids'], \
                labels=inputs['labels'], x_prime=x_prime, explainer=explainer)

        # self.infor_loss = F.softmax(p,dim=2)[:,:,1:2].mean()
        probs_pos = F.softmax(p,dim=2)[:,:,1]
        probs_neg = F.softmax(p,dim=2)[:,:,0]
        self.infor_loss = torch.mean(probs_pos * torch.log(probs_pos+1e-8) + probs_neg*torch.log(probs_neg+1e-8))
        return output#, output_tk
