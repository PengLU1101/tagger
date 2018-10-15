import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import os
#from Layer.CNN_Layer import *
#from Layer.RNN_Layer import *
from Layer.CRF import *
from Layer.Share_Bilstm_Layer import *
from Layer.Embedding_Layer import *
import util
import pickle
USE_CUDA = torch.cuda.is_available()

class Fc_Layer(nn.Module):
    def __init__(self, d_in, d_out, dropout, crf=False, activation=False):
        super(Fc_Layer, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.crf = crf
        if self.crf:
            self.d_out = d_out + 2
        self.dropout_p = dropout
        self.Dropout = nn.Dropout(dropout)
        self.activation = activation

        self.fc = nn.Linear(d_in, self.d_out)

    def forward(self, inputs):
        seq_len, batch_size, d_in = inputs.size()
        assert d_in == self.d_in
        #inputs = self.Dropout(inputs)
        out = self.fc(inputs.contiguous().view(seq_len*batch_size, -1))
        if self.activation:
            out = F.relu(out)
        return out.contiguous().view(seq_len, batch_size, -1)

class Classifier(nn.Module):
    def __init__(self, layer, crf=True):
        super(Classifier, self).__init__()
        self.crf_flag = crf
        if crf:
            self.crf = layer
        else:
            self.logsm = nn.LogSoftmax(dim=-1)

    def forward(self,
                logits,
                real_tag,
                mask):
        seq_len, batchsize, d_in = logits.size()
        lens = self.get_lens(mask)
        logits = logits.transpose(1, 0) * mask.unsqueeze(-1)
        if self.crf_flag:
            neglog = self.loglik(real_tag, lens, logits)
            return neglog.mean()
        else:
            neglog = self.logsm(logits)
            return neglog
    def inference(self,
                  logits,
                  mask):
        seq_len, batchsize, d_in = logits.size()
        lens = self.get_lens(mask)
        if self.crf_flag:
            logits = logits.transpose(1, 0) * mask.unsqueeze(-1)
            scores, preds = self.crf.viterbi_decode(logits, lens)
        else:
            logits = logits.transpose(1, 0) * mask.unsqueeze(-1)
            out = self.logsm(logits)
            scores, preds = torch.max(out, dim=-1)
            preds = (preds.detach().float() * mask).long()

        return scores, preds

    def _bilstm_score(self,
                      logits,
                      real_tag,
                      lens):
        real_tag_exp = real_tag.unsqueeze(-1)
        scores = torch.gather(logits, 2, real_tag_exp).squeeze(-1)
        mask = sequence_mask(lens).float()
        scores = scores * mask
        score = scores.sum(1).squeeze(-1)

        return score
    def score(self,
              real_tag,
              lens,
              logits):
        transition_score = self.crf.transition_score(real_tag, lens)
        bilstm_score = self._bilstm_score(logits, real_tag, lens)
        score = transition_score + bilstm_score

        return score

    def loglik(self,
               real_tag,
               lens,
               logits):
        norm_score = self.crf(logits, lens)
        sequence_score = self.score(real_tag, lens, logits)
        loglik = norm_score - sequence_score
        return loglik
    def get_lens(self, mask):
        """
        Arguments:
            mask: batch x seqlen
        """
        lens = torch.sum(mask, dim=1)
        return lens


class Neural_Tagger(nn.Module):
    def __init__(self, Word_embeddings, Feature_embeddings,
        ShareRNN, FNNlist, Classifierlist, concat_flag):
        super(Neural_Tagger, self).__init__()
        self.Word_embeddings = Word_embeddings
        self.Feature_embeddings = Feature_embeddings
        self.ShareRNN = ShareRNN
        self.FClist = FNNlist
        self.Classifierlist = Classifierlist
        self.concat_flag = concat_flag
        #self.regulizaion_flag = regulizaion_flag

    def forward(self, src_seqs, src_masks, src_feats,
               tgt_seqs, tgt_masks, 
               idx):

        taskidx = idx
        src_words = src_seqs
        src_masks = src_masks
        src_feats = src_feats
        tgt_seqs = tgt_seqs
        tgt_masks = tgt_masks
        logits, h_p, h_s = self.encode(src_words, src_masks, src_feats, taskidx)
        return self.decode(logits, h_p, h_s, tgt_seqs, src_masks, taskidx)

    def encode(self, src_words, src_masks, src_feats, taskidx):

        batch_size, seq_len = src_words.size()
        word_embeds = self.Word_embeddings(src_words)
        feat_embeds = self.Feature_embeddings(src_feats)
        inputs = torch.cat((word_embeds, feat_embeds), dim=-1)

        h_f, c_f, final_list = self.ShareRNN(inputs, src_masks)

        ####
        if self.concat_flag:
            concat_hid = torch.cat((final_list[taskidx], h_f), dim=-1)
        else:
            concat_hid = final_list[taskidx] + h_f
        logits = self.FClist[taskidx](concat_hid)
        h_p = final_list[taskidx].detach()
        h_ff = h_f.detach() 
        return logits, h_p, h_ff
    def decode(self, logits, h_p, h_s, tgts, src_mask, taskidx):
        batch_size, seq_len, d_hid = logits.size()
        return self.Classifierlist[taskidx](logits, tgts, src_mask), h_p, h_s

    def predict(self, src_seqs, src_masks, src_feats,
               tgt_seqs, tgt_masks, idx):
        taskidx = idx
        src_words = src_seqs
        src_masks = src_masks
        src_feats = src_feats
        tgt_seqs = tgt_seqs
        tgt_masks = tgt_masks
        logits, _, _ = self.encode(src_words, src_masks, src_feats, taskidx)
        scores, preds = self.Classifierlist[taskidx].inference(logits, src_masks)
        return scores, preds


def build_model(para):
    emsize = para["d_emb"]
    d_hid = para["d_hid"]
    d_feat = para["d_feat"]
    n_layers = para["n_layers"]
    dropout = para["dropout"]
    n_feats = para["n_feats"]
    n_vocs = para["n_vocs"]
    n_tasks = para["n_tasks"]
    crf_flag = para["crf"]
    out_size = para["out_size"]
    concat_flag = para["concat_flag"]
    print(para)
    Word_embeddings = Embedding_Layer(n_vocs, emsize)
    Feature_embeddings = Embedding_Layer(n_feats, d_feat)
    rnn = StackedLSTMCell(emsize+d_feat, d_hid//2, n_layers, dropout, share=n_tasks)
    ShareRNN = BiSLSTM(rnn)
    if concat_flag:
        d_in_fc = d_hid * 2
    else:
        d_in_fc = d_hid
    FNNlist = nn.ModuleList([Fc_Layer(d_in=d_in_fc, d_out=d_out, dropout=dropout, crf=crf_flag) for d_out in out_size])
    
    if crf_flag:
        crf_list = nn.ModuleList([CRF_Layer(d_out) for d_out in out_size])
    else:
        crf_list = [d_out for d_out in out_size]
    Classifierlist = nn.ModuleList([Classifier(layer, crf=crf_flag) for layer in crf_list])
    model = Neural_Tagger(Word_embeddings, Feature_embeddings,
        ShareRNN, FNNlist, Classifierlist, concat_flag)
    if USE_CUDA:
        model = model.cuda()

    return model
    

def save_model(path, model, para):
    model_path = os.path.join(path, 'model.pt')
    torch.save(model.state_dict(), model_path)
    para_path = os.path.join(path, 'para.pkl')
    with open(para_path, "wb") as f:
        pickle.dump(para, f)

def read_model(path, model):
    model_path = os.path.join(path, 'model.pt')
    model.load_state_dict(torch.load(model_path))
    return model
