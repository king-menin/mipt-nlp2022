import abc

import torch
from torch import nn
from torch.nn import functional
from torch.nn import init
from torch.nn.utils import rnn as rnn_utils


class BiLSTM(nn.Module):

    def __init__(self, embedding_size=300, hidden_dim=100, rnn_layers=1, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_size,
            hidden_dim // 2,
            rnn_layers, batch_first=True, bidirectional=True)

    def forward(self, input_, input_mask):
        length = input_mask.sum(-1)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_ = input_[sorted_idx]
        packed_input = rnn_utils.pack_padded_sequence(input_, sorted_lengths.data.tolist(), batch_first=True)
        output, (hidden, _) = self.lstm(packed_input)
        padded_outputs = rnn_utils.pad_packed_sequence(output, batch_first=True)[0]
        _, reversed_idx = torch.sort(sorted_idx)
        return padded_outputs[reversed_idx], hidden[:, reversed_idx]

    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)


class Linear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True):
        super(Linear, self).__init__(in_features, out_features, bias=bias)
        init.orthogonal_(self.weight)


class Linears(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hiddens,
                 bias=True,
                 activation='tanh'):
        super(Linears, self).__init__()
        assert len(hiddens) > 0

        self.in_features = in_features
        self.out_features = self.output_size = out_features

        in_dims = [in_features] + hiddens[:-1]
        self.linears = nn.ModuleList([Linear(in_dim, out_dim, bias=bias)
                                      for in_dim, out_dim
                                      in zip(in_dims, hiddens)])
        self.output_linear = Linear(hiddens[-1], out_features, bias=bias)
        self.activation = getattr(functional, activation)

    def forward(self, inputs):
        linear_outputs = inputs
        for linear in self.linears:
            linear_outputs = linear.forward(linear_outputs)
            linear_outputs = self.activation(linear_outputs)
        return self.output_linear.forward(linear_outputs)


# TODO: move to utils
def log_sum_exp(tensor, dim=0):
    """LogSumExp operation."""
    m, _ = torch.max(tensor, dim)
    m_exp = m.unsqueeze(-1).expand_as(tensor)
    return m + torch.log(torch.sum(torch.exp(tensor - m_exp), dim))


def sequence_mask(lens, max_len=None):
    batch_size = lens.size(0)

    if max_len is None:
        max_len = lens.max().item()

    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)

    if lens.data.is_cuda:
        ranges = ranges.cuda()

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp

    return mask


class CRF(nn.Module):
    def forward(self, *input_):
        return self.viterbi_decode(*input_)

    def __init__(self, label_size):
        super(CRF, self).__init__()

        self.label_size = label_size
        self.start = self.label_size - 2
        self.end = self.label_size - 1
        transition = torch.randn(self.label_size, self.label_size)
        self.transition = nn.Parameter(transition)
        self.initialize()

    def initialize(self):
        self.transition.data[:, self.end] = -100.0
        self.transition.data[self.start, :] = -100.0

    @staticmethod
    def pad_logits(logits):
        # lens = lens.data
        batch_size, seq_len, label_num = logits.size()
        # pads = Variable(logits.data.new(batch_size, seq_len, 2).fill_(-1000.0),
        #                 requires_grad=False)
        pads = logits.new_full((batch_size, seq_len, 2), -1000.0,
                               requires_grad=False)
        logits = torch.cat([logits, pads], dim=2)
        return logits

    def calc_binary_score(self, labels, lens):
        batch_size, seq_len = labels.size()

        # labels_ext = Variable(labels.data.new(batch_size, seq_len + 2))
        labels_ext = labels.new_empty((batch_size, seq_len + 2))
        labels_ext[:, 0] = self.start
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(lens + 1, max_len=(seq_len + 2)).long()
        # pad_stop = Variable(labels.data.new(1).fill_(self.end))
        pad_stop = labels.new_full((1,), self.end, requires_grad=False)
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        trn = self.transition
        trn_exp = trn.unsqueeze(0).expand(batch_size, *trn.size())
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), trn.size(0))
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = sequence_mask(lens + 1).float()
        trn_scr = trn_scr * mask
        score = trn_scr

        return score

    @staticmethod
    def calc_unary_score(logits, labels, lens):
        labels_exp = labels.unsqueeze(-1)
        scores = torch.gather(logits, 2, labels_exp).squeeze(-1)
        mask = sequence_mask(lens).float()
        scores = scores * mask
        return scores

    def calc_gold_score(self, logits, labels, lens):
        unary_score = self.calc_unary_score(logits, labels, lens).sum(
            1).squeeze(-1)
        binary_score = self.calc_binary_score(labels, lens).sum(1).squeeze(-1)
        return unary_score + binary_score

    def calc_norm_score(self, logits, lens):
        batch_size, seq_len, feat_dim = logits.size()
        # alpha = logits.data.new(batch_size, self.label_size).fill_(-10000.0)
        alpha = logits.new_full((batch_size, self.label_size), -100.0)
        alpha[:, self.start] = 0
        # alpha = Variable(alpha)
        lens_ = lens.clone()

        logits_t = logits.transpose(1, 0)
        for logit in logits_t:
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   *self.transition.size())
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  *self.transition.size())
            trans_exp = self.transition.unsqueeze(0).expand_as(alpha_exp)
            mat = logit_exp + alpha_exp + trans_exp
            alpha_nxt = log_sum_exp(mat, 2).squeeze(-1)

            mask = (lens_ > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            lens_ = lens_ - 1

        alpha = alpha + self.transition[self.end].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def viterbi_decode(self, logits, lens):
        """Borrowed from pytorch tutorial
        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, seq_len, n_labels = logits.size()
        # vit = logits.data.new(batch_size, self.label_size).fill_(-10000)
        vit = logits.new_full((batch_size, self.label_size), -100.0)
        vit[:, self.start] = 0
        # vit = Variable(vit)
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        pointers = []
        for logit in logits_t:
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transition.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transition[self.end].unsqueeze(
                0).expand_as(vit_nxt)

            c_lens = c_lens - 1

        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        # idx = idx.squeeze(-1)
        paths = [idx.unsqueeze(1)]
        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths


class CRFDecoder(nn.Module):
    def __init__(self, crf, label_size, input_dim, input_dropout=0.5):
        super(CRFDecoder, self).__init__()
        self.input_dim = input_dim
        self.input_dropout = nn.Dropout(p=input_dropout)
        self.linear = Linears(in_features=input_dim,
                              out_features=label_size,
                              hiddens=[input_dim // 2])
        self.crf = crf
        self.label_size = label_size

    def forward_model(self, inputs):
        batch_size, seq_len, input_dim = inputs.size()
        output = inputs.contiguous().view(-1, self.input_dim)
        output = self.input_dropout(output)
        # Fully-connected layer
        output = self.linear.forward(output)
        output = output.view(batch_size, seq_len, self.label_size)
        return output

    def forward(self, inputs, labels_mask):
        lens = labels_mask.sum(-1)
        logits = self.forward_model(inputs)
        logits = self.crf.pad_logits(logits)
        scores, preds = self.crf.viterbi_decode(logits, lens)
        return preds

    def score(self, inputs, labels_mask, labels):
        lens = labels_mask.sum(-1)
        logits = self.forward_model(inputs)
        logits = self.crf.pad_logits(logits)
        norm_score = self.crf.calc_norm_score(logits, lens)
        gold_score = self.crf.calc_gold_score(logits, labels, lens)
        loglik = gold_score - norm_score
        return -loglik.mean()

    @classmethod
    def create(cls, label_size, input_dim, input_dropout=0.5):
        return cls(CRF(label_size + 2), label_size, input_dim, input_dropout)


class NerModel(nn.Module, metaclass=abc.ABCMeta):
    """Base class for all BERT Models"""

    @abc.abstractmethod
    def forward(self, batch):
        raise NotImplementedError("abstract method forward must be implemented")

    @abc.abstractmethod
    def score(self, batch):
        raise NotImplementedError("abstract method score must be implemented")

    @abc.abstractmethod
    def create(self, *args, **kwargs):
        raise NotImplementedError("abstract method create must be implemented")

    def get_n_trainable_params(self):
        pp = 0
        for p in list(self.parameters()):
            if p.requires_grad:
                num = 1
                for s in list(p.size()):
                    num = num * s
                pp += num
        return pp


class BiLSTMCRF(NerModel):

    def __init__(self, lstm, crf, device="cuda", embeddings=None):
        super(BiLSTMCRF, self).__init__()
        self.embeddings = embeddings
        self.lstm = lstm
        self.crf = crf
        self.to(device)

    def forward(self, batch):
        if self.embeddings is not None:
            input_embeddings = self.embeddings(batch["texts"])
        else:
            input_embeddings = batch["texts"]
        output, _ = self.lstm.forward(input_embeddings, batch["labels_mask"])
        return self.crf.forward(output, batch["labels_mask"])

    def score(self, batch):
        if self.embeddings is not None:
            input_embeddings = self.embeddings(batch["texts"])
        else:
            input_embeddings = batch["texts"]
        output, _ = self.lstm.forward(input_embeddings, batch["labels_mask"])
        return self.crf.score(output, batch["labels_mask"], batch["labels"])

    @classmethod
    def create(
            cls,
            label_size,
            # BiLSTM params
            embedding_size=300, hidden_dim=100, rnn_layers=1, lstm_dropout=0.3,
            # CRFDecoder params
            crf_dropout=0.5,
            # Global params
            device="cuda",
            use_embeds=False
    ):
        embeddings = None
        if use_embeds:
            embeddings = torch.nn.Embedding(use_embeds, embedding_size)
        lstm = BiLSTM.create(
            embedding_size=embedding_size, hidden_dim=hidden_dim, rnn_layers=rnn_layers, dropout=lstm_dropout)
        crf = CRFDecoder.create(label_size, hidden_dim, crf_dropout)
        return cls(lstm, crf, device, embeddings=embeddings)
