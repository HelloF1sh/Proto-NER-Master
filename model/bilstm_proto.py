import torch
import torch.nn as nn
from .crf import CRF

class BiLSTM_Proto(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels, hidden_dropout_prob=0.2):
        super(BiLSTM_Proto, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.batch_size = 0
        self.words_num = 0
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, batch_first=True,
                            num_layers=1, bidirectional=True)
        self.num_labels = num_labels
        # Maps the output of the LSTM into tag space.
        self.hidden2label = nn.Linear(hidden_dim, num_labels)
        self.init_weights()
        self.prototypes = self.init_proto().cuda()

    def init_weights(self):
        nn.init.uniform_(self.word_embeds.weight)
        nn.init.xavier_normal_(self.hidden2label.weight)
        for weights in [self.lstm.weight_hh_l0, self.lstm.weight_ih_l0]:
            nn.init.orthogonal_(weights)

    def init_proto(self):
        """
        Returns: init prototypes which size is [numlabels, hidden_dim]
        """
        # res = torch.randn([self.num_labels, self.hidden_dim]).cuda()
        res = torch.zeros([self.num_labels, self.hidden_dim]).cuda()
        return res

    def dist(self,x,y,dim):
        return -(torch.pow(x-y,2)).sum(dim)

    def batch_dist(self, inputs, prototypes):
        """
        inputs = [batch_size, max_words_len, hidden_dim]
        prototypes = [proto_num, hidden_dim]
        """
        # we resize inputs to [batch_size * max_words_len, hidden_dim]
        # print(f"inputs size:{inputs.size()}")
        inputs = inputs.contiguous().view(-1, inputs.size(-1)).cuda()
        return self.dist(prototypes.unsqueeze(0), inputs.unsqueeze(1), 2)

    def forward(self, input_ids, attention_mask=None):
        '''
        input_ids:  (batch_size, max_seq_length)
        return: (batch_size, max_seq_length, num_labels)
        '''
        # (batch_size, max_seq_length, word_embedding_dim)
        self.batch_size = input_ids.size(0)
        self.words_num = input_ids.size(-1)
        embeds = self.word_embeds(input_ids)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        lstm_proto = self.batch_dist(inputs=lstm_out, prototypes=self.prototypes)
        # print(f"lstm_proto.size:{lstm_proto.size()}")
        # lstm_feats = self.hidden2label(lstm_out)
        # print(f"lstm_feats.size:{lstm_feats.size()}")
        lstm_feats = lstm_proto.view([self.batch_size, self.words_num, self.num_labels])
        # print(f"lstm_feats:{lstm_feats.size()}")
        return lstm_feats


class BiLSTM_CRF_PROTO(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels, hidden_dropout_prob=0.2, use_crf=False):
        super(BiLSTM_CRF_PROTO, self).__init__()
        self.num_labels = num_labels
        self.use_crf = use_crf
        self.bilstm = BiLSTM_Proto(vocab_size, embedding_dim, hidden_dim, num_labels, hidden_dropout_prob)

        if self.use_crf:
            self.crf = CRF(num_labels)

    def forward(self, input_ids, attention_mask=None, pred_mask=None, input_labels=None):
        '''
        input_ids:  (batch_size, max_seq_length)
        attention_mask:  (batch_size, max_seq_length)
        pred_mask: (batch_size, max_seq_length)
        input_labels: (batch_size, )

        return: (batch_size, max_seq_length)
        '''
        # (batch_size, max_seq_length, num_labels)
        emissions = self.bilstm(input_ids)

        if self.use_crf:
            preds = self.crf.decode(emissions, pred_mask)
            preds = [seq + [-1]*(pred_mask.size(1)-len(seq)) for seq in preds]
            preds = torch.tensor(preds).to(input_ids.device)
        else:
            preds = torch.argmax(emissions, dim=-1)

        output = (preds, )

        if input_labels is not None:
            if self.use_crf:
                loss = -1*self.crf(emissions, input_labels, attention_mask)
            else:
                loss_fct = nn.CrossEntropyLoss()
                if pred_mask is not None:
                    pred_pos = pred_mask.view(-1) == 1
                    logits = emissions.view(-1, self.num_labels)[pred_pos]
                    input_labels = input_labels.view(-1)[pred_pos]
                    loss = loss_fct(logits, input_labels)
                else:
                    loss = loss_fct(emissions.view(-1, self.num_labels), input_labels.view(-1))
            output += (loss, )

        return output #(preds, loss)
