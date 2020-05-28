import torch
import torch.nn as nn
import torch.nn.functional as F

#模型定义
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, number_layers, drop_rate, batch_size, bidirect):
        super(LSTMTagger,self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.num_layers = number_layers
        self.drop_rate = drop_rate
        self.batch_size = batch_size
        self.bidirect = bidirect

        self.word_embeddings = nn.Embedding(self.vocab_size,self.embedding_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size = self.embedding_dim, hidden_size = self.hidden_dim, num_layers = self.num_layers, \
                            dropout = self.drop_rate, batch_first=True, bidirectional=self.bidirect)
        ## 如果batch_first为True，输入输出数据格式是(batch, seq_len, feature)
        ## 为False，输入输出数据格式是(seq_len, batch, feature)，
        self.dropout = nn.Dropout(self.drop_rate)

        if self.bidirect:
            self.hidden2tag = nn.Linear(self.hidden_dim*2, self.tagset_size)
        else:
            self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)# hidden to tag

    def forward(self, sentence, lens_batch, is_test=False):
        embeds = self.word_embeddings(sentence)# sentence 是词seq的索引

        if is_test==False:
            self.input_tensor = nn.utils.rnn.pack_padded_sequence(embeds, lens_batch, batch_first=True)
            lstm_out,self.hidden = self.lstm(self.input_tensor)
            lstm_out_pack,_ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

            x = self.dropout(lstm_out_pack)#add
            x = torch.tanh(x)#add
            tag_space = self.hidden2tag(lstm_out_pack)

        else:
            lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1))
            x = self.dropout(lstm_out)
            x = torch.tanh(x)
            tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))

        #tag_scores = F.log_softmax(tag_space, dim=1)
        
        return tag_space
