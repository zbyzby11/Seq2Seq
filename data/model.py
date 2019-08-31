"""
模型层的定义，encoder、decoder等组件
"""
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, voca_size, emb_dim, hidden_dim):
        """
        初始化函数
        :param voca_size:在Encoder端的词的字典大小（Encoder端的词汇量）
        :param emb_dim: 在Encoder端需要embedding的维度
        :param hidden_dim: 在Encoder端RNN的隐藏层的维度
        """
        super(Encoder, self).__init__()
        self.voca_size = voca_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(self.voca_size, self.emb_dim)
        # self.dropout = nn.Dropout(0.2)
        self.rnn = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x, seq_len_list):
        """
        前向传播，这里需要传入每句话的长度
        :param x: 输入的句子，批量，长度需要从大到小排序
        :param seq_len_list: 这批句子对应的长度
        :return: 最后一个词时刻的hidden_state
        """
        # x = [batch_size, seq_len]
        bs = x.size(0)
        # x_emb = [batch_size, seq_len, emb_dim]
        x_emb = self.embedding(x)
        # 使用变长的padding，因为需要每个句子EOS的hidden状态
        packed_seq = nn.utils.rnn.pack_padded_sequence(x_emb, seq_len_list, batch_first=True)
        # 将数据输入到GRU
        # output_pack为PackedSequence类型, h_pack就是这批句子最后一个状态的hidden_state
        # h = (num_layers(1) * num_directions(2), batch, hidden_size)
        output_pack, h_pack = self.rnn(packed_seq)
        print('h_pack shape is:', h_pack.shape)
        # 因为是双向RNN，所以取最后一层的前向和后向向量进行cat
        h_back = h_pack[-1, :, :]
        h_forward = h_pack[-2, :, :]
        # h = [batch_size, 2*hidden_size]
        h = torch.cat((h_back, h_forward), dim=1)
        return h


class Decoder(nn.Module):
    def __init__(self, voca_size, emb_size, decoder_dim):
        super(Decoder, self).__init__()
        self.voca_size = voca_size
        self.emb_size = emb_size
        self.decoder_dim = decoder_dim
        self.decoder_embedding = nn.Embedding(self.voca_size, self.emb_size)
        self.decoder_rnn = nn.GRU(self.emb_size, self.decoder_dim, num_layers=1, bidirectional=True, batch_first=True)
        # 全连接层，上一层是decoder的维度，下一层是所有词汇表的大小
        self.fc = nn.Linear(self.decoder_dim, self.voca_size)
    def forward(self, x, init_hidden, ):
        ...


if __name__ == '__main__':
    # gru = nn.GRU(input_size=128, hidden_size=32, num_layers=2, bidirectional=True)
    # x = torch.randn(300, 100, 128)
    # output, h = gru(x)
    # output = output.view(300, 100, 2, 32)
    # h = h.view(2,2,100,32)
    # a = torch.randn(2, 32, 128)
    # b = torch.randn(1,32,128)
    # print(a[-2, :, :].shape)
    # c = torch.cat((a, b), dim=1)
    # print(c.shape)
    a = torch.randn(100, 200, 128)
    l = [10] * 200
    pac = nn.utils.rnn.pack_padded_sequence(a, l)
    print(pac)
