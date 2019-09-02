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
        # h = (batch, num_layers(1) * num_directions(2), hidden_size)
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
        self.decoder_rnn = nn.GRU(self.emb_size, self.decoder_dim, num_layers=1, batch_first=True)
        # 全连接层，上一层是decoder的维度，下一层是所有词汇表的大小
        self.fc = nn.Linear(self.decoder_dim, self.voca_size)

    def forward(self, x, init_hidden_state):
        """
        在这里
        1. decoder部分的emb_dim的维度需要等于encoder部分的emb_dim
        2. decoder部分的decoder_dim（即decoder部分的hidden_dim）需要等于2*encoder_hidden_dim，这个是自定义模型决定的
        3. decoder部分是一个字一个字的输出的
        """
        # x = [batch_size, 1], 这个就是decoder的开始token：'SOS'
        # init_hidden_state是encoder部分的context向量
        # init_hidden_state = [batch_size, 1, encoder_hidden_dim]
        bs = x.size(0)
        # x_emb = [batch_size, 1, emb_dim]， 将'SOS'进行embedding
        x_emb = self.embedding(x)
        # 通过rnn层
        # x_emb = [batch_size, 1, emb_dim]
        # init_hidden_state = [batch_size, 1, 2*encoder_hidden_dim]
        # output = [batch_size, seq_len=1, num_directions(1) * decoder_dim]
        output, _ = self.decoder_rnn(x_emb, init_hidden_state)
        # output = [batch_size, decoder_dim]
        output = output[:, -1, :]
        # out = [batch_size, voca_size]，就是下一个词
        out = self.fc(output)
        return out


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, per_src_length):
        """
        输入两个batch，一个是待翻译的batch，一个是翻译后的batch
        :param src: 带翻译的batch
        :param trg: 翻译好的batch
        :param per_src_length: 使用pack_padded_sequence机制传入的每批src的真正长度
        :return: 模型翻译出来的整个序列
        """
        # src = [batch_size, seq_len]
        # trg = [batch_size, seq_len]
        src_batch_size = src.size(0)
        # 翻译出来的序列的最大长度
        trg_max_length = trg.size(1)
        # 翻译的词汇的大小
        trg_voca_size = self.decoder.voca_size
        # 经过一层encoder层
        # encoder_output = [batch_size, 2*encoder_hidden_size]
        encoder_output = self.encoder(src, per_src_length)
        # 获取到trg的每批的'BOS'
        # init_tokens =

if __name__ == '__main__':
    gru = nn.GRU(input_size=128, hidden_size=32, num_layers=2, bidirectional=True)
    x = torch.randn(300, 100, 128)
    output, h = gru(x)
    # output = output.view(300, 100, 2, 32)
    # h = h.view(2,2,100,32)
    # a = torch.randn(2, 32, 128)
    # b = torch.randn(1,32,128)
    # print(a[-2, :, :].shape)
    # c = torch.cat((a, b), dim=1)
    # print(c.shape)
    # a = torch.randn(100, 200, 128)
    # l = [10] * 200
    # pac = nn.utils.rnn.pack_padded_sequence(a, l)
    # print(pac)
