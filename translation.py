"""
进行翻译的主文件,英文翻译成中文
"""
import torch
from torch import nn, optim
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq
from data_processing import Data


class Translation(object):
    def __init__(self,
                 training_times=100,
                 lr=0.01,
                 batch_size=32,
                 emb_size=128,
                 encoder_hiddein_size=100,
                 decoder_hidden_size=200):
        self.data = Data()
        self.batch_size = batch_size
        self.train_iter, self.test_iter = self.data.create_iter(self.batch_size)
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hiddein_size
        self.decoder_hidden_size = decoder_hidden_size
        self.english_voca_size = self.data.english_voca_size
        self.chinese_voca_size = self.data.chinese_voca_size
        self.training_times = training_times
        self.lr = lr
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.encoder = Encoder(self.english_voca_size, self.emb_size, self.encoder_hidden_size).to(self.device)
        self.decoder = Decoder(self.chinese_voca_size, self.emb_size, self.decoder_hidden_size).to(self.device)
        self.seq2seq = Seq2Seq(self.encoder, self.decoder).to(self.device)
        self.optimizer = optim.Adam(self.seq2seq.parameters(), lr=self.lr)
        # 这里需要注意的是需要在seq_length上做交叉熵
        # self.createon = nn.CrossEntropyLoss()

    def train(self):
        total_loss = 0.0
        print('中文的词汇大小：', self.chinese_voca_size)
        en_dict = self.data.english.vocab.stoi
        pad_id = self.data.english.vocab.stoi['<pad>']
        # print(pad_id)
        createon = nn.CrossEntropyLoss(ignore_index=pad_id)
        for epoch in range(self.training_times):
            self.seq2seq.train()
            flag = True
            for index, i in enumerate(self.train_iter):
                # chinese和english当中包含了当前批次中每句话的原始长度
                chinese = i.chinese
                english = i.english
                # 源语言的批次中的每句话的长度
                english_seq_length = english[1].to(self.device)
                # 源语言的batch向量
                src = english[0].to(self.device)
                # 目标语言的batch向量
                trg = chinese[0].to(self.device)
                # print('src: ',src.shape)
                # print('trg: ',trg.shape)
                # out = [batch_size, trg_max_length, trg_voca_size]
                out = self.seq2seq(src, trg, english_seq_length)
                # print('out: ',out.shape)
                # out = [-1, trg_voca_size],这里是需要对每个预测的词做交叉熵
                out = out[:, 1:, :].contiguous().view(-1, self.chinese_voca_size)
                # print(out.shape)
                # 这里的target参数需要是[-1],代表了每个批次出来的真正的词
                loss = createon(out, target=trg[:, 1:].contiguous().view(-1).cpu())
                # print(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print('epoch {}|| step {}|| loss is :{}'.format(epoch, index, loss.item()))
                # if flag:
                #     print('epoch {}|| loss is :{}'.format(epoch, loss.item()))
                #     flag = False
                self.seq2seq.eval()
                if index % 5 == 0:
                    l = []
                    s = 'BOS i am a chinese . EOS'
                    valid = s.split(' ')
                    for token in valid:
                        if token in en_dict:
                            l.append(en_dict[token])
                        else:
                            l.append(en_dict['<unk>'])
                    valid_tensor = torch.tensor(l).unsqueeze(0).to(self.device)
                    output = self.seq2seq(valid_tensor, trg, [len(valid)])
                    print(output)
                    # for t in self.test_iter:
                    #     o = self.seq2seq(t.english[0], trg, t.english[1])
                    #     print(o)


if __name__ == '__main__':
    s = Translation()
    s.train()
