"""
数据与处理部分
"""
from torchtext import data
from torchtext.data import Field, Dataset, Iterator


class Data(object):
    def __init__(self):
        """
        构造函数，进行数据预处理的类
        """
        # 这里注意需要将句子的原始长度返回，是因为encoder模型需要知道每个句子从哪里结束
        self.chinese = Field(sequential=True, use_vocab=True, batch_first=True,
                             init_token='BOS', eos_token='EOS', include_lengths=True, lower=True)
        self.english = Field(sequential=True, use_vocab=True, batch_first=True,
                             init_token='BOS', eos_token='EOS', include_lengths=True, lower=True)
        self.english_list = []
        self.chinese_list = []
        # 英文的字典中的字的数量
        self.english_voca_size = 0
        # 中文的字典中字的数量
        self.chinese_voca_size = 0
        for en, ch in self.generate_token('./data/en-zh.txt'):
            self.english_list.append(en)
            self.chinese_list.append(ch)

    def generate_token(self, filename):
        """
        对指定文件进行分词, 生成器
        :param filename: 指定的需要分词的源文件
        :return: 生成器产生的token
        """
        f = open(filename, 'r', encoding='utf8').readlines()
        for line in f:
            line = line.strip()
            ls = line.split('\t')
            english, chinese = ls[0], ls[1]
            yield english, chinese

    def create_iter(self, batch_size):
        """
        构建迭代器
        :param batch_size: 每批的大小
        :return: iter
        """
        # 定义torchtext中的Field
        fields = [('english', self.english), ('chinese', self.chinese)]
        examples = []
        # 构建中英文example
        for en, ch in zip(self.english_list, self.chinese_list):
            item = [en, ch]
            examples.append(data.Example().fromlist(item, fields))
        # 划分训练集，测试集
        train, test = Dataset(examples=examples, fields=fields).split(split_ratio=0.8)
        self.english.build_vocab(train)
        self.chinese.build_vocab(train)
        self.english_voca_size = len(self.english.vocab)
        self.chinese_voca_size = len(self.chinese.vocab)
        train_iter, test_iter = Iterator.splits((train, test), batch_sizes=(batch_size, len(test)),
                                                sort_key=lambda x: len(x.english), sort_within_batch=True, device=-1)

        return train_iter, test_iter


if __name__ == '__main__':
    data_set = Data()
    x,y = data_set.create_iter(100)
    print(data_set.english.vocab.stoi)
    for i in x:
        print(i)
        print(i.english[0][0])
        print(i.english[1])
        break

