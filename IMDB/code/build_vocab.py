import re#正则函数，方便剔除一些符号
import os
from torch.utils.data import Dataset
import numpy as np
from tqdm import *
import pickle
def tokenize(text):
    # fileters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    fileters = ['"','#','$','%','&','\(','\)','\*','\+',',','-','\.','/',':',';','<','=','>','@'
        ,'\[','\\','\]','^','_','`','\{','\|','\}','~','\t','\n','\x97','\x96','”','“',]
    # sub方法是替换
    text = re.sub("<.*?>"," ",text,flags=re.S)	# 去掉<...>中间的内容，主要是文本内容中存在<br/>等内容
    text = re.sub("|".join(fileters)," ",text,flags=re.S)	# 替换掉特殊字符，'|'是把所有要匹配的特殊字符连在一起
    return [i.strip() for i in text.split()]

def read_imdb(path='./data/aclImdb', is_train=True):
    reviews, labels = [], []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(path, 'train' if is_train else 'test', label)
        for filename in os.listdir(folder_name):
            with open(os.path.join(folder_name, filename), mode='r', encoding='utf-8') as f:
                reviews.append(tokenize(f.read().strip()))#读出内容,然后把前后的空格给删除
                labels.append(1 if label == 'pos' else 0)
    return reviews, labels

class myDataset(Dataset):
    def __init__(self,mode):
        super(myDataset,self).__init__()
        if(mode == "train"):
            self.reviews,self.labels = read_imdb()
        else:
            self.reviews,self.labels = read_imdb(is_train=False)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        label = self.labels[index]
        review = self.reviews[index]
        return review,label

class Word2Sequence:
    # 未出现过的词
    UNK_TAG = "UNK"
    PAD_TAG = "PAD"
    # 填充的词
    UNK = 0
    PAD = 1
    #因为后面
    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.count = {}
 
    def to_index(self, word):
        """word -> index"""
        return self.dict.get(word, self.UNK)

    def __len__(self):
        return len(self.dict)
 
    def fit(self, sentence):
        """count字典中存储每个单词出现的次数"""
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1
            '''是对进行计数word出现的频率进行统计，
            当word不在words时，返回值是0，
            当word在words中时，返回对应的key的值
            后面的+1，是计数
            以此进行累计计数'''
 
    def build_vocab(self, min_count=None, max_count=None, max_feature=None):
        """
        构建词典
        只筛选出现次数在[min_count,max_count]之间的词
        词典最大的容纳的词为max_feature，按照出现次数降序排序，要是max_feature有规定，出现频率很低的词就被舍弃了
        """
        if min_count is not None:
            self.count = {word: count for word, count in self.count.items() if count >= min_count}
 
        if max_count is not None:
            self.count = {word: count for word, count in self.count.items() if count <= max_count}
 
        if max_feature is not None:
            self.count = dict(sorted(self.count.items(), lambda x: x[-1], reverse=True)[:max_feature])
        # 给词典中每个词分配一个数字ID
        for word in self.count:
            self.dict[word] = len(self.dict)#取出来了就是值了
            '''这里的思路就是将这个dict中的word分配一个id，这个id是根据字典的大小变化的，这也是为什么后面加上了len'''
    
    def transform(self, sentence, max_len=None):
        """
        根据词典给每个词分配的数字ID，将给定的sentence（字符串序列）转换为数字序列
        max_len：统一文本的单词个数
        """
        if max_len is not None:
            r = [self.PAD] * max_len#填充为1
        else:
            r = [self.PAD] * len(sentence)
        # 截断文本
        if max_len is not None and len(sentence) > max_len:
            sentence = sentence[:max_len]#取前max_len个元素
        for index, word in enumerate(sentence):
            r[index] = self.to_index(word)#这里的to_index是放了一个get函数，原理同上，没找到用self.unk填充
        return np.array(r, dtype=np.int64)#转换成array类型，这是一个数字序列

def fit_save_word_sequence():
    word_to_sequence = Word2Sequence()
    train_path = [os.path.join('./data/aclImdb/', i) for i in ["train/neg", "train/pos"]]
    # total_file_path_list存储总的需要读取的txt文件
    total_file_path_list = []
    for i in train_path:
        total_file_path_list.extend([os.path.join(i, j) for j in os.listdir(i)])
    # tqdm是显示进度条的
    for cur_path in tqdm(total_file_path_list, ascii=True, desc="fitting"):
        word_to_sequence.fit(tokenize(open(cur_path, encoding="utf-8").read().strip()))
    word_to_sequence.build_vocab()
    # 对wordSequesnce进行保存
    pickle.dump(word_to_sequence, open("./model/ws.pkl", "wb"))
    #创建一个词表的pkl文件