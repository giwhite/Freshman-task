import os
import pickle
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import *

from torch.utils.tensorboard import SummaryWriter
log_dir = './data/'
log_dir_train = os.path.join(log_dir,'train_fig')
log_dir_test = os.path.join(log_dir,'test_fig')
train_wr = SummaryWriter(log_dir=log_dir_train)
test_wr = SummaryWriter(log_dir=log_dir_test)


train_batch_size = 512
test_batch_size = 500
max_len = 260
def tokenize(text):
    # fileters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    fileters = ['"','#','$','%','&','\(','\)','\*','\+',',','-','\.','/',':',';','<','=','>','@'
        ,'\[','\\','\]','^','_','`','\{','\|','\}','~','\t','\n','\x97','\x96','”','“',]
    # sub方法是替换
    text = re.sub("<.*?>"," ",text,flags=re.S)	# 去掉<...>中间的内容，主要是文本内容中存在<br/>等内容
    text = re.sub("|".join(fileters)," ",text,flags=re.S)	# 替换掉特殊字符，'|'是把所有要匹配的特殊字符连在一起
    return [i.strip() for i in text.split()]	# 去掉前后多余的空格

def read_imdb(path='./data/aclImdb', is_train=True):
    reviews, labels = [], []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(path, 'train' if is_train else 'test', label)
        for filename in os.listdir(folder_name):
            with open(os.path.join(folder_name, filename), mode='r', encoding='utf-8') as f:
                reviews.append(tokenize(f.read().strip()))#读出内容
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
        # 构建一个数字映射到单词的词典，方法反向转换，但程序中用不太到
        #self.inversed_dict = dict(zip(self.dict.values(), self.dict.keys()))
    
    def transform(self, sentence, max_len=None):
        """
        根据词典给每个词分配的数字ID，将给定的sentence（字符串序列）转换为数字序列
        max_len：统一文本的单词个数
        """
        if max_len is not None:
            r = [self.PAD] * max_len
        else:
            r = [self.PAD] * len(sentence)
        # 截断文本
        if max_len is not None and len(sentence) > max_len:
            sentence = sentence[:max_len]
        for index, word in enumerate(sentence):
            r[index] = self.to_index(word)
        return np.array(r, dtype=np.int64)

# 建立词表
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

ws = pickle.load(open("./model/ws.pkl", "rb"))
def collate_fn(batch):
    # 手动zip操作，并转换为list，否则无法获取文本和标签了
    offsets = [0]
    batch = list(zip(*batch))
    labels = torch.tensor(batch[1], dtype=torch.int32)
    texts = batch[0]
    texts = torch.tensor([ws.transform(i, max_len) for i in texts])
    del batch
    # 注意这里long()不可少，否则会报错
    return labels.long(), texts.long()
def get_dataloader(train=True):
    if train:
        mode = 'train'
    else:
        mode = "test"
    dataset = myDataset(mode)
    batch_size = train_batch_size if train else test_batch_size
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

class IMDBmodel(nn.Module):
    def __init__(self) -> None:
        super(IMDBmodel,self).__init__()
        self.embedding = nn.Embedding(len(ws), 100, padding_idx=ws.PAD)
        self.fc = nn.Linear(max_len*100,2)
    
    def forward(self,x):
        embed = self.embedding(x)
        embed = embed.view(x.size(0),-1)
        out = self.fc(embed)
        return nn.functional.log_softmax(out,dim = -1)

imdb_model = IMDBmodel()
# 优化器
learning_rate = 0.5
optimizer = optim.Adam(imdb_model.parameters(),lr=learning_rate)

# 交叉熵损失
criterion = nn.CrossEntropyLoss()

def train(epoch):
    mode = True
    i = 1 + epoch*5
    train_dataloader = get_dataloader(mode)
    for idx, (target, input) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = imdb_model(input)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(input), len(train_dataloader.dataset),
                       100. * idx / len(train_dataloader), loss.item()))
            train_wr.add_scalar("train_loss2",loss.item(),i)
            i = i+1
 
 
def test(epoch):
    test_loss = 0
    correct = 0
    mode = False
    imdb_model.eval()
    test_dataloader = get_dataloader(mode)
    with torch.no_grad():
        for target, input in test_dataloader:
            output = imdb_model(input)
            test_loss += F.nll_loss(output, target, reduction="sum")
            pred = torch.max(output, dim=-1, keepdim=False)[-1]
            correct += pred.eq(target.data).sum()
        test_loss = test_loss / len(test_dataloader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_dataloader.dataset),
            100. * correct / len(test_dataloader.dataset)))
        test_wr.add_scalar("test_acci2",100. * correct / len(test_dataloader.dataset),epoch)

if __name__ == '__main__':

    test(0)
    for i in range(20):
        train(i)
        print(
            "训练第{}轮的测试结果-----------------------------------------------------------------------------------------".format(
                i + 1))
        test(i+1)
    train_wr.close()
    test_wr.close()

