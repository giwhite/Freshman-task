import numpy as np
import pickle
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import re
import os
from tqdm import tqdm
from torch import optim
#超参数
train_batch_size = 64
test_batch_size = 512
max_len = 260
def tokenize(text):
    # fileters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    fileters = ['"','#','$','%','&','\(','\)','\*','\+',',','-','\.','/',':',';','<','=','>','@'
        ,'\[','\\','\]','^','_','`','\{','\|','\}','~','\t','\n','\x97','\x96','”','“',]
    # sub方法是替换
    text = re.sub("<.*?>"," ",text,flags=re.S)	# 去掉<...>中间的内容，主要是文本内容中存在<br/>等内容
    text = re.sub("|".join(fileters) , " " , text,flags=re.S)	# 替换掉特殊字符，'|'是把所有要匹配的特殊字符连在一起
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
    UNK_TAG = "UNK"
    PAD_TAG = "PAD"
    UNK = 1
    PAD = 0

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.count = {}
 
    def to_index(self, word):
        return self.dict.get(word, self.UNK)

    def __len__(self):
        return len(self.dict)
 
    def fit(self, sentence):
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1
 
    def build_vocab(self, min_count=None, max_count=None, max_feature=None):
        if min_count is not None:
            self.count = {word: count for word, count in self.count.items() if count >= min_count}
 
        if max_count is not None:
            self.count = {word: count for word, count in self.count.items() if count <= max_count}
 
        if max_feature is not None:
            self.count = dict(sorted(self.count.items(), lambda x: x[-1], reverse=True)[:max_feature])

        for word in self.count:
            self.dict[word] = len(self.dict)
    
    def transform(self, sentence, max_len=None):
        if max_len is not None:
            r = [self.PAD] * max_len
        else:
            r = [self.PAD] * len(sentence)
        if max_len is not None and len(sentence) > max_len:
            sentence = sentence[:max_len]
        for index, word in enumerate(sentence):
            r[index] = self.to_index(word)
        return np.array(r, dtype=np.int64)
    
ws = pickle.load(open("./model/ws.pkl", "rb"))#这是词表
class TextRNN(nn.Module):
    def __init__(self) -> None:
        super(TextRNN,self).__init__()
        self.embedding_size = 256
        self.hidden_dim = 100
        self.out_dim = 2
        self.embedding = nn.Embedding(len(ws),self.embedding_size,padding_idx=ws.PAD)
        self.rnn = nn.RNN(self.embedding_size,self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim,self.out_dim)
    
    def forward(self,x):
        embed_x = self.embedding(x)
        #the shape of embed_x is [260,64,256],which equals to [max_sentence_len,batch_size,vec_dim_per_word]
        out,h = self.rnn(embed_x.T)
        #the shape of out is [260,64,100],which equals to [max_len,batch_size,hidden_dim]
        #the shape of h is [1,64, 100],which  equals to [1,batch_size,hidden_len]
        
        #这个地方是确定以下最后一层的数据和记录了每一层数据的out的最新的那层是不是一样的
        assert torch.equal(out[-1, :, ], h.squeeze(0))
        return F.softmax(self.fc(h.squeeze(0)),dim = -1)


    
class TextCNN(nn.Module):
    def __init__(self) -> None:
        super(TextCNN,self).__init__()
        self.windows_size = [3,4,5]
        self.embedding_size = 256
        self.feature_size = 100
        self.max_len = 260
        self.embedding = nn.Embedding(len(ws), self.embedding_size, padding_idx=ws.PAD)#len（ws）是总的词量
        #输入的batch_size 是64,然后截取的长度是260，那么这个输入的矩阵就是512*260，然后embedding层对这里面的多有词建立了一个100维的矩阵
        #然后这个100就是词向量的维度
        self.conv1ds = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=self.embedding_size,out_channels=self.feature_size,kernel_size=h),        
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=self.max_len-h+1)
            )
            for h in self.windows_size
        ])#尽量不要使用
        self.fc = nn.Linear(self.feature_size*len(self.windows_size),2)
        #使用的每个channel都是1，但是有好几个不同维度的kernelsize
    def forward(self, x):
        embed_x = self.embedding(x)
        embed_x = embed_x.permute(0,2,1)
        out = [conv(embed_x) for conv in self.conv1ds]
        out = torch.cat(out,dim = 1)
        out = out.view(-1, out.size(1))
        out = self.fc(out)
        return out
def collate_fn(batch):
    # 手动zip操作，并转换为list，否则无法获取文本和标签了

    batch = list(zip(*batch))
    labels = torch.tensor(batch[1], dtype=torch.int32)
    texts = batch[0]
    texts = torch.tensor(np.array([ws.transform(i, max_len) for i in texts]))
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

imdb_model = TextRNN()
# 优化器
learning_rate = 0.0001
optimizer = optim.Adam(imdb_model.parameters(),lr=learning_rate)

# 交叉熵损失
criterion = nn.CrossEntropyLoss()

def train(epoch):
    mode = True

    train_dataloader = get_dataloader(mode)
    step = 0
    pbar = tqdm(train_dataloader,desc='epoch:{}'.format(epoch))
    for target, input in pbar:
        optimizer.zero_grad()
        #imdb_model.train()
        output = imdb_model(input)
        #loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        step +=1
        #if step % 10 == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, step * len(input), len(train_dataloader.dataset),
        #               100. * step / len(train_dataloader), loss.item()))
        pbar.set_postfix({"loss": loss.item()})
         
 
 
def test():
    test_loss = 0
    correct = 0
    mode = False
    imdb_model.eval()
    test_dataloader = get_dataloader(mode)
    with torch.no_grad():
        for target, input in tqdm(test_dataloader):
            output = imdb_model(input)
            test_loss += criterion(output, target)
            pred = torch.max(output, dim=-1, keepdim=False)[-1]
            correct += pred.eq(target.data).sum()
        test_loss = test_loss / len(test_dataloader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_dataloader.dataset),
            100. * correct / len(test_dataloader.dataset)))
       

if __name__ == '__main__':

    #test()
    for i in range(20):
        train(i)
        print(
            "训练第{}轮的测试结果-----------------------------------------------------------------------------------------".format(
                i + 1))
        test()
