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
max_epoch = 15
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
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

ls = []

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
            ls.append(max_len)
        else:
            ls.append(len(sentence))
        for index, word in enumerate(sentence):
            r[index] = self.to_index(word)
        r2 = r[:len(sentence)]
        return np.array(r2, dtype=np.int64)
    
ws = pickle.load(open("./model/ws.pkl", "rb"))#这是词表
class TextRNN(nn.Module):
    def __init__(self) -> None:
        super(TextRNN,self).__init__()
        self.embedding_size = 256
        self.hidden_dim = 100
        self.out_dim = 2
        self.layernum = 2
        self.state = None
        self.embedding = nn.Embedding(len(ws),self.embedding_size,padding_idx=ws.PAD)
        self.rnn = nn.RNN(self.embedding_size,self.hidden_dim,num_layers = self.layernum,dropout = 0.3, batch_first=True)
        self.gru = nn.GRU(self.embedding_size, self.hidden_dim,batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim,self.out_dim)
    def forward(self,x):#,num,mode):#这里还要传入字长
        embed_x = self.embedding(x)
        #the shape of embed_x is [64,260,256],which equals to [max_sentence_len,batch_size,vec_dim_per_word]
        '''这里是使用gather的版本'''
        global ls
        pack = torch.nn.utils.rnn.pack_padded_sequence(embed_x, ls, batch_first=True)
        output,_ = self.gru(pack,self.state)
        output, others = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = output[:,-1,:].squeeze(dim=1)
        output = self.fc1(output)
        
        
        return F.softmax(output.squeeze(1),dim=-1)





'''
填充的全局函数

'''
PAD_token = 0
def pad_seq(seq, seq_len, max_length):
    seq = list(seq)
    if seq_len != 260:
        for _ in range(max_length - seq_len):
            seq.append(PAD_token)
    return seq


def collate_fn(batch):
    # 手动zip操作，并转换为list，否则无法获取文本和标签了
    batch = list(zip(*batch))
    labels = torch.tensor(batch[1], dtype=torch.int32)
    texts = batch[0]
    texts = [ws.transform(i, max_len) for i in texts]
    texts = sorted(texts, key = lambda tp: len(tp), reverse=True)
    global ls
    ls = sorted(ls, key = lambda tp: tp, reverse=True)
    #texts = torch.tensor(np.array([ws.transform(i, max_len) for i in texts]))
    del batch
    pad_seqs = []  # 填充后的数据
    for i in texts:
        pad_seqs.append(pad_seq(i, len(i), max_len))
    # 注意这里long()不可少，否则会报错
    texts = torch.tensor(pad_seqs)
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
imdb_model.to(device)
# 优化器
learning_rate = 0.01
optimizer = optim.Adam(imdb_model.parameters(),lr=learning_rate)
scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,
                                                        T_max =  max_epoch)

# 交叉熵损失
criterion = nn.CrossEntropyLoss()
criterion.to(device)
def train(epoch):
    mode = True

    train_dataloader = get_dataloader(mode)
    step = 0
    pbar = tqdm(train_dataloader,desc='epoch:{}'.format(epoch))
    for target, input in pbar:

        optimizer.zero_grad()
        #imdb_model.train()
        target.to(device)
        input.to(device)
        output = imdb_model(input)#,step,mode)#这里传入的num出问题了
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
        global ls
        ls.clear()
    scheduler.step()

 
 
def test():
    test_loss = 0
    correct = 0

    mode = False
    imdb_model.eval()
    test_dataloader = get_dataloader(mode)
    test_step = 0
    with torch.no_grad():
        for target, input in tqdm(test_dataloader):
            target.to(device)
            input.to(device)
            output = imdb_model(input)#,test_step,mode)
            test_loss += criterion(output, target)
            pred = torch.max(output, dim=-1, keepdim=False)[-1]
            correct += pred.eq(target.data).sum()
            test_step += 1
            ls.clear()#每个batch清理一次

        test_loss = test_loss / len(test_dataloader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_dataloader.dataset),
            100. * correct / len(test_dataloader.dataset)))

    
       

if __name__ == '__main__':

    #test()
    for i in range(max_epoch):
        train(i)
        print(
            "训练第{}轮的测试结果-----------------------------------------------------------------------------------------".format(
                i + 1))
        test()
