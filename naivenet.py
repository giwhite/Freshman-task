
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class, typeOfNet, dropout,device):
        super(TextClassificationModel, self).__init__()
        self.embed_dim=embed_dim
        self.vocab_size=vocab_size
        self.num_class=num_class
        self.maxlen=260
        self.Cout=128
        self.n_layers=1
        self.hidden_dim=15
        self.filters=[3,4,5]
        self.Hout=[]
        for u in self.filters:
            self.Hout.append(self.maxlen-u+1)
        self.typeOfNet=typeOfNet

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.vocab_size-2)
        
        self.fc = nn.Linear(self.embed_dim*self.maxlen, self.num_class)
        

        self.cnns=[]
        for u in self.filters:
            self.cnns.append(nn.Conv2d(1,self.Cout,(u,embed_dim)))
        self.link=nn.Linear(len(self.filters)*self.Cout, self.num_class)
        self.dropout = nn.Dropout(dropout)
        self.device=device

        self.rnn = nn.LSTM(embed_dim, self.hidden_dim, num_layers=self.n_layers,batch_first=True,
                           bidirectional=True)

        self.linrnn = nn.Linear(self.hidden_dim * 2, num_class)
        

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        
    def initdict(self):
        vocab=dict()
        with open("./aclImdb/imdb.vocab",'r') as f:
            i=0
            for line in f:
                vocab[i]=line.split('\n')[0]
                i+=1
        self.dict=vocab

    def pretask(self, text, offsets):
        #self.initdict()
        nptext=text.clone().detach().cpu().numpy()
        text=np.array(nptext).tolist()
        fin=[]
        offsets=offsets.clone().detach().cpu().numpy()
        beg=np.array(offsets).tolist()
        n=len(beg)
        beg.append(len(text))
        maxlen=self.maxlen
        for i in range(n):
            lst=[]
            for j in range(beg[i],min(beg[i+1],maxlen+beg[i])):
                lst.append(text[j])
            for j in range(max(0,maxlen-beg[i+1]+beg[i])):
                lst.append(self.vocab_size-2)
            fin.append(lst)

        return fin

    def forward(self, text, offsets):
        fin=self.pretask(text,offsets)
        embedded = self.embedding(torch.LongTensor(fin))#设置paddingidx为vocab_size-1
        embedded=embedded.to(self.device)
        if self.typeOfNet==0:
            out = self.fc(embedded.view(embedded.shape[0],-1))
            return nn.functional.log_softmax(out,dim = -1)
        elif self.typeOfNet==1:
            #embedded=self.dropout(embedded)
            embedded=embedded.unsqueeze(1)#扩张一个维度
            temp=[F.relu(conv(embedded)).squeeze(3) for conv in self.cnns]
            pooled = [F.max_pool1d(conv,conv.shape[2]).squeeze(2) for conv in temp]
            cat = self.dropout(torch.cat(pooled, dim=1))
            logits=self.link(cat)
            return logits
        else:
            embedded = self.dropout(embedded)
            output, (hidden, cell) = self.rnn(embedded)

            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

            return self.linrnn(hidden.squeeze(0))
