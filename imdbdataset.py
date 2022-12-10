import os, sys, glob, shutil, json
import numpy as np

import torch
from torch.utils.data.dataset import Dataset


class imdbdataset(Dataset):
    def __init__(self,root,transform=None):
        self.path=root
        vocab=dict()
        with open("./aclImdb/imdb.vocab",'r') as f:
            i=0
            for line in f:
                vocab[line.split('\n')[0]]=i
                i+=1
        self.vocab=vocab
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    def __getitem__(self, index):
        idx=index%12500
        tar=''
        label=np.zeros(1)
        if index<12500:
            tar=os.path.join(self.path,'neg')
            for i in range(1,6):
                pend=os.path.join(tar,str(idx)+'_'+str(i)+'.txt')
                if os.path.exists(pend):
                    tar=pend
                    break
        else:
            label=np.ones(1)
            tar=os.path.join(self.path,'pos')
            for i in range(5,11):
                pend=os.path.join(tar,str(idx)+'_'+str(i)+'.txt')
                if os.path.exists(pend):
                    tar=pend
                    break
        line=open(tar, 'r', encoding='UTF-8').readline()
        line=line.lower()
        last=''
        lst=[]
        for u in line:
            if u.isalpha()==True or u=='-' or u=="\'" or u==' ':
                last+=u
            elif u=='?' or u== '!':
                last+=' '
                last+=u
        strss=last.split(' ')
        for myword in strss:
            if self.vocab.get(myword):
                lst.append(self.vocab[myword])
        retl=0 if index<12500 else 1
        return retl,lst

    def __len__(self):
        return 25000

def collate_batch(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(_label)
         processed_text = torch.tensor(_text, dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list, offsets#删去todevice


def get_dataset(root,mode):
    if mode=='train':
        cmt_pth=os.path.join(root,'train')
    else:
        cmt_pth=os.path.join(root,'test')
    return imdbdataset(cmt_pth)

    
if __name__ == '__main__':
    root='./aclImdb'
    train_loader = torch.utils.data.DataLoader(get_dataset(root,'train'), 
        batch_size=10, # 每批样本个数
        shuffle=True, # 是否打乱顺序
        num_workers=10, # 读取的线程个数
        collate_fn=collate_batch,
    )
    for data in train_loader:
        print(data)
        break