import pickle
import torch
import torch.nn.functional as Fc
from torch.utils.data import DataLoader,Dataset
from build_vocab import read_imdb,Word2Sequence
from torch import nn

max_len = 260
train_batch_size = 512
test_batch_size = 500
ws = pickle.load(open("./model/ws.pkl", "rb"))#这里ws还是word2sequence的类
def collate_fn(batch):
    # 手动zip操作，并转换为list，否则无法获取文本和标签了
    batch = list(zip(*batch))
    labels = torch.tensor(batch[1], dtype=torch.int32)
    texts = batch[0]
    texts = torch.tensor([ws.transform(i, max_len) for i in texts])#在收集数据的时候进行截取
    del batch
    # 注意这里long()不可少，否则会报错
    return labels.long(), texts.long()

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
#这是集中写了一个获取data的函数
class IMDBmodel(nn.Module):
    def __init__(self) -> None:
        super(IMDBmodel,self).__init__()
        self.embedding = nn.Embedding(len(ws), 100, padding_idx=ws.PAD)
        self.fc = nn.Linear(max_len*100,2)
    
    def forward(self,x):
        embed = self.embedding(x)
        embed = embed.view(x.size(0),-1)
        out = self.fc(embed)
        return Fc.log_softmax(out,dim = -1)

def get_dataloader(train=True):
    if train:
        mode = 'train'
    else:
        mode = "test"
    dataset = myDataset(mode)
    batch_size = train_batch_size if train else test_batch_size
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
'''
或者也可以这样写：
dataset_train = myDataset('train')
train_loader = DataLoader(dataset_train,batch_size = batch_size,shuffle = True, collate_fn = collate_fn)
dataset_test = myDataset('test')
test_loader = DataLoader(dataset_test,batch_size = batch_size,shuffle = True,collate_fn = collate_fn) 
'''
