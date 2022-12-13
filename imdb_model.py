# -*-coding:utf-8-*-
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import dataset_train
from vocab import Vocab
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

global j
j = 0
voc_model = pickle.load(open("./models/vocab1.pkl", "rb"))
sequence_max_len = 300
Vocab()
writer = SummaryWriter('./log')
epochs = 20
learning_rate = 0.001
train_batch_size = 128
test_batch_size = 128
loss_fn = nn.CrossEntropyLoss()
filter_num = 200
embedding_dim = 300
hid_size = 300
kernel_sizes = [2, 3, 4]


def collate_fn(batch):
    reviews, labels = zip(*batch)
    reviews = torch.LongTensor(reviews)
    labels = torch.LongTensor(labels)
    return reviews, labels


def get_dataset(train=True):
    return dataset_train.ImdbDataset(train)


def get_dataloader(train=True):
    imdb_dataset = get_dataset(train)
    batch_size = train_batch_size if train else test_batch_size
    return DataLoader(imdb_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


class ImdbModel(nn.Module):
    def __init__(self):
        super(ImdbModel, self).__init__()
        self.embedding = nn.EmbeddingBag(num_embeddings=len(voc_model), embedding_dim=300, sparse=True)
        self.fc = nn.Linear(300, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text):
        embedded = self.embedding(text)
        out = self.fc(embedded)
        out = self.dropout(out)
        return out


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(voc_model), embedding_dim=embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, filter_num, (kernel, embedding_dim), padding=(2, 0))
                                    for kernel in kernel_sizes])
        self.fc = nn.Linear(filter_num * len(kernel_sizes), 2)
        self.dropout = nn.Dropout(p=0.3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)  # (128, 300, 300)
        x = x.unsqueeze(1)  # (128, 1, 300, 300)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)  # (128, 600)
        x = self.dropout(x)
        x = self.fc(x)  # (128, 2)
        return self.softmax(x)


class Rnn(nn.Module):
    def __init__(self):
        super(Rnn, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(voc_model), embedding_dim=embedding_dim)
        self.RNN = nn.RNN(embedding_dim, hid_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hid_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.embedding(x)  # (128, 300, 300)
        output, h_n = self.RNN(x)
        output = output[:, :, -1]
        out = self.dropout(output)
        out = self.fc(out)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = input("请选择模型(1:RNN/2:TextCNN/3:Linear):")
if model == "1":
    imdb_model = Rnn().to(device)
elif model == "2":
    imdb_model = TextCNN().to(device)
else:
    imdb_model = ImdbModel().to(device)
optimizer = torch.optim.Adam(imdb_model.parameters(), lr=learning_rate)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


def train(imdb_model, epoch):
    global j
    imdb_model.train()
    train_dataloader = get_dataloader(train=True)
    bar = tqdm(train_dataloader, total=len(train_dataloader))
    for idx, (data, target) in enumerate(bar):
        j += 1
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = imdb_model(data)
        loss = loss_fn(output, target)
        loss.backward()
        writer.add_scalar('train_loss', loss, j)
        optimizer.step()
        bar.set_description("epoch:{}  idx:{}   loss:{:.6f}".format(i, idx, loss.item()))


def tst(imdb_model, epoch):
    test_loss = 0
    correct = 0
    imdb_model.eval()
    test_dataloader = get_dataloader(train=False)
    with torch.no_grad():
        for data, target in tqdm(test_dataloader):
            data = data.to(device)
            target = target.to(device)
            output = imdb_model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        accuracy = float(correct) / len(test_dataloader.dataset)
        if (i + 1) % 2 == 0:
            writer.add_scalar('accuracy', accuracy, k)
        test_loss /= len(test_dataloader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_dataloader.dataset),
            100. * correct / len(test_dataloader.dataset)))


if __name__ == '__main__':
    for i in range(epochs):
        train(imdb_model, epochs)
        if (i + 1) % 2 == 0:
            k = (i + 1) / 2
            tst(imdb_model, epochs)
        scheduler.step()
