import torch
import torch.nn as nn
import os
from torch.utils import data
from tqdm import tqdm
from naivenet import TextClassificationModel
import imdbdataset
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
lr = 0.01
batch = 256
epochs = 30
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_dim = 50
num_class = 2
nettype = 2#更改这里可以改变网络的类型

def train_epoch(model, dataloader, criterion: dict, optimizer,
                scheduler, epoch, device):
    model.train()
    bar = tqdm(dataloader)
    bar.set_description(f'epoch {epoch:2}')
    correct, total = 0, 0
    last_loss=0
    for dat in bar:
        label, text, offsets=dat
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total+=len(label)
        correct+=(predicted_label.argmax(1) == label).sum().item()

        bar.set_postfix_str(f'lr={scheduler.get_last_lr()[0]:.4f} acc={correct / total * 100:.2f} loss={loss.item():.2f}')
        last_loss=loss.item()
    writer.add_scalar('Accuracy/train',correct / total * 100, epoch)
    writer.add_scalar('Loss/train',last_loss, epoch)
    scheduler.step()


def test_epoch(model, dataloader, device, epoch):
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for dat in dataloader:
            labx, text, offset=dat
            predicted_label = model(text, offset)
            correct+=(predicted_label.argmax(1) == labx).sum().item()
            total+=len(labx)

        print(f' val acc: {correct / total * 101:.2f}')
        writer.add_scalar('Accuracy/test',correct / total * 100, epoch)


def main():
    root='./aclImdb'
    workspace_dir='./models'
    trainset=imdbdataset.get_dataset(root,mode='train')
    trainloader = data.DataLoader(trainset,#修改了下路径
                                  batch_size=batch, shuffle=True, num_workers=10,collate_fn=imdbdataset.collate_batch)
    testloader = data.DataLoader(imdbdataset.get_dataset(root,mode='vald'),
                                 batch_size=batch, shuffle=True, num_workers=10,collate_fn=imdbdataset.collate_batch)
    vocab_size=len(trainset.vocab)
    
    model = TextClassificationModel(vocab_size=vocab_size+4,embed_dim=embed_dim,num_class=num_class,typeOfNet=nettype,dropout=0.4,device=device).to(device)
    #model.load_state_dict(torch.load(os.path.join(workspace_dir, 'dcgan_g.pth')))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95,
                                                last_epoch=-1)
    criterion =nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_epoch(model, trainloader, criterion, optimizer,
                    scheduler, epoch, device)
        test_epoch(model, testloader, device, epoch)
        if (epoch+1) % 4 == 0:
            torch.save(model.state_dict(), os.path.join(workspace_dir, f'dcgan_g.pth'))
    writer.flush()
    


if __name__ == '__main__':
    main()