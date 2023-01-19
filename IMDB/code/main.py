from estb_model import IMDBmodel,get_dataloader,torch,Fc,Word2Sequence
from torch import optim,nn
imdb_model = IMDBmodel()
# 优化器
learning_rate = 2
optimizer = optim.Adam(imdb_model.parameters(),lr=learning_rate)
# 交叉熵损失
criterion = nn.CrossEntropyLoss()

def train(epoch):
    mode = True
    train_dataloader = get_dataloader(mode)
    for idx, (target, input) in enumerate(train_dataloader):#这里返回的idx是batch的个数，
        optimizer.zero_grad()#清零
        output = imdb_model(input)
        loss = Fc.nll_loss(output, target)#计算差值
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}'.format(
                epoch, idx * len(input), len(train_dataloader.dataset),
                       100. * idx / len(train_dataloader), loss.item()))
            #torch.save(imdb_model.state_dict(), "model/mnist_net.pkl")
            #torch.save(optimizer.state_dict(), 'model/mnist_optimizer.pkl')
 
 
def test():
    test_loss = 0
    correct = 0
    mode = False
    imdb_model.eval()
    test_dataloader = get_dataloader(mode)
    with torch.no_grad():
        for target, input in test_dataloader:
            output = imdb_model(input)
            test_loss += Fc.nll_loss(output, target, reduction="sum")#这个计算是总的loss
            pred = torch.max(output, dim=-1, keepdim=False)[-1]
            '''Returns a namedtuple (values, indices) where values is the maximum value of
             each row of the input tensor in the given dimension dim.
              And indices is the index location of each maximum value found (argmax)
            > a = torch.randn(4, 4)
            > a
            tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
                    [ 1.1949, -1.1127, -2.2379, -0.6702],
                    [ 1.5717, -0.9207,  0.1297, -1.8768],
                    [-0.6172,  1.0036, -0.6060, -0.2432]])
            > torch.max(a, 1)
            torch.return_types.max(values=tensor([0.8475, 1.1949, 1.5717, 1.0036]), indices=tensor([3, 0, 0, 1])  
              
            '''

            correct += pred.eq(target.data).sum()#每一轮下来的结果
            #>>> torch.eq(x,3) 等于tensor([0, 0, 0, 1, 0], dtype=torch.uint8)
        test_loss = test_loss / len(test_dataloader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_dataloader.dataset),
            100. * correct / len(test_dataloader.dataset)))


if __name__ == '__main__':

    test()
    for i in range(10):
        train(i+1)
        print(
            "训练第{}轮的测试结果-----------------------------------------------------------------------------------------".format(
                i + 1))
        test()