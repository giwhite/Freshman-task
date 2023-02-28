import torch.nn as nn
import torch
#这个self_attention需不需要embedding

class CONFIG(object):
    def __init__(self,batch_size,max_len,is_embedding = True,embedding_dim = None,samples_num = None) -> None:
        self.batch_size = batch_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.samples_num = samples_num
        self.is_embedding = is_embedding
        
config = CONFIG(32,50,False,100,1000)
#输入的数据集
inputs = {}#数据集都是1000个句子，每个句子最大50个词
inputs['random_inputs'] = torch.randn(32,50,100)#创建一个
inputs['ones_inputs'] = torch.ones(32,50,100)



class My_self_attention(nn.Module):
    def __init__(self,config,headers = 1) -> None:
        super().__init__()
        assert headers >=1 ,"headers can't be less than 1"
        self.config = config
        self.headers = headers
        #三个需要调优的矩阵
        self.qs = []
        self.ks = []
        self.vs = []
        self.q = nn.Parameter(torch.randn(self.config.embedding_dim,self.config.embedding_dim),requires_grad=True)
        self.k = nn.Parameter(torch.randn(self.config.embedding_dim,self.config.embedding_dim),requires_grad=True)
        self.v = nn.Parameter(torch.randn(self.config.embedding_dim,self.config.embedding_dim),requires_grad=True)

        self.w0 = nn.Parameter(torch.randn(self.config.embedding_dim*self.headers,
                                           self.config.embedding_dim),
                                           requires_grad=True)
        for _ in range(headers):#下面的方阵也是需要调优的
            if headers == 1:
                break
            self.qs.append(nn.Parameter(torch.randn(self.config.embedding_dim,self.config.embedding_dim),requires_grad=True))
            self.ks.append(nn.Parameter(torch.randn(self.config.embedding_dim,self.config.embedding_dim),requires_grad=True))
            self.vs.append(nn.Parameter(torch.randn(self.config.embedding_dim,self.config.embedding_dim),requires_grad=True))

        if self.config.is_embedding:
            self.embed_x = nn.Embedding(self.config.samples_num,self.config.embedding_dim,padding_idx=0)
        

    def forward(self,x):#x.shape = [32,50] or 已经在外部进行了embedding .x.shape = [32,50,100]
        if self.config.is_embedding:
            x = self.embed_x(x)
        #此时x一定是[32,50,100]

        Q = torch.matmul(x,self.q)#Q.shape = [32,50,100]
        K = torch.matmul(x,self.k)
        V = torch.matmul(x,self.v)
        outs = []
        if self.headers > 1:
            for i in range(self.headers):
                q = torch.matmul(Q,self.qs[i])
                k = torch.matmul(K,self.ks[i])
                v = torch.matmul(V,self.vs[i])
                a = torch.matmul(q,k.permute(0,2,1))
                
                a1 = nn.functional.softmax(a,dim=1)
                outs.append(torch.bmm(a1,v))

            output = torch.matmul(self.w0,torch.cat(outs,dim=1))

        else:
            A = torch.matmul(Q,K.permute(0,2,1))#A.shape = [32,50,50]
            A_1 = nn.functional.softmax(A,dim=1)
            output = torch.bmm(A_1,V)#output.shape = [32,50,100]

        return output
    
if __name__ == "__main__":
    sat = My_self_attention(config,2)
    out = sat.forward(inputs['ones_inputs'])
    print("over!")