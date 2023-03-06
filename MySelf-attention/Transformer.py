import math
import torch.nn as nn
from Trans_toolkits import EncoderBlock,DecoderBlock,PositionalEncoding,EncoderDecoder
from d2l import torch as d2l
class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
    
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
    

if __name__ == "__main__":
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1,64, 10#这个num_steps就是键值对的个数
    lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
    ffn_num_input, ffn_num_hiddens, num_heads = 32,64,4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]
    train_iter,src_vocab,tgt_vocab = d2l.load_data_nmt(batch_size=batch_size,num_steps=num_steps)
    encoder = TransformerEncoder(
        len(src_vocab), key_size,query_size, value_size,num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens,num_heads,
        num_layers, dropout
    )
    decoder = TransformerDecoder(
        len(tgt_vocab), key_size, query_size,value_size,num_hiddens,
        norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,
        num_layers,dropout
    )
    net = EncoderDecoder(encoder,decoder)
    d2l.train_seq2seq(net,train_iter,lr,num_epochs,tgt_vocab,device)