import torch
import torch.nn as nn
from transformers import BartPretrainedModel,BartForConditionalGeneration



class Dotattention(nn.Module):
    """ 论文中的attention 计算方式

    Attibutes:
        W_t : decoder时间步的连接层
        W_i : encoder时间步的连接层
        W_v : 输出全连接层
    """
    def __init__(self, args, hidden_size) -> None:
        super().__init__()

        self.W_t = nn.Linear(hidden_size, hidden_size)
        self.W_i = nn.Linear(args.input_max_len, args.decoder_max_len, bias=True)
        self.W_v = nn.Linear(hidden_size, args.input_max_len)


    def forward(self, encoder_hidden_state, decoder_hidden_state):

        encoder_matrix = self.W_i(encoder_hidden_state.permute(0,2,1))#batch_size,hidden_size,t
        decoder_matrix = self.W_t(decoder_hidden_state)#batch_size,t,hidden_size
        encoder_matrix = encoder_matrix.permute(0,2,1)#batch_size,t, hidden_size
        logits = nn.functional.tanh(encoder_matrix+decoder_matrix)
        return nn.functional.softmax(self.W_v(logits),dim=-1)


class Possibility_vcb(nn.Module):
    def __init__(self, dropout_rate, hidden_dim, bias, hidden_size, vocab_size) -> None:
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_rate)
        self.W_y = nn.Linear(hidden_size*2, hidden_dim, bias=bias)
        self.fn = nn.Linear(hidden_dim, vocab_size)


    def forward(self,
                attention_weight,
                encoder_hidden_state,
                decoder_hidden_state):
        '''

        Args:
            attention_weight: batch_size, target_lenght(t), seq_len(i)
            encoder_hidden_state: batch_size, seq_lenght(i), hidden_dim
            decoder_hidden_state: batch_size ,seq_length(t),hidden_dim

        Returns:
            logits: 计算的概率
        '''
       
        context_matrix = torch.bmm(attention_weight,encoder_hidden_state)# batch_size, t,hidden_dim
        concat = torch.cat([context_matrix,decoder_hidden_state],dim=2)# batch_size, t, hidden_dim*2
        
        output = self.W_y(concat)
        output = self.dropout(output)
        logits = self.fn(output)
        return logits  #得到不同t个state的每个的词的概率，这里不需要进行计算softmax
    
class LSTMEncoder(nn.Module):
    def __init__(self,vocab_size, hidden_dim, embedding_dim) -> None:

        super().__init__()
        self.encoder = nn.LSTM(embedding_dim,hidden_dim,bidirectional=True)
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
    
    def forward(self, all_sentences, sentences_lens):
        #此时传入的all_sentences_ids是已经填充过了的
        all_sentences = self.embedding(all_sentences)
        lens,indices = sentences_lens.sort(descending = True)
        all_sentences = all_sentences[indices]

        packed_sent = nn.utils.rnn.pack_padded_sequence(all_sentences,lens,batch_first=True)
        outputs,(hidden,cell) = self.encoder(packed_sent)
        outputs,_ = nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)

        _,indices = indices.sort()

        outputs = outputs[indices]  #这个地方需要调整不同的序列的顺序，需要把这些序列的顺序该返还回去

        hidden = hidden[:,indices]

        return outputs  #将第一维的1压缩回去，得到每个句子序列中的词的hidden_state，也就是我们的词向量



class MyBart(BartPretrainedModel):
    def __init__(self, config, args) -> None:
        super(MyBart,self).__init__(config)

        self.args = args
        self.config = config
        self.bart = BartForConditionalGeneration(config=config)
        self.Pw = Possibility_vcb(args.dropout_rate,
                                  args.hidden_dim,
                                  True,
                                  config.d_model,
                                  config.vocab_size)
        self.attention = Dotattention(args,config.d_model)
        

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        output = self.bart(input_ids=input_ids,
                           attention_mask=attention_mask,
                           decoder_input_ids=decoder_input_ids,
                           decoder_attention_mask=decoder_attention_mask,
                           output_hidden_states=True)
        #直接调用decoder_hidden_state,用字典方式
        decoder_hidden_state = output['decoder_hidden_states'][6]#取出最后一个8*56*1024
        encoder_hidden_state = output['encoder_last_hidden_state']#8*780*1024

        attention_weight = self.attention(encoder_hidden_state=encoder_hidden_state,
                                          decoder_hidden_state=decoder_hidden_state)
        
        logits = self.Pw(attention_weight=attention_weight,
                         encoder_hidden_state=encoder_hidden_state,
                         decoder_hidden_state=decoder_hidden_state)
        
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.permute(0,2,1), decoder_input_ids)
        # pending need to finish
        return (logits,loss)
    

    def text_generate(self, input_ids, attention_mask, tokenizer):  #batchsize 为1

        decoder_input_ids = torch.tensor([[tokenizer.bos_token_id]])

        while True:
            output = self.bart(input_ids=input_ids,
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids)
            next_token_logits = output.logits[:,-1,:]
            next_token_id = torch.argmax(next_token_logits,dim = -1).unsqueeze(0)
            decoder_input_ids = torch.cat([decoder_input_ids,next_token_id],dim=-1)  #在行上面进行操作

            if next_token_id == tokenizer.eos_token_id or decoder_input_ids.size(-1) == self.args.decoder_max_len:
                break

        summary = tokenizer.decode(decoder_input_ids[0],skip_special_tokens=True)  #这里放0是因为之前的decoder_input_ids放的是一个二维数组
        return summary
