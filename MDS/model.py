import torch
import torch.nn as nn
from transformers import BartPretrainedModel,BartForConditionalGeneration



'''

encoder_hidden_states (tuple(torch.FloatTensor),
 optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True)
   — Tuple of torch.FloatTensor (one for the output of the embeddings, 
   if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size)

decoder_hidden_states:batch_size, sequence_length, hidden_size

decoder_attentions: batch_size, num_heads, sequence_length, sequence_length
'''


class Dotattention(nn.Module):
    def __init__(self,args,hidden_size) -> None:
        super().__init__()
        self.W_t = nn.Linear(hidden_size,hidden_size)
        self.W_i = nn.Linear(args.input_max_len,args.decoder_max_len,bias=True)
        self.W_v = nn.Linear(hidden_size,args.input_max_len)
    def forward(self,encoder_hidden_state,decoder_hidden_state):
        encoder_matrix = self.W_i(encoder_hidden_state.permute(0,2,1))#batch_size,hidden_size,t
        decoder_matrix = self.W_t(decoder_hidden_state)#batch_size,t,hidden_size
        encoder_matrix = encoder_matrix.permute(0,2,1)#batch_size,t, hidden_size
        logits = nn.functional.tanh(encoder_matrix+decoder_matrix)
        return nn.functional.softmax(self.W_v(logits),dim=-1)


class Possibility_vcb(nn.Module):
    def __init__(self,dropout_rate,hidden_dim,bias,hidden_size,vocab_size) -> None:
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_rate)
        self.W_y = nn.Linear(hidden_size*2,hidden_dim,bias=bias)
        self.fn = nn.Linear(hidden_dim,vocab_size)

    def forward(self,
                attention_weight,
                encoder_hidden_state,
                decoder_hidden_state):
        '''
        attention_weight: batch_size, target_lenght(t), seq_len(i)

        encoder_hidden_state: batch_size, seq_lenght(i), hidden_dim

        decoder_hidden_state: batch_size ,seq_length(t),hidden_dim
        '''
       
        context_matrix = torch.bmm(attention_weight,encoder_hidden_state)# batch_size, t,hidden_dim
        concat = torch.cat([context_matrix,decoder_hidden_state],dim=2)# batch_size, t, hidden_dim*2
        #addcat = context_matrix + encoder_hidden_state 如果是使用加和方式的话

        output = self.W_y(concat)
        output = self.dropout(output)
        logits = self.fn(output)
        return logits#得到不同t个state的每个的词的概率，这里不需要进行计算softmax
    
class LSTMEncoder(nn.Module):
    def __init__(self,vocab_size,hidden_dim,embedding_dim) -> None:
        super().__init__()
        self.encoder = nn.LSTM(embedding_dim,hidden_dim)
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
    
    def forward(self,all_sentences,sentences_lens):
        #此时传入的all_sentences_ids是已经填充过了的
        all_sentences = self.embedding(all_sentences)
        lens,indices = sentences_lens.sort(descending = True)
        '''
        这一步的意思是根据indices对input进行排序，indices是一个整数张量，它表示了原始input中每个元素的新位置。
        例如，如果input是[[1, 2], [3, 4], [5, 6]]，indices是[2, 0, 1]，那么input[indices]就是[[5, 6], [1, 2], [3, 4]]。
        这样做的目的是为了让不同长度的序列按照长度降序排列，这样才能使用打包序列的方法。
        '''
        all_sentences = all_sentences[indices]
        packed_sent = nn.utils.rnn.pack_padded_sequence(all_sentences,lens,batch_first=True)
        outputs,(hidden,cell) = self.encoder(packed_sent)
        outputs,_ = nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)
        _,indices = indices.sort()
        '''
        这样做的目的是为了让output和hidden能够按照原始顺序还原，这样才能和输入序列对应起来。
        '''
        outputs = outputs[indices]#这个地方需要调整不同的序列的顺序，需要把这些序列的顺序该返还回去
        '''
        不是的，填充之后的隐藏状态的时间步是和输入序列的长度一样的，只是那些无效时间步的隐藏状态是没有被更新过的，也就是说，它们是上一个有效时间步的隐藏状态。
        你可以通过打印填充后和打包后的隐藏状态来对比一下，你会发现，打包后的隐藏状态只包含最后一个有效时间步的隐藏状态，
        而填充后的隐藏状态包含所有时间步的隐藏状态，但是那些无效时间步的隐藏状态和上一个有效时间步的隐藏状态是一样的。那只需要取出最后一个时间步即可
        '''
        hidden = hidden[:,indices]
        '''
        这一步的意思是根据indices对hidden的第二个维度进行排序，hidden是一个三维张量，它表示了LSTM层的最后一个隐藏状态，
        它的形状是[num_layers * num_directions, batch_size, hidden_size]。这样做的目的是为了让hidden和input保持一致的顺序，这样才能正确地生成输出序列。
        已收到消息. 这一步的意思是根据indices对hidden的第二个维度进行排序，hidden是一个三维张量，
        它表示了LSTM层的最后一个隐藏状态，它的形状是[num_layers * num_directions, batch_size, hidden_size]。
        这样做的目的是为了让hidden和input保持一致的顺序，这样才能正确地生成输出序列。
        这里的batchsize其实就是序列的个数
        '''
        return outputs,torch.squeeze(hidden)#将第一维的1压缩回去，得到每个句子序列中的词的hidden_state，也就是我们的词向量


        


class MyBart(BartPretrainedModel):
    def __init__(self,config,args) -> None:
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
        

        
    def forward(self,input_ids,attention_mask,decoder_input_ids,decoder_attention_mask):
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
        loss = loss_fn(logits.permute(0,2,1),decoder_input_ids)
        # pending need to finish
        return (logits,loss)
    
    def text_generate(self,input_ids,attention_mask,tokenizer):#batchsize 为1
        decoder_input_ids = torch.tensor([[tokenizer.bos_token_id]])
        while True:
            output = self.bart(input_ids=input_ids,
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids)
            next_token_logits = output.logits[:,-1,:]
            next_token_id = torch.argmax(next_token_logits,dim = -1).unsqueeze(0)
            decoder_input_ids = torch.cat([decoder_input_ids,next_token_id],dim=-1)#在行上面进行操作
            if next_token_id == tokenizer.eos_token_id or decoder_input_ids.size(-1) == self.args.decoder_max_len:
                break
        summary = tokenizer.decode(decoder_input_ids[0],skip_special_tokens=True)#这里放0是因为之前的decoder_input_ids放的是一个二维数组
        return summary
