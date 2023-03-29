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
        return nn.functional.softmax(logits,dim=-1)#得到不同t个state的每个的词的概率，然后求解即可
    
class MyMMR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,x):
        pass

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
        