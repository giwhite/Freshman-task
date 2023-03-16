
import torch.nn as nn
from transformers import BartPretrainedModel,BartModel
class MyBart(BartPretrainedModel):
    def __init__(self,args,config) -> None:
        super(MyBart,self).__init__(config)
        self.args = args
        self.bart = BartModel(config)
        
    def forward(self,input_ids,attention_mask,decoder_input_ids,decoder_attention_mask):
        output = self.bart(input_ids=input_ids,
                           attention_mask=attention_mask,
                           decoder_input_ids=decoder_input_ids,
                           decoder_attention_mask=decoder_attention_mask)
        # pending need to finish
        return output
        