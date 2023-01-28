from transformers import BertModel, BertPreTrainedModel,BertConfig
from torchcrf import CRF
from torch import nn


class JointBert(BertPreTrainedModel):
    def __init__(self, config,args,intent_labels_list,slot_labels_list):
        super(JointBert, self).__init__(config)
        self.args = args
        self.bert = BertModel(config=config)
        self.num_intent_labels = len(intent_labels_list)
        self.num_slot_labels = len(slot_labels_list)
        #因为这里是双向并行，所以使用nn.Sequential()
        self.seq_slot = nn.Sequential(
                            nn.Dropout(args.dropout_rate),
                            nn.Linear(config.hidden_size, self.num_slot_labels)
                        )
        self.seq_intent = nn.Sequential(
                            nn.Dropout(args.dropout_rate),
                            nn.Linear(config.hidden_size, self.num_intent_labels)
                        )
        if self.args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)
    
    def forward(self,input_ids,attention_mask,token_type_id):
        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_id)
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        slot_logits = self.seq_slot(sequence_output)
        intent_logits = self.seq_intent(pooled_output)
        # sequence_output = self.dropout(sequence_output)
        # pooled_output = self.dropout(pooled_output)
        # slot_logits = self.linear(sequence_output)
        # intent_logits = self.linear(pooled_output)
        return intent_logits,slot_logits
        
        