'''
就是要准备bert输入模型的五个参数
其中3个放在Bert模型中，身下的几个放在后面使用
阿里云http://mirrors.aliyun.com/pypi/simple/
input_ids = tokenizer.convert_tokens_to_ids(tokens)
'''
from utils import getlabels,tokenizer
import os
import torch
from torch.utils.data import TensorDataset

class processor(object):
    def __init__(self,args) -> None:
        self.args = args
        self.intent_labels = getlabels(args.task,'intent')
        self.slot_labels = getlabels(args.task,'slot')
        self.input_text_file = 'seq.in'
        self.intent_labels_file = 'label'
        self.slot_labels_file = 'seq.out'

    def data_processing(self,mode):

        '''
        mode: train, eval, test
        '''
        #首先处理labels
        #注意一定要处理数据中的不存在的地方，以为后面进行eval和test的时候可能会有没有发现的数据，在这里的处理的时候需要
        #考虑
        #pad_slot_id = 0#这个放在slot填充
        pad_token_label_id=-100
        cls_token_segment_id=0
        pad_token_segment_id=0
        sequence_a_segment_id=0
        mask_padding_with_zero=True


        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        unk_token = tokenizer.unk_token
        pad_token_id = tokenizer.pad_token_id


        intent_labels_ids = []
        with open(os.path.join('./','data',self.args.task,mode,self.intent_labels_file),'r') as fp:
            for label in fp:
                if label.strip() in self.intent_labels:
                    intent_labels_ids.append(self.intent_labels.index(label.strip()))
                else:
                    intent_labels_ids.append(self.intent_labels.index("UNK"))
                
        slot_labels_ids_raw = []
        with open(os.path.join('./','data',self.args.task,mode,self.slot_labels_file),'r') as fp:
            for words in fp:
                mid_ls  = []
                for word in words.strip().split():
                    if word in self.slot_labels:
                        mid_ls.append(self.slot_labels.index(word))
                    else:
                        mid_ls.append(self.slot_labels.index('UNK'))
                slot_labels_ids_raw.append(mid_ls)
        #处理text，这里text需要处理
        #padding
        #attention_mask
        #type_id
        features = []
        
        with open(os.path.join('./','data',self.args.task,mode,self.input_text_file),'r') as fp:
            for i,texts in enumerate(fp):
                feature = {}
                tokens = []
                slot_labels_ids = []
                slot_labels_ids_raw_sin = slot_labels_ids_raw[i]
                for text,slot_singleword_id in zip(texts.strip().split(),slot_labels_ids_raw_sin):
                    processed_text = tokenizer.tokenize(text)
                    if not processed_text:
                        processed_text = [unk_token]#这里加上了中间括号需方便使用extend
                    tokens.extend(processed_text)
                    slot_labels_ids.extend([int(slot_singleword_id)] + [pad_token_label_id]*(len(processed_text)-1))
                #添加特殊字符
                #对每一个字符串处理
                specitial_tokens_nums = 2
                if(len(tokens)>self.args.max_seq_len - specitial_tokens_nums):
                    tokens = tokens[:(self.args.max_seq_len - specitial_tokens_nums)]
                    slot_labels_ids = slot_labels_ids[:(self.args.max_seq_len - specitial_tokens_nums)]
                #插入特殊字符
                #[SEP]需要放在最后
                tokens += [sep_token]
                slot_labels_ids = slot_labels_ids + [pad_token_label_id]
                token_type_ids = [sequence_a_segment_id] * len(tokens)
                    
                #[CLS]需要放在最前面
                tokens = [cls_token]+tokens
                slot_labels_ids = [pad_token_label_id] + slot_labels_ids
                token_type_ids = [cls_token_segment_id]* len(tokens)
                #将数据转化成可计算的量
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                attention_mask = [1 if mask_padding_with_zero else 0]*len(input_ids)
                
                padding_num = self.args.max_seq_len - len(input_ids)
                input_ids += [pad_token_id]*padding_num
                slot_labels_ids += [pad_token_label_id]* padding_num
                attention_mask += [0 if mask_padding_with_zero else 1]* padding_num
                token_type_ids += [pad_token_segment_id]*padding_num
                
                assert len(input_ids) == self.args.max_seq_len, "Error with input length"
                assert len(slot_labels_ids) == self.args.max_seq_len,'Error with slot_labels_ids length'
                assert len(attention_mask) == self.args.max_seq_len,'Error with attention_mask length'
                assert len(token_type_ids) == self.args.max_seq_len,'Error with token_type_ids'

                feature['input_ids'] = input_ids
                feature['slot_labels_ids'] = slot_labels_ids
                feature['intent_label_ids'] = int(intent_labels_ids[i])
                feature['attention_mask'] = attention_mask
                feature['token_type_ids'] = token_type_ids
                features.append(feature)
        
        return features
    def get_dataset(self,features):
        all_input_ids = torch.tensor([f['input_ids'] for f in features])
        all_slot_labels_ids = torch.tensor([f['slot_labels_ids'] for f in features])
        all_intent_label_ids = torch.tensor([f['intent_label_ids'] for f in features])
        all_attention_mask = torch.tensor([f['attention_mask'] for f in features])
        all_token_type_ids = torch.tensor([f['token_type_ids'] for f in features])
        dataset = TensorDataset(all_input_ids, all_attention_mask,
                        all_token_type_ids, all_intent_label_ids, all_slot_labels_ids)
        return dataset



def load_and_save_data(args,mode):
    #首先创建这个处理器
    data_processor = processor(args)
    #创建文件名字
    
    file_name = "saved_{}_{}_{}_{}".format(
                    args.task,
                    mode,
                    args.max_seq_len,
                    'e'
                )
    save_to_file_name = os.path.join('./','data',args.task,file_name)
    if os.path.exists(save_to_file_name):
        dataset = torch.load(save_to_file_name)
    else:
        rawdata = data_processor.data_processing(mode)
        dataset = data_processor.get_dataset(rawdata)
        torch.save(dataset,save_to_file_name)
    return dataset

        
    


    