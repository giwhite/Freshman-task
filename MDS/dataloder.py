import os
import torch
import logging
import pickle
from tqdm import tqdm
from torch.utils.data import TensorDataset


logger = logging.getLogger(__name__)


# class InputExample(object):
#     def __init__(self,id,article,highlight) -> None:
#         self.id = id
#         self.article = article
#         self.highlight = highlight


# class InputFeatures(object):
#     def __init__(self,ids,input_ids,attention_mask,decoder_input_ids,decoder_attention_mask) -> None:
#         self.ids = ids
#         self.input_ids = input_ids
#         self.attention_mask = attention_mask
#         self.decoder_input_ids = decoder_input_ids
#         self.decoder_attention_mask = decoder_attention_mask

class data_loader(object):
    def __init__(self,args) -> None:
        self.args = args
        
    
    def load_create_dateset(self,
                        tokenizer,
                        pad_token_id = 0,
                        mask_padding_with_zero = True,
                        pad_segment_id = 0
                        ):

        #ids = []
        all_input_ids = []
        all_attention_mask = []
        all_decoder_input_ids = []
        all_decoder_attention_ids = []
        
        filename = os.path.join(self.args.root_dir,self.args.task,'stories')
        for file in tqdm(os.listdir(filename)):#第一篇文章
            name = os.path.join(filename,file)
            #id = file[:-6]

            article = None
            highlight = None
            flag = 0
            with open(name,'r',encoding='utf-8') as fp:
                for i in fp:

                    if i == '@highlight\n':  
                        flag = 1
                        continue
                    if i == '\n':
                        continue
                    if flag == 0:
                        if article is None:
                            article = i.strip().replace(u'\xa0', ' ')
                        else:
                            article += ' ' + i.strip().replace(u'\xa0', ' ')
                    else:
                        if highlight is None:
                            highlight = i.strip().replace(u'\xa0', ' ')
                        else:
                            highlight +=' ' + i.strip().replace(u'\xa0', ' ')
        
            #tokens
            article_ids_masks = tokenizer(article)
            highlight_ids_masks = tokenizer(highlight)
            article_ids,attention_mask = article_ids_masks['input_ids'],article_ids_masks['attention_mask']
        
            highlight_ids,decoder_attention_mask = highlight_ids_masks['input_ids'],highlight_ids_masks['attention_mask']

            if len(article_ids) > self.args.input_max_len:
                article_ids = article_ids[:self.args.input_max_len]
                attention_mask = attention_mask[:self.args.input_max_len]
        

            if len(highlight_ids) > self.args.decoder_max_len:
                highlight_ids = highlight_ids[:self.args.decoder_max_len]
                decoder_attention_mask = decoder_attention_mask[:self.args.decoder_max_len]
            
            padding_len_en = self.args.input_max_len - len(article_ids)
            padding_len_de = self.args.decoder_max_len - len(highlight_ids)

            input_ids = article_ids + [pad_token_id]*padding_len_en
            attention_mask = attention_mask + [0 if mask_padding_with_zero else 1 ]*padding_len_en
            decoder_input_ids = highlight_ids + [pad_segment_id]* padding_len_de
            decoder_attention_mask = decoder_attention_mask + [0 if mask_padding_with_zero else 1] * padding_len_de

            #判断输入有没有错
            assert len(input_ids) == self.args.input_max_len , "wrong len of the input_ids: {} vs {}".format(len(input_ids),
                                                                                                             self.args.input_max_len)
            assert len(attention_mask) == self.args.input_max_len, "wrong len of the attention_mask: {} vs {}".format(len(attention_mask),
                                                                                                                      self.args.input_max_len)
            assert len(decoder_input_ids) == self.args.decoder_max_len,"wrong len of the decoder_input_ids: {} vs {}".format(len(decoder_input_ids),
                                                                                                                             self.args.decoder_max_len)
            assert len(decoder_attention_mask) == self.args.decoder_max_len ,"wrong len of the decoder_attention_mask: {} vs {}".format(len(decoder_attention_mask),
                                                                                                                                        self.args.input_max_len)

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_decoder_input_ids.append(decoder_input_ids)
            all_decoder_attention_ids.append(decoder_attention_mask)
        #read and process data in one iteration
        return TensorDataset(torch.tensor(all_input_ids),
                             torch.tensor(all_attention_mask),
                             torch.tensor(all_decoder_input_ids),
                             torch.tensor(all_decoder_attention_ids))
    

def cache_and_load(args,tokenizer,mode):
    Dloader = data_loader(args)
    
    data_features_dir = 'cached_{}_{}_features.pkl'.format(args.task,mode)
    file_to = os.path.join(args.data_dir,data_features_dir)

    if os.path.exists(file_to):
        logger.info('looking into {}'.format(file_to))
        dataset = pickle.load(open(file_to, "rb"))
    else:
        
        logger.info('create dataset in file: {}'.format(file_to))
        dataset = Dloader.load_create_dateset(tokenizer)

        logger.info('saving dataset in file: {}'.format(file_to))
        pickle.dump(dataset, open(file_to, "wb")) 

    return dataset
