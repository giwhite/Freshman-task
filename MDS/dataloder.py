import os
import torch
import logging
import pickle
from tqdm import tqdm
from datasets import load_dataset
from utils import MyMMR
from torch.utils.data import TensorDataset


logger = logging.getLogger(__name__)


# class InputExample(object):


class data_loader(object):
    def __init__(self,args) -> None:
        self.args = args
        self.mmr = MyMMR(args)
        
    def load_test_dataset(self,raw_data,tokenizer):

        '''
        需要准备每个句子的id，拼接起来，这里就需要考虑需不需要裁剪了，对每个句子
        all_sentences_ids

        需要准备一个topic的文章的拼接ids，这里需要裁剪，但是是用哪一个版本
        直接使用上面的sentences_ids拼接？，然后将将选中的句字部分attention——mask置1
        concat_sentences_ids

        attention_mask 所有拼接文章的需要的掩码，这个最后的效果应该需要和concat_sentences_ids的长度一样

        padding
        
        '''
        
        topic_sum = []
        all_text = []
        text_pieces = []
        bar_data = tqdm(raw_data,desc='processing the test data')
        for topic in bar_data:
            topic_arti = []#用来放每个文章的句子ids，
           
            sentences_ids = None
            for artic in topic['context']:
                sentences = artic.split('\n')[1:-1]#不需要转化成ID
                sentence_ids = tokenizer(sentences)['input_ids']#如果使用LSTM那就有要考虑padding的问题
                topic_arti.append(sentence_ids)
                if sentences_ids == None:#需要把这个topic的所有句子都合并到一个中
                    
                    sentences_ids = sentence_ids
                else:
                   
                    sentences_ids += sentence_ids
            topic_sum.append(topic['summary'][0])
            text_pieces.append(sentence_ids)#一篇文章的ids，用来计算document向量
            all_text.append(sentences_ids)#一个topic的ids，用来计算MMR得分，选出句子
            #出现问题，需要填充
        all_text = torch.tensor(all_text)
        text_pieces = torch.tensor(text_pieces)
        return (TensorDataset(all_text,text_pieces),topic_sum)#这个sum放出去就是用来计算的
        '''
        最后的return的结果应该是，all_sentences_ids, concat_sentences_ids，attention_mask
        然后mask是通过mmr生成的
        '''
    
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
        if mode == 'train':
            dataset = Dloader.load_create_dateset(tokenizer)
        else :
            raw_data = load_dataset('nbtpj/DUC2004')
            dataset = Dloader.load_test_dataset(raw_data['train'],tokenizer)
            #这个返回的数据集['train']['summary']这个是又50个摘要，这个50个中分别是4个
            # ['train']['context']这个是有50篇topic,每个topic有10篇文章

        logger.info('saving dataset in file: {}'.format(file_to))
        pickle.dump(dataset, open(file_to, "wb")) 

    return dataset
