import logging
import os
import pickle
import torch
from tqdm import tqdm

from datasets import load_dataset
from torch.utils.data import TensorDataset
from utils import MyMMR


logger = logging.getLogger(__name__)




class data_loader(object):
    """ 用来加载数据和处理数据的类

    Attributes:
        args: 从main中传入的需要使用的参数
        mmr: MMR模块，用在处理test阶段数据集的时候
    """
    def __init__(self, args) -> None:
        self.args = args
        self.mmr = MyMMR(args)
        
    def load_test_dataset(self, raw_data: dict, tokenizer):
        ''' 处理和准备测试的时候的数据集

        主要功能是通过mmr的get_vector得到句子和文章向量。然后通过rank_select计算得分并
        给句子排序，选出前k个，选出来的句子进行拼接和padding，最后输出为tensordataset

        Args:
            raw_data: 通过datasets库中的load_dataset导入的原始数据
            tokenizer: 需要用来分词的分词器

        Returns:
            all_input_ids: 将MMR中选中的句子拼接之后在添加特俗处理字符之后, 转换并padding之后
            all_attention_mask: 有效的部分        
        '''
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        #pad_id = 0
        # all_article_ids = []
        # all_len_article = []
        # all_sentences_text = []
        # all_sentences_ids = []
        selected_ids = []
        selected_sents = []
        topic_summaries = []
        bar_data = tqdm(raw_data,desc='processing the test data')
        for topic in bar_data:
            len_arti = [0]#用来放每个文章的最开始的句子在总的list中的位置
            article_ids = []
            arti_raw_sentences = []
            sentences_ids = []
            sent_lens = []
            for id, artic in enumerate(topic['context']):
                raw_sentences = artic.split('\n')[1:-1]#不需要转化成ID
                sentence_ids = tokenizer(raw_sentences, 
                                        padding=True,
                                        max_length=self.args.decoder_max_len,
                                        add_special_tokens=False)['input_ids']#如果使用LSTM那就有要考虑padding的问题
                article_ids = article_ids + [id]*len(raw_sentences)
                len_arti.append(len_arti[id]+len(raw_sentences)+1)
                sent_lens.append(len(raw_sentences))
                arti_raw_sentences.extend(raw_sentences)
                sentences_ids.extend(sentence_ids)
                #topic_arti.append(sentence_ids)
                # if sentences_ids == None:#需要把这个topic的所有句子都合并到一个中
                    
                #     sentences_ids = sentence_ids
                # else:
                   
                #     sentences_ids += sentence_ids
            #这里直接在这个for循环下面进行处理就可以了
            sent_vecs,doc_vecs = self.mmr.get_vectors(all_sentences_ids=sentences_ids,
                                    sentences_len=sent_lens,
                                    artcile_ids=article_ids)
            for i in range(16):
                ids = self.mmr.rank_select(sentenc_vectors=sent_vecs,
                                           doc_vectors=doc_vecs,
                                           sentence_text=arti_raw_sentences,
                                           article_ids=article_ids,
                                           summary_list=selected_sents)
                ids = [int(i) for i in ids]
                selected_ids.extend(ids)
                selected_sents.extend([arti_raw_sentences[i] for i in ids])
            # all_sentences_ids.extend(sentences_ids)
            # all_sentences_text.extend(arti_raw_sentences)
            # all_article_ids.append(article_ids)
            # all_len_article.append(len_arti)
            topic_summaries.append(topic['summary'][0])
            #text_pieces.append(sentence_ids)#一篇文章的ids，用来计算document向量
            #一个topic的ids，用来计算all_textMMR得分，选出句子
            #出现问题，需要填充
        
        #return (TensorDataset(all_text,text_pieces),topic_summaries)#这个sum放出去就是用来计算的

    
    def load_create_dateset(self,
                        tokenizer,
                        pad_token_id = 0,
                        mask_padding_with_zero = True,
                        pad_segment_id = 0
                        ):

        all_input_ids = []
        all_attention_mask = []
        all_decoder_input_ids = []
        all_decoder_attention_ids = []
        
        filename = os.path.join(self.args.root_dir,self.args.task,'stories')
        for file in tqdm(os.listdir(filename)):  #第一篇文章
            name = os.path.join(filename,file)

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
        
            #tokens,存在问题
            article_ids_masks = tokenizer(article)
            highlight_ids_masks = tokenizer(highlight)
            article_ids,attention_mask = article_ids_masks['input_ids'],article_ids_masks['attention_mask']
        
            highlight_ids,decoder_attention_mask = highlight_ids_masks['input_ids'],highlight_ids_masks['attention_mask']


            #截断
            if len(article_ids) > self.args.input_max_len:
                article_ids = article_ids[:self.args.input_max_len]
                attention_mask = attention_mask[:self.args.input_max_len]

            if len(highlight_ids) > self.args.decoder_max_len:
                highlight_ids = highlight_ids[:self.args.decoder_max_len]
                decoder_attention_mask = decoder_attention_mask[:self.args.decoder_max_len]
            

            #padding
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
    

def cache_and_load(args, tokenizer, mode: str):
    """ 保存和加载模型

    Args:
        tokenizer: 分词器，传入到内部函数中去
        mode: 字符串，train or test

    return:
        dataset: use for train or test
    
    """
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
        pickle.dump(dataset, open(file_to, "wb"))  #Save dataset

    return dataset
