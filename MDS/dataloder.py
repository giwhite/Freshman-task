import os
import torch
import logging
from tqdm import tqdm
from torch.utils.data import TensorDataset


logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self,id,article,highlight) -> None:
        self.id = id
        self.article = article
        self.highlight = highlight


class InputFeatures(object):
    def __init__(self,ids,input_ids,attention_mask,decoder_input_ids,decoder_attention_mask) -> None:
        self.ids = ids
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.decoder_input_ids = decoder_input_ids
        self.decoder_attention_mask = decoder_attention_mask

class data_loader(object):
    def __init__(self,args) -> None:
        self.args = args
        
    def create_examples(self):
        exampels = []

        filename = os.path.join(self.args.root_dir,self.args.task,'stories')
        for file in tqdm(os.listdir(filename)):#第一篇文章
            name = os.path.join(filename,file)
            ids = file[:-6]

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
            exampels.append(InputExample(ids,article,highlight))
            
        return exampels
        # 这个用来得到tensor的数据



def Examples2Features(args,
                             examples,
                             tokenizer,
                             pad_token_id = 0,
                             mask_padding_with_zero = True,
                             pad_segment_id = 0):
    
    '''
    need to prepare attention_mask,input_ids(article_ids),decoder_inputs(highlight_ids),decoder_mask
    '''
    
    features = []
    article_ids = None
    highlight_ids = None
    for ex_ids, example in enumerate(examples):
        if ex_ids % 5000 == 0:
            logger.info("Writing example %d of %d"%(ex_ids,len(examples)))
        #tokens

        article_ids = tokenizer(example.article)
        highlight_ids = tokenizer(example.highlight)
        if len(article_ids) > args.input_max_len:
            article_ids = article_ids[:args.input_max_len]

        if len(highlight_ids) > args.decoder_max_len:
            highlight_ids = highlight_ids[:args.decoder_max_len]

        #prepare attention_mask
        attention_mask = [1 if mask_padding_with_zero else 0] * len(article_ids)
        decoder_attention_mask = [1 if mask_padding_with_zero else 0] * len(highlight_ids)


        #padding
        padding_len_en = args.input_max_len - len(article_ids)
        padding_len_de = args.decoder_max_len - len(highlight_ids)

        input_ids = article_ids + [pad_token_id]*padding_len_en
        attention_mask = attention_mask + [0 if mask_padding_with_zero else 1 ]*padding_len_en
        decoder_input_ids = highlight_ids + [pad_segment_id]* padding_len_de
        decoder_attention_mask = decoder_attention_mask + [0 if mask_padding_with_zero else 1] * padding_len_de

        #判断输入有没有错
        assert len(input_ids) == args.input_max_len , "wrong len of the input_ids: {} vs {}".format(len(input_ids),args.input_max_len)
        assert len(attention_mask) == args.input_max_len, "wrong len of the attention_mask: {} vs {}".format(len(attention_mask),args.input_max_len)
        assert len(decoder_input_ids) == args.decoder_max_len,"wrong len of the decoder_input_ids: {} vs {}".format(len(decoder_input_ids),args.decoder_max_len)
        assert len(decoder_attention_mask) == args.decoder_max_len ,"wrong len of the decoder_attention_mask: {} vs {}".format(len(decoder_attention_mask),args.input_max_len)

        if ex_ids < 5:
            logger.info('******examples******')
            logger.info('id %s' % example.id)
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("decoder_input_ids: %s" % " ".join([str(x) for x in decoder_input_ids]))
            logger.info("decoder_attention_mask: %s" % " ".join([str(x) for x in decoder_attention_mask]))

        features.append(
            InputFeatures(
                        example.id,
                        input_ids,
                        attention_mask,
                        decoder_input_ids,
                        decoder_attention_mask,
                        ))
    return features


def cache_and_load(args,tokenizer,mode):
    Dloader = data_loader(args)
    
    data_features_dir = 'cached_{}_{}_features'.format(args.task,mode)
    file_to = os.path.join(args.data_dir,data_features_dir)
    features = None
    if os.path.exists(file_to):
        logger.info('looking into {}'.format(file_to))
        features = torch.load(file_to)
    else:
        logger.info('create dataset in file: {}'.format(file_to))
        examples = Dloader.create_examples()
        features = Examples2Features(args,examples,tokenizer)
        logger.info('saving features in file: {}'.format(file_to))
        torch.save(features,file_to)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_decoder_input_ids = torch.tensor([f.decoder_input_ids for f in features], dtype=torch.long)
    all_decoder_attention_mask = torch.tensor([f.decoder_attention_mask for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_decoder_input_ids,
        all_decoder_attention_mask
    )
    
    return dataset
