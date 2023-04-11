
import logging
import evaluate
from transformers import BartConfig,BartTokenizer
from model import LSTMEncoder

def get_tokenizer():
    return BartTokenizer.from_pretrained("bart-large")



def init_logger():
    logging.basicConfig(filename= 'log.txt',
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    

def comput_metrics(generated_summary,reference_summaries):
    rouge = evaluate.load('rouge')
    all_rouges = []
    
    for reference_summary in reference_summaries:
        score = rouge.compute(predictions=generated_summary, references=reference_summary)
        all_rouges.append(score)

    return all_rouges#这个返回的每个句子和标准再要的rouge

class MyMMR(object):
    def __init__(self,args) -> None:
        
        self.encoder =LSTMEncoder(args.vocab_size,hidden_dim=args.decode_hidden_dim,embedding_dim=args.embedding_dim)
    
    def comput(self,all_sentences_ids,sentences_len,artcile_ids):
        '''
        句子id，句子长度，文章中的句子部分
        '''
        _,encoded_sentences_vec = self.encoder(all_sentences_ids,sentences_len)
        #这个出来猜测应该是 b * l * dim
        #l 求平均就是句子向量
        # 然后通过artcile选出来的句子向量作平均就是doc_vec
        