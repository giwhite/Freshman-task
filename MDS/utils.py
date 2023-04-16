
import logging
import torch
from rouge import Rouge



from transformers import BartConfig,BartTokenizer
from model import LSTMEncoder

def get_tokenizer():
    return BartTokenizer.from_pretrained("bart-large")



def init_logger():
    logging.basicConfig(filename= 'log.txt',
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    

def comput_metrics(generated_summaries, reference_summaries, mode = 0):
    '''计算rouge得分

    因为句子有的时候并不会很数量上是对应的，所以这某些时候需要进行复制使之对应
    
    Args:
        generated_summaries: 一般是产生的句子
        reference_summaries: 一般是标准摘要的句子
        mode: 
            0: 不进行复制
            1: 将generated_summaries复制到和reference_summaries的长度
            2: 就是将refernece_summaries 复制到generated_summaries的长度
            
    Returns:
        all_rouges: 句子之间的所有类型的rouge得分
        rougeL: 返回句子的rougeL
    '''
    rouge = Rouge()

    if mode == 1:
        generated_summaries = [generated_summaries]*len(reference_summaries)
    elif mode == 2:
        reference_summaries = [reference_summaries]*len(generated_summaries)
    all_rouges = rouge.get_scores(generated_summaries,reference_summaries)
    rougeL = [score['rouge-l']['f'] for score in all_rouges]

    return all_rouges, rougeL  #这个返回的每个句子和标准再要的rouge
 
class MyMMR(object):
    def __init__(self,args) -> None:
        self.sim_function = torch.nn.CosineSimilarity(dim=1,eps=1e-6)
        self.encoder =LSTMEncoder(args.vocab_size,hidden_dim=args.encode_hidden_dim,embedding_dim=args.embedding_dim)
    
    def get_vectors(self,all_sentences_ids, sentences_len, artcile_ids):

        outputs = self.encoder(all_sentences_ids,sentences_len)
        shape = outputs.shape

        sent_vecs = []
        arti_vecs = []
        for i in range(shape[0]):
            vector = outputs[i,sentences_len[i]-1,:]
            sent_vecs.append(vector.view(1,-1))
        sent_vecs = torch.concat(sent_vecs,dim=0)
        for i in range(len(artcile_ids)-1):
            doc_vec = torch.mean(sent_vecs[artcile_ids[i]:artcile_ids[i+1],:],dim=0)
            
            arti_vecs.append(doc_vec.view(1,-1))
        arti_vecs = torch.concat(arti_vecs,dim=0)
        return sent_vecs,arti_vecs

    def rank_select(self,sentenc_vectors,
                    doc_vectors,
                    sentence_text,
                    article_ids,
                    summary_list,
                    lambda_value = 0.7,
                    K = 2):
        """ 选出前k个句子

        Args:
            article_ids:辅助选择文章向量,类似[1,1,1,1,2,3,4,4,5]
        
        """

        sentences_scores= []#计算了分数的sent列表
        
        for i,sent_txt in enumerate(sentence_text):
            sent_vec = sentenc_vectors[i]
            sim1 = self.sim_function(torch.reshape(sent_vec,(1,-1)),
                                     torch.reshape(doc_vectors[article_ids[i]],(1,-1)))
            
            if len(summary_list) == 0:
                #如果是0，那就是最开始的，此时就此时的句子和其他的都做相似度计算。
                other_txt = [sents for j, sents in enumerate(sentence_text) if j != i]  #生成其他的，然后排除自己
                _, all_rougeL = comput_metrics(generated_summaries=other_txt,
                                                   reference_summaries=sent_txt,mode=2)
                sim2 = torch.max(torch.tensor(all_rougeL))
            else:
                _, all_rougeL = comput_metrics(generated_summaries=summary_list,
                                                reference_summaries=sent_txt,mode=2)
                sim2 = torch.max(torch.tensor(all_rougeL))
                
            score = lambda_value* sim1 - (1-lambda_value)*sim2
            sentences_scores.append(score.item())
        ids = torch.argsort(torch.tensor(sentences_scores))[-2:]
        return ids  #然后外部就需要添加这些东西了

        
        
                


                