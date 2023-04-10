
import logging
import evaluate
from transformers import BartConfig,BartTokenizer

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

    return all_rouges

if __name__ == "__main__":
    generated_summary = ["The US has reported more than 200,000 new Covid-19 cases and 100,000 hospitalizations."]
    
    reference_summaries =[["The US has hit a new record for Covid-19 cases and hospitalizations."]]*4
    comput_metrics(generated_summary,reference_summaries)
