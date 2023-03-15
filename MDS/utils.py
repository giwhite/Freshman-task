
import logging

from transformers import BartConfig,BartTokenizer

def get_tokenizer():
    return BartTokenizer.from_pretrained("bart-base")

def get_config():
    return BartConfig.from_pretrained("bart-base")

def init_logger():
    logging.basicConfig(filename= 'log.txt',
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)