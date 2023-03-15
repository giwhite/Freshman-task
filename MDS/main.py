import argparse
import logging
from dataloder import data_loader
from dataloder import cache_and_load
from utils import init_logger,get_tokenizer




def main(args):
    init_logger()
    tokenizer = get_tokenizer()
  
    dataset = cache_and_load(args,tokenizer,'train')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='dailymail', required=False, type=str, help="the dataset to load and use")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--root_dir", default="../data/cnn_dailymail/", type=str, help="raw data file dir")
    parser.add_argument("--input_max_len", default=780, type=int, help="the max input len of a doc")
    parser.add_argument("--decoder_max_len", default=56, type=int, help="the max input len of a doc in decoder")
    args= parser.parse_args()
    args.model_name_or_path = './bart-base'
    main(args)