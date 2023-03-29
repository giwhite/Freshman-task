import argparse
import logging
from trainer import Trainer
from dataloder import data_loader
from dataloder import cache_and_load
from utils import init_logger,get_tokenizer




def main(args):
    init_logger()
    tokenizer = get_tokenizer()
  
    train_dataset = cache_and_load(args,tokenizer,'train')
    #tdataset = cache_and_load(args,tokenizer,'test')
   # test_dataset = tdataset[0]
    #test_summary = tdataset[1]
    my_trainer = Trainer(args,train_dataset)
    
    my_trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='dailymail', required=False, type=str, help="the dataset to load and use")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--model_dir", default='./checkpoints', required=False, type=str, help="Path to save, load model")

    parser.add_argument("--root_dir", default="../data/cnn_dailymail/", type=str, help="raw data file dir")
    parser.add_argument("--input_max_len", default=780, type=int, help="the max input len of a doc")
    parser.add_argument("--decoder_max_len", default=56, type=int, help="the max input len of a doc in decoder")
    parser.add_argument("--dropout_rate", default=0.1, type=int, help="dropout rate ")
    parser.add_argument("--hidden_dim", default=4096, type=int, help="hidden_size of the possiblity_vcb")

    parser.add_argument("--train_batch_size", default=8, type=int, help="training batch size")
    parser.add_argument("--test_batch_size", default=16, type=int, help="batch size for test")
    parser.add_argument("--epoch_nums", default=10, type=int, help="epoch numbers")
    parser.add_argument("--learning_rate", default=1e4, type=int, help="learning rate of the training prograss")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")

    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")

    args= parser.parse_args()
    args.model_name_or_path = './bart-large'
    main(args)