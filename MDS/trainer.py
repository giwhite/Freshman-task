import torch

from model import MyBart
from tqdm import tqdm,trange
from dataloder import data_loader
from transformers import BartConfig
from torch.utils.data import DataLoader,RandomSampler
class Trainer(object):
    def __init__(self,args,train_dataset,eval_dataset = None,test_dataset = None) -> None:
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.config = BartConfig.from_pretrained(args.model_name_or_path,finetuning_task=args.task)
        self.model = MyBart.from_pretrained(args.model_name_or_path,
                                            config=self.config,
                                            args=args)
        self.device = self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

        pass
    def train(self):
        self.model.train()
        self.model.zero_grad()
        train_sampler = RandomSampler(self.train_dataset)
        train_dataLoader = DataLoader(dataset=self.train_dataset,
                                      sampler=train_sampler,
                                      batch_size=self.args.train_batch_size)
        global_step = 0
        loss = 0
        epoch_iterator = trange(int(self.args.epoch_nums),desc="epoch iterating")
        for _ in epoch_iterator:
            train_iterator = tqdm(train_dataLoader,desc="iterating")
            for i, batch in enumerate(train_dataLoader):
                inputs = {'input_ids':batch[0],
                          'attention_mask':batch[1],
                          'decoder_input_ids':batch[2],
                          'decoder_attention_mask':batch[3]}
                self.model(**inputs)


    def evaluate(self,mode):
        #mode : train,eval,test
        pass