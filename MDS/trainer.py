import torch

from model import MyBart

from dataloder import data_loader
from transformers import BartConfig
class Trainer(object):
    def __init__(self,args,train_dataset,eval_dataset,test_dataset) -> None:
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.config = BartConfig.from_pretrained(args.model_name_or_path,finetuning_task=args.task)
        self.model = MyBart.from_pretrained(args.model_name_or_path,
                                            args=self.args,
                                            config=self.config)
        self.device = self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

        pass
    def train(self):
        pass

    def evaluate(self,mode):
        #mode : train,eval,test
        pass