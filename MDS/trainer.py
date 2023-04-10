import os
import torch
import logging
from model import MyBart
from tqdm import tqdm,trange
from utils import comput_metrics
from dataloder import data_loader
from transformers import BartConfig,AdamW,get_linear_schedule_with_warmup
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
class Trainer(object):
    def __init__(self,args,train_dataset,test_summary = None,test_dataset = None) -> None:
        self.args = args
        self.tb = SummaryWriter(log_dir="tb_file")
        self.train_dataset = train_dataset
        self.test_summary = test_summary
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
        #计算总的batchs
        t_batchs = len(train_dataLoader)//self.args.gradient_accumulation_steps * self.args.epoch_nums#需要看回传的 

        #准备优化器和规划器
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],#n是每一层的name，p是那一层的参数
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_batchs)
        logger.info("*******start training *******")
        epoch_iterator = trange(int(self.args.epoch_nums),desc="epoch iterating")
        for _ in epoch_iterator:
            train_iterator = tqdm(train_dataLoader,desc="iterating")
            for step, batch in enumerate(train_iterator):
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids':batch[0],
                          'attention_mask':batch[1],
                          'decoder_input_ids':batch[2],
                          'decoder_attention_mask':batch[3]}
                output = self.model(**inputs)
                loss = output[1]
                loss.backward()#计算梯度
                

                if (step+1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    self.tb.add_scalar("loss_train_per/batch",loss.item(),global_step=global_step)
                    global_step += 1

                    #进行判断是否需要参数回传
                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        logger.info("saving model *******")
                        self.save_model()
        logger.info("training finished*******")
        

    def evaluate(self):
        #mode : train,eval,test
        self.model.eval()
        self.model.zero_grad()
        sampler = SequentialSampler(self.test_dataset)
        test_dataloader = DataLoader(self.test_dataset,batch_size=self.args.test_batch_size,sampler=sampler)
        logger.info("********start test**********")
        test_bar = tqdm(test_dataloader,desc='test iterating')
        for docs in test_bar:
            
            pass

    def save_model(self):
         #首先判断路径
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_save = self.model.module if hasattr(self.model,'module') else self.model
        #如果又module属性，那就使用里面的属性否者直接使用
        model_save.save_pretrained(self.args.model_dir)#这个可以看源代码，save_pretrained属性会把config和模型参数放入
        logger.info('saving the model in {}'.format(self.args.model_dir))

    def load_model(self):
        if not os.path.exists(self.args.model_dir):
            raise Exception('模型的参数文件不存在,请先训练')
        
        self.model = MyBart.from_pretrained(self.args.model_name_or_path,
                                            config=self.config,
                                            args=self.args)
        self.model.to(self.device)
