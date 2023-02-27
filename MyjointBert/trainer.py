from torch.utils.data import DataLoader, RandomSampler,SequentialSampler
from torch import nn
import torch
from model import JointBert
from transformers import BertConfig,AdamW,get_linear_schedule_with_warmup
from data_loador import *
from tqdm import tqdm,trange
from torchcrf import CRF
from utils import getlabels,compute_result
import numpy as np

class trainer(object):
    def __init__(self,args) -> None:
        self.args = args
        self.config = BertConfig.from_pretrained(args.model_name_or_path, finetuning_task=args.task)

        self.intent_labels = getlabels(args.task,'intent')
        self.slot_labels = getlabels(args.task,'slot')
        self.pad_token_label_id = args.ignore_index
        self.num_intent_labels = len(self.intent_labels)
        self.num_slot_labels = len(self.slot_labels)

        self.model = JointBert.from_pretrained(args.model_name_or_path,
                                        config=self.config,
                                        args=args,
                                        intent_labels_list=self.intent_labels,
                                        slot_labels_list=self.slot_labels)
        
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        if self.args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True).to(self.device)

    def loss_compute(self,intent_logits,slot_logits,intent_ids,slot_ids,attention_mask):
        total_loss = 0
        #把loss汇总，两个任务的loss
        #这里是intent的loss
        if self.num_intent_labels == 1:
            intent_loss_fn = nn.MSELoss().to(self.device)
            total_loss += intent_loss_fn(intent_logits.view(-1),intent_ids.view(-1))
        else:
            intent_loss_fn = nn.CrossEntropyLoss().to(self.device)
            #这里是intent_loss
            total_loss += intent_loss_fn(intent_logits.view(-1,self.num_intent_labels),intent_ids.view(-1))
        if slot_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss
            else:
                slot_loss_fn = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index).to(self.device)

                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1#active_loss.shape = [1600]
                    #slot_logits.view(-1, self.num_slot_labels).shape = [1600,122]
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]#先将每个batch中的展开然后，进行对应，取出shape is [419,122]
                    active_labels = slot_ids.view(-1)[active_loss]#shape is [419,]
                    slot_loss = slot_loss_fn(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fn(slot_logits.view(-1,self.num_slot_labels),slot_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss
        return total_loss

    def train(self):
        self.model.train()
        train_dataset = load_and_save_data(self.args,'train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(dataset=train_dataset,sampler = train_sampler,batch_size=self.args.train_batch_size)
        #计算总的batch数量
        # if self.args.max_steps > 0:
        #     #如果给了最大的步长，那就通过这个步长来算需要训练多少个epoch
        #     t_batchs = self.args.max_steps
        #     self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        # else :
        #这里可以不用上面那个东西，因为我们epoch的数量是给定的
        t_batchs = len(train_dataloader)//self.args.gradient_accumulation_steps * self.args.num_train_epochs#需要看回传的

        self.model.zero_grad()#清一次梯度
        gb_step = 0

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],#n是每一层的name，p是那一层的参数
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        #https://blog.csdn.net/orangerfun/article/details/120400247 函数使用
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_batchs)
        print('————————————开始训练——————————————')
        epoch_iterator = trange(int(self.args.num_train_epochs),desc='epoch iterating')
        for _ in epoch_iterator:
            pbar = tqdm(train_dataloader,desc="Iteration")
            for step,batch in enumerate(pbar):
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {
                    'input_ids':batch[0],
                    'attention_mask': batch[1],
                    'token_type_id':batch[2]
                }
                intent_logit,slot_logit = self.model(**inputs)
                loss = self.loss_compute(intent_logits=intent_logit,
                                        slot_logits=slot_logit,
                                        intent_ids=batch[3],
                                        slot_ids=batch[4],
                                        attention_mask = batch[1])
                loss.backward()
                #上面进行完了loss的计算，放到了每一层的梯度上了
                #后面就要进行梯度裁剪和梯度回传更新参数
                if (step+1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    gb_step += 1

                    #进行判断是否需要参数回传
                    if self.args.logging_steps > 0 and gb_step % self.args.logging_steps == 0:
                        self.evaluate('dev')
                        eval_step = gb_step/self.args.logging_steps
                        #这个eval_step是用来可视化的global_step
                    if self.args.save_steps > 0 and gb_step % self.args.save_steps == 0:
                        self.save_model()
        print('————————————训练结束—————————————')
        return gb_step 
        
    def evaluate(self,mode):
        '''
        mode： dev,test
        '''
        #load dataset 
        if mode == 'dev':
            dataset = load_and_save_data(self.args,'dev')
        elif mode == 'test':
            dataset = load_and_save_data(self.args,'test')
        else:
            raise Exception("Only dev and test dataset available")

        sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset=dataset,sampler=sampler,batch_size=self.args.eval_batch_size)
        self.model.eval()
        loss = 0.0
        batch_step = 0
        #这里是模型的预测结果
        slot_preds = None
        intent_preds = None
        #下面是参考答案
        slot_labels_ids = None
        intent_labels_ids = None

        eval_bar = tqdm(eval_dataloader,desc='evaling')
        for batch in eval_bar:
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    'input_ids':batch[0],
                    'attention_mask': batch[1],
                    'token_type_id':batch[2]
                }
                intent_logits,slot_logits = self.model(**inputs)
                eval_loss = self.loss_compute(intent_logits=intent_logits,
                                        slot_logits=slot_logits,
                                        intent_ids=batch[3],
                                        slot_ids=batch[4],
                                        attention_mask = batch[1])
                loss += eval_loss.mean().item()
            batch_step += 1
            #计算的结果已经出来了
            #预测结果intent
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                intent_labels_ids = batch[3].detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds,intent_logits.detach().cpu().numpy(),axis= 0)
                intent_labels_ids = np.append(intent_labels_ids,batch[3].detach().cpu().numpy(),axis=0)
            #预测结果slot
            if slot_preds is None:
                if self.args.use_crf:
                    slot_preds = np.array(self.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()
                slot_labels_ids = batch[4].detach().cpu().numpy()
            else :
                if self.args.use_crf:
                    slot_preds = np.append(slot_preds,self.crf.decode(slot_logits))
                else:
                    slot_preds = np.append(slot_preds,slot_logits.detach().cpu().numpy(),axis= 0)
                slot_labels_ids = np.append(slot_labels_ids,batch[4].detach().cpu().numpy(),axis= 0)

        #计算结果
        intent_results = np.argmax(intent_preds,axis= 1)
        #slot 计算结果 
        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds,axis= 2)
        slot_map = {i: label for i, label in  enumerate(self.slot_labels)}
        filtered_slot_labels = [[] for _ in range(slot_labels_ids.shape[0])]
        filtered_slot_preds = [[] for _ in range(slot_labels_ids.shape[0])]

        for i in range(slot_labels_ids.shape[0]):
            for j in range(slot_labels_ids.shape[1]):
                if slot_labels_ids[i][j] != self.pad_token_label_id:
                    filtered_slot_labels[i].append(slot_map[slot_labels_ids[i][j]])
                    filtered_slot_preds[i].append(slot_map[slot_preds[i][j]])
        
        #这里计算出来的 intent_results,intent_ids,filtered_slot_labels,filtered_slot_preds，放到一个函数中
        results = compute_result(intent_results,
                                intent_labels=intent_labels_ids,
                                slot_preds=filtered_slot_preds,
                                slot_labels=filtered_slot_labels)
        print('————————————预测结果——————————————')
        for key in sorted(results.keys()):
            print('{} = {}'.format(key,str(results[key])))
        return results




    def save_model(self):
        #首先判断路径
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_save = self.model.module if hasattr(self.model,'module') else self.model
        #如果又module属性，那就使用里面的属性否者直接使用
        model_save.save_pretrained(self.args.model_dir)#这个可以看源代码，save_pretrained属性会把config和模型参数放入
        print('saving the model in {}'.format(self.args.model_dir))

    
    def load_model(self):
        if os.path.exists(self.args.model_dir):
            raise Exception('{} do not exist, please train first'.format(self.args.model_dir))
        
        try:
            self.model = JointBert.from_pretrained(self.args.model_dir,
                                        args=self.args,
                                        intent_labels_list=self.intent_labels,
                                        slot_labels_list=self.slot_labels)
            self.model.to(self.device)
            print("————————————model loaded————————————")
        except:
            raise Exception("Some model files might be missing...")