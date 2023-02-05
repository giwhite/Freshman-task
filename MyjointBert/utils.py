import os
from transformers import BertConfig,BertTokenizer
from seqeval.metrics import precision_score, recall_score, f1_score
import numpy as np

def getlabels(task,aim):
    file_name = aim + '_labels.txt'

    file_from = os.path.join('./','data',task,file_name)
    fp = open(file_from,'r')
    all_ls = [label.strip() for label in fp]
    return list(all_ls)


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def compute_result(intent_preds,intent_labels,slot_preds,slot_labels):
    result = {}
    assert(len(slot_preds) == len(slot_labels))
    intent_acc = (intent_preds == intent_labels).mean()
    slot_precision = precision_score(slot_labels, slot_preds),
    slot_recall = recall_score(slot_labels, slot_preds),
    slot_f1 = f1_score(slot_labels, slot_preds)
    result['intent_acc'] = intent_acc
    result['slot_precision'] = slot_precision
    result['slot_recall'] = slot_recall
    result['slot_f1'] = slot_f1

    #计算语义和框架的准确率
    intent_result = (intent_preds == intent_labels)
    slot_result = []
    for preds,labels in zip(slot_preds,slot_labels):
        for p,l in zip(preds,labels):
            single_slot_tag = True
            if p != l:
                single_slot_tag = False
                break
        slot_result.append(single_slot_tag)
    slot_result = np.array(slot_result)
    sementic_frame_acc = np.multiply(intent_result,slot_result).mean()
    result['sementic_frame_acc'] = sementic_frame_acc
    return result



           