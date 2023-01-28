import os
from transformers import BertConfig,BertTokenizer
def getlabels(task,aim):
    file_name = aim + '_labels.txt'

    file_from = os.path.join('./','data',task,file_name)
    fp = open(file_from,'r')
    all_ls = [label.strip() for label in fp]
    return list(all_ls)


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# if __name__ == "__main__":
#     ls = getlabels('atis','intent')
#     print(ls)
           