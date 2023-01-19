import os

def data_process(task):
    file_from_intent ='./data/'+task+'/train/label'
    file_from_slot = os.path.join('./','data',task,'train','seq.out')
    file_to_intent = os.path.join('./','data',task,'intent_labels.txt')
    file_to_slot = os.path.join('./','data',task,'slot_labels.txt')
    intent_labels = set()
    slot_labels = set()
    special_keys = ['UNK','PAD']
    #lables for intent
    with open(file_from_intent,'r') as fp:
        for label in fp:
            intent_labels.add(label.strip())
        
    with open(file_from_slot,'r') as fp:
        for words in fp:
            for word in words.strip().split():
                slot_labels.add(word)
    intent_labels = sorted(list(intent_labels), key=None, reverse=False) 
    slot_labels = sorted(list(slot_labels),key=lambda x: x[2:],reverse=False)
    intent_labels.insert(0,'UNK')
    for key in special_keys:
        slot_labels.insert(0,key)
    with open(file_to_intent,'w') as fp:
        for label in list(intent_labels):
            fp.write(label+'\n')
    with open(file_to_slot,'w') as fp:
        for label in list(slot_labels):
            fp.write(label+'\n')
if __name__ == "__main__":
    data_process('atis')
    data_process('snips')
    print("over!")
   
