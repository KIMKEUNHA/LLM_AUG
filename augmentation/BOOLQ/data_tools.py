import json
import json
from datasets import load_dataset
import random
import re
with open("boolq_obqa.json","r") as f:
    data_format=json.load(f)
    
boolq_dataset=load_dataset(data_format["BoolQ"]['name'])
obqa_dataset=load_dataset(data_format["OBQA"]['name'])

boolq_dataset['train']=list(boolq_dataset['train'])
obqa_dataset['train']=list(obqa_dataset['train'])



def save_results(results, filename):
    with open(filename,encoding="utf-8", mode="a") as f:
        for line in results:
            f.write(json.dumps(line)+"\n")

def post_process(responses, dname):
    keys=[]
    if dname=="BoolQ":
        keys=["Question", "Passage", "Answer" ]
    elif dname=="OBQA":
        keys=["Question", "Choices", "Answer"]
    regex= r":\s*(.*?)\s*".join(keys)+":\s*(.*?)$"
    gens = []
    for response in responses:
        # print(response)
        match = re.search(regex, response, re.DOTALL)
        # print(match)
        if match:
            temp={}
            for idx,reg in enumerate(keys):
                temp[reg] =  match.group(idx+1).strip()
            gens.append(temp)
            # print(gens)
    
    return gens

def make_prompt(dname):
    # with open("boolq_obqa.json","r") as f:
    #     data_format=json.load(f)
    # dname=input('Data Name: ')
    if dname=='BoolQ':
        dataset=boolq_dataset
        text=data_format[dname]['front']
        idxs=random.sample(range(1,len(dataset["train"])),5)
        for idx in idxs:
            data=dataset['train'][idx]
            info = {}
            for i in data_format[dname]["keys"]:
                info[i] = data[i]
            prompt = data_format[dname]["prompt"]
            
            text+=prompt.format(**info)+"\n\n"
        import pdb; pdb.set_trace()
        for del_idx in sorted(idxs, reverse=True):
            dataset['train'].pop(del_idx)
            
        text += data_format[dname]["end"]
        # print(text)

    elif dname=='OBQA':
        dataset=obqa_dataset
        text=data_format[dname]['front']
        idxs=random.sample(range(1,len(dataset["train"])),5)
        for idx in idxs:
            data=dataset['train'][idx]
            info = {}
            for i in data_format[dname]["keys"]:
                info[i] = data[i]
            info['choiceDict']={}
            info['choiceString']=''
            for i in range(len(info['choices']['label'])):
                info['choiceDict'][info['choices']['label'][i]]=info['choices']['text'][i]
                info['choiceString']+=f'({info["choices"]["label"][i]}) {info["choices"]["text"][i]} '
            # info['choiceString'].strip()
            info['answerString']=info['choiceDict'][info['answerKey']]
            prompt = data_format[dname]["prompt"]
            text+=prompt.format(**info)+"\n\n"
        for del_idx in sorted(idxs, reverse=True):
            dataset['train'].pop(del_idx)
        text += data_format[dname]["end"]
        # print(text)
    return text
    # elif dname=='StoryCloze':
    #     dataset=load_dataset(data_format[dname]['name'])
    #     text=data_format[dname]['front']
    #     idxs=random.sample(range(1,dataset["test"].num_rows),5) # test밖에 없음
    #     for idx in idxs:
    #         data=dataset['test'][idx]
    #         info = {}
    #         for i in data_format[dname]["keys"]:
    #             info[i] = data[i]
    #         info['answerString']=data['sentence_quiz1'] if data['answer_right_ending']==1 else data['sentence_quiz2']
    #         prompt = data_format[dname]["prompt"]
    #         text+=prompt.format(**info)+"\n\n"
    #     text += data_format[dname]["end"]
    #     print(text)

# import pdb; pdb.set_trace()