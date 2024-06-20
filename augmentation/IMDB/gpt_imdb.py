import json
from datasets import load_dataset
import random
import re
import openai
from openai import OpenAI
import traceback
import os

client = OpenAI(api_key='')

while True:
    name = "IMDB"
    print("데이터를 불러옵니다.")
    dataset = load_dataset("stanfordnlp/imdb")
    
    idxs = [x for x in range(dataset["train"].num_rows)]
    random.shuffle(idxs)
    text = "Given an example as a binary sentiment classification dataset consisting of texts from a movie review and labels of each text, perform the task of classifying if the text is positive or negative. Create at least two new data like the examples provided without numbering. Please ensure the text has more than 4 sentences.\n\nExample dataset :\n\n"
    num_label = 2
    example_num = 2
    label_chk = {}
    len = 0
    status = {}
    keys = ["text", "label"]

    for idx in idxs:
        if len==example_num *num_label:
            break
        if dataset["train"][idx]["label"] not in label_chk:
            label_chk[dataset["train"][idx]["label"]]=1
        elif label_chk[dataset["train"][idx]["label"]] == example_num :
            continue
        else:
            label_chk[dataset["train"][idx]["label"]]+=1
        data = dataset["train"][idx]
        info = {}
        for i in keys:
            info[i] = data[i]
        
        text+= "text: "+info["text"]+"\n"+"label: "+("positive" if info['label']==1 else "negative") + "\n\n"
        len+=1
    text+="New data :\n\n"
    import pdb; pdb.set_trace()
    regex= r":\s*(.*?)\s*".join(keys)+":\s*(.*?)$"
 
    #import pdb; pdb.set_trace()
    responses = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages = [{"role": "system", "content": text}],
    )
    import pdb; pdb.set_trace()
    response_text = responses.choices[0].message.content
 
    #response_text = 'sentence: Sarah asked Lisa to lend her a book, but she refused.\npronoun: she\ncandidates: Sarah,Lisa\nlabel: Lisa\n\nsentence: When Tom saw Mark, he greeted him warmly.\npronoun: he\ncandidates: Tom,Mark\nlabel: Mark'
    #import pdb; pdb.set_trace()
    new_data = response_text.split("\n\n")
    
    gens = []
    for response in new_data:
        match = re.search(regex, response, re.DOTALL)
        if match:
            temp={}
            for idx,reg in enumerate(keys):
                temp[reg] =  match.group(idx+1).strip()
            gens.append(temp)
    print(gens)
    #import pdb; pdb.set_trace()
    with open(name+".jsonl",encoding="utf-8", mode="a") as f:
        for line in gens:
            f.write(json.dumps(line)+"\n")
    break