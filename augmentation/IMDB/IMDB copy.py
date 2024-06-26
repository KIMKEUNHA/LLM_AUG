import json
from datasets import load_dataset
import random
import re
import openai
from openai import OpenAI

import traceback
import os
import pdb 

client = OpenAI(api_key='')

for i in range(3000):
    
    
    text = "Given an example as a binary sentiment classification dataset consisting of texts from a movie review and labels of each text, perform the task of classifying if the text is positive or negative. Create at least two new data like the examples provided without numbering. Please ensure the text has more than 4 sentences.\n\nExample dataset :\n\n"
    
    
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
    import pdb; pdb.set_trace()
    text+="New data :\n\n"
    #pdb.set_trace()
    regex= r":\s*(.*?)\s*".join(keys)+":\s*(.*?)$"
    
    responses = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages = [{"role": "system", "content": text}],
    )
    response_text = responses.choices[0].message.content
    new_data = response_text.split("\n\n")
    
    gens = []
    for response in new_data:
        match = re.search(regex, response, re.DOTALL)
        if match:
            temp={}
            for idx,reg in enumerate(keys):
                temp[reg] =  match.group(idx+1).strip()
            gens.append(temp)
    
    with open(name+"_fixed.jsonl",encoding="utf-8", mode="a") as f:
        for line in gens:
            f.write(json.dumps(line)+"\n")