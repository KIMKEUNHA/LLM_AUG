import json
from datasets import load_dataset
import random
import re
import openai
from openai import OpenAI

import traceback
import os
# OpenAI API 키 설정

client = OpenAI(api_key='')
with open("prompts.json", "r") as f:
    prompts = json.load(f)
    
print( "위 코드는 저장된 프롬프트 데이터가 올바른지 확인을 위한 코드 입니다. 원하는 프롬프트 이름을 입력하세요." )

for i in range(200):
    print("현재 저장된 데이터 : ")
    for idx, prompt in enumerate(prompts.keys()):
        print(f"{idx+1}. " + str(prompt))
    name = "WNLI"
    if name not in prompts.keys():
        print("해당 데이터셋이 없습니다.")
        continue
    print("데이터를 불러옵니다.")
    prompt = prompts[name]
    dataset = load_dataset(prompt["name"])
    idxs = [x for x in range(dataset["train"].num_rows)]
    random.shuffle(idxs)
    text = prompt["fix"]
    num_label = prompt["num_label"]
    label_chk = {}
    len = 0
    status = {}
    for data in dataset["train"]:
        if data[prompt["label_name"]] in status.keys():
            status[data[prompt["label_name"]]]+=1
        else:
            status[data[prompt["label_name"]]]=0
    print(status)
    for idx in idxs:
        if len==4:
            break
        if dataset["train"][idx][prompt["label_name"]] not in label_chk:
            label_chk[dataset["train"][idx][prompt["label_name"]]]=1
        elif label_chk[dataset["train"][idx][prompt["label_name"]]] == 2:
            continue
        else:
            label_chk[dataset["train"][idx][prompt["label_name"]]]+=1
        data = dataset["train"][idx]
        info = {}
        for i in prompt["keys"]:
            info[i] = data[i]
        text += prompt["prompt"].format(**info) + "\n\n"
        len+=1
    text += prompt["end"]
    print(text)
    regex= r":\s*(.*?)\s*".join(prompt["regex"])+":\s*(.*?)$"
    responses = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages = [{"role": "system", "content": text}],
    )
    
    response_text = responses.choices[0].message.content
    # response_text = 'Premise: The team worked hard all season and finally won the championship.\nHypothesis: The team lost every game.\nLabel: not entailment\n\nPremise: The suspect was seen on security cameras robbing the convenience store.\nHypothesis: The suspect was arrested for shoplifting.\nLabel: entailment'
    new_data = response_text.split("\n\n")
    gens = []
    for response in new_data:
        match = re.search(regex, response, re.DOTALL)
        if match:
            temp={}
            for idx,reg in enumerate(prompt["regex"]):
                temp[reg] =  match.group(idx+1).strip()
            gens.append(temp)
    with open(name+"_fixed.jsonl",encoding="utf-8", mode="a") as f:
        for line in gens:
            f.write(json.dumps(line)+"\n")