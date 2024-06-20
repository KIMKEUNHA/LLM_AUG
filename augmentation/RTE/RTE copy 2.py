import json
from datasets import load_dataset
import random
import re
import openai
import traceback
import os
# OpenAI API 키 설정
openai.api_key = ''

with open("prompts.json", "r") as f:
    prompts = json.load(f)
    
print( "위 코드는 저장된 프롬프트 데이터가 올바른지 확인을 위한 코드 입니다. 원하는 프롬프트 이름을 입력하세요." )

while True:
    print("현재 저장된 데이터 : ")
    for idx, prompt in enumerate(prompts.keys()):
        print(f"{idx+1}. " + str(prompt))
    name = "STS"
    if name not in prompts.keys():
        print("해당 데이터셋이 없습니다.")
        continue
    print("데이터를 불러옵니다.")
    prompt = prompts[name]
    dataset = load_dataset(prompt["name"])
    idxs = [x for x in range(dataset["train"].num_rows)]
    random.shuffle(idxs)
    text = prompt["front"]
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
        if len==2*num_label:
            break
        if dataset["train"][idx][prompt["label_name"]] not in label_chk:
            label_chk[dataset["train"][idx][prompt["label_name"]]]=1
        elif label_chk[dataset["train"][idx][prompt["label_name"]]] == 1:
            continue
        else:
            label_chk[dataset["train"][idx][prompt["label_name"]]]+=1
        data = dataset["train"][idx]
        info = {}
        for i in prompt["keys"]:
            info[i] = data[i]
        text += "sentence : "+info["sentence"]+"\n+"+"pronoun : "+info["pronoun"]+"\n+"+"candidates : "+info["candidates"]+"\n+"+"label : "+info["candidates"][info["label"]]+"\n
        len+=1
        import pdb; pdb.set_trace()
    text += prompt["end"]
    print(text)
    regex= r":\s*(.*?)\s*".join(prompt["regex"])+":\s*(.*?)$"
    responses = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        response_format={ "type": "json_object" },
        messages = [{"role": "system", "content": text[:50]}],
    )
    import pdb; pdb.set_trace()
    gens = {}
    id = 0
    for response in responses[1:]:
        import pdb; pdb.set_trace()
        match = re.search(regex, prompt["regex"][0]+response, re.DOTALL)
        
        if match:
            gens[id] = {}
            for idx,reg in enumerate(prompt["regex"]):
                gens[id][reg] =  match.group(idx+1).strip()
            id+=1
    print(gens)