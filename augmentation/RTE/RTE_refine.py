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
data_list = []
with open("RTE_fixed.jsonl") as f:
    for line in f:
        data_list.append( json.loads(line) )

name = "RTE"
with open(name+"_re-fined_gpt4o.jsonl") as f:
    start = len(f.readlines())
    print(start)
    import pdb; pdb.set_trace()
for i in range(start,len(data_list),2):
    print("데이터를 불러옵니다.")
    # text = "Given an example as a cross-reference dataset consisting of sentences containing pronouns and a list of reference candidates, perform the task of determining which candidate the pronoun refers to. Use uncommon names. Use jobs from a variety of categories. Create at least two new data without numbering based on the examples provided.\n\nExample dataset :\n\n"
    text = """In the GLUE benchmark, the Recognizing Textual Entailment (RTE) task evaluates models' capability to determine whether a hypothesis can be logically inferred from a given premise. Below are data from the RTE task. Each dataset entry should be evaluated to ensure that it meets the following criteria: clarity, variety, grammatical correctness, and contextual correctness. If the data is found to be inappropriate, it should be deleted. Add a “status” key to the dictionary, labeling it “Delete” if the data should be deleted, or “Appropriate” if the data is appropriate.Do not create any additional new data, and omit any comments about your evaluation. The output should be in json format.\n\ninput data :\n\n"""
    num_label = 2
    example_num = 2
    label_chk = {}
    len = 0
    status = {}
    # for data in dataset["train"]:
    #     if data["label"] in status.keys():
    #         status[data["label"]]+=1
    #     else:
    #         status[data["label"]]=0
    # print(status)
    pass_dic = {}
    pass_dic['data']=[]
    for k in [i,i+1]:
        try:
            pass_dic['data'].append( data_list[k])
        except:
            print("last")
    text+= json.dumps(pass_dic,indent=2)
    text+="\n\nEvaluated data :\n\n"
    responses = client.chat.completions.create(
        model="gpt-4o",
        messages = [{"role": "system", "content": text}],
    )
    
    response_text = responses.choices[0].message.content
    response_text = response_text[response_text.find("{"):response_text.rfind("}")+1]
    print(response_text)
    
    # response_text = 'sentence: Sarah asked Lisa to lend her a book, but she refused.\npronoun: she\ncandidates: Sarah,Lisa\nlabel: Lisa\n\nsentence: When Tom saw Mark, he greeted him warmly.\npronoun: he\ncandidates: Tom,Mark\nlabel: Mark'
    new_data = json.loads(response_text)
    
    gens = []
    
    with open(name+"_re-fined_gpt4o.jsonl",encoding="utf-8", mode="a") as f:
        if type(new_data) is type([]):
            for line, exc in zip(new_data,pass_dic['data']):
                    try:
                        f.write(json.dumps(line)+"\n")
                    except:
                        f.write(json.dumps(exc)+"\n")
        elif 'data' in new_data.keys():
            for line, exc in zip(new_data['data'],pass_dic['data']):
                try:
                    f.write(json.dumps(line)+"\n")
                except:
                    f.write(json.dumps(exc)+"\n")
            