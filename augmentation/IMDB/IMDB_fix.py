
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

name = "IMDB_fixed"
with open(name+".jsonl") as f:
    for line in f:
        data_list.append( json.loads(line) )

# with open(name+"_re-fined_gpt4o.jsonl","r") as f:
#     start = len(f.readlines())
#     print(start)
start=0
for i in range(start,len(data_list)):
    # text = "Given an example as a cross-reference dataset consisting of sentences containing pronouns and a list of reference candidates, perform the task of determining which candidate the pronoun refers to. Use uncommon names. Use jobs from a variety of categories. Create at least two new data without numbering based on the examples provided.\n\nExample dataset :\n\n"
    text = """IMDB is a large movie review dataset, a dataset for binary sentiment classification. The data should be evaluated to ensure that it has appropriate labels for binary sentiment classification and meets criteria such as clarity, grammatical accuracy, and contextual accuracy. If the data is found to be inappropriate, it should be deleted. Write a one-sentence comment about the evaluation, pre-add a “reason” key, and pre-add a “status” key to indicate “delete” if the data should be deleted or “appropriate” if the data is appropriate. The output only requires “reason” and “status” and must be in JSON format.\n\nInput data:\n\n"""
    text+= json.dumps(data_list[i],indent=2,ensure_ascii=False)
    text+="\n\nEvaluated data :\n\n"
    while True:
        try:
            responses = client.chat.completions.create(
                model="gpt-4o",
                messages = [{"role": "system", "content": text}],
            )
            print(text)
            response_text = responses.choices[0].message.content
            response_text = response_text[response_text.find("{"):response_text.rfind("}")+1]
            print(response_text)
            import pdb; pdb.set_trace()
            # response_text = 'sentence: Sarah asked Lisa to lend her a book, but she refused.\npronoun: she\ncandidates: Sarah,Lisa\nlabel: Lisa\n\nsentence: When Tom saw Mark, he greeted him warmly.\npronoun: he\ncandidates: Tom,Mark\nlabel: Mark'
            new_data = json.loads(response_text)
            if "status" not in new_data.keys() or "reason" not in new_data.keys():
                continue
            data_list[i]["reason"] = new_data["reason"]
            data_list[i]["status"] = new_data["status"]
            with open(name+"_re-fined_gpt4o.jsonl",encoding="utf-8", mode="a") as f:
                f.write(json.dumps(data_list[i],ensure_ascii=False)+"\n")
            break
        except:
            print("다시")
            
