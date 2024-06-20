import json
from datasets import load_dataset
import random
import re
import openai
from openai import OpenAI

import traceback
import os

data=[]
with open("additional/BoolQ_re-fined_gpt4o.jsonl","r") as f:
    for line in f:
        data.append( json.loads(line) )
cnt=0
with open("additional/BoolQ_gold_gpt4o.jsonl",encoding="utf-8", mode="w") as f:
    for line in data:
        if line['status']=='appropriate':
            cnt+=1
            del(line['status'])
            del(line['reason'])
            f.write(json.dumps(line,ensure_ascii=False)+"\n")
print(str(cnt)+"/"+str(len(data)))
