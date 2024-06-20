from data_tools import make_prompt, post_process, save_results
from openai import OpenAI
import re
client = OpenAI(api_key='')

def chat_gpt(text):
    responses = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages = [{"role": "system", "content": text}],
    )
    # print(responses)
    responses = responses.choices[0].message.content
    responses = responses.split("\n\n")
    return responses

while True:
    dname=input("BoolQ or OBQA\n")
    num=int(input("iteration number\n"))

    for i in range(num):
        prompt=make_prompt(dname)
        print(prompt)
        
        responses=chat_gpt(prompt)
        print(responses)
        
        results=post_process(responses, dname)
        print("result###")
        print(results)
        
        filename=f"additional/{dname}.jsonl"
        # filename=f'{dname}.jsonl'
        save_results(results, filename)
    
    tmp=input("Stop?")
    if tmp=="o":
        break
    
    