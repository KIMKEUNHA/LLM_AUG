from flask import Flask, request, render_template
from openai import OpenAI

import os
client = OpenAI(api_key='')

def chat_gpt3(text):
    responses = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages = [{"role": "system", "content": text}],
    )
    # print(responses)
    responses = responses.choices[0].message.content
    # responses = responses.split("\n\n")
    # print(text)
    print(responses)
    return responses

def chat_gpt4(text):
    responses = client.chat.completions.create(
        model="gpt-4o",
        messages = [{"role": "system", "content": text}],
    )
    # print(responses)
    responses = responses.choices[0].message.content
    responses = responses[responses.find("{"):responses.rfind("}")+1]
    # responses = responses.split("\n\n")
    # print(text)
    print(responses)
    return responses

app = Flask(__name__)
@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route('/get_response_3', methods=['GET','POST'])
def get_response_3():
    if request.method=='POST':
        prompt=request.get_json(silent=True)
        print("PROMPT:")
        print(prompt)
        response=chat_gpt3(prompt['txt'])
        return response
    
@app.route('/get_response_4', methods=['GET','POST'])
def get_response_4():
    if request.method=='POST':
        prompt=request.get_json(silent=True)
        print("PROMPT:")
        print(prompt)
        response=chat_gpt4(prompt['txt'])
        return response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 88)