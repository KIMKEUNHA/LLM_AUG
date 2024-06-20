
const first_prompts_dict={
    'imdb':`Given an example as a binary sentiment classification dataset consisting of texts from a movie review and labels of each text, perform the task of classifying if the text is positive or negative. Create at least two new data like the examples provided without numbering. Please ensure the text has more than 4 sentences.\n\nExample dataset :\n\n`,
    'boolq':`The following are examples of datasets used for binary classification tasks, specifically determining whether the answer to a given question is true or false based on information provided in a passage. Each dataset contains a question, a passage, and an answer. Refer to the example and create two new data. Please vary the topics of your questions to people, regions, places, science, common sense, life, society, etc. Make your questions as specific as possible. Avoid questions about famous topics or well-known places, people, animals, and regions. True or false labels must always match the facts. \n\nExample dataset :\n\n`,
    'piqa':`piqa prompt`,
    'wnli':`The Winograd Schema Challenge assesses a system's capability to determine whether a sentence or clause beginning with a pronoun accurately refers to the entity mentioned earlier in the text. This evaluation focuses on the system's proficiency in interpreting pronouns within the given context. Below are example datasets illustrating this concept. Refer to each example to generate at least two data.\n\nExample dataset :\n\n`,
    'rte':`In the GLUE benchmark, the Recognizing Textual Entailment (RTE) task evaluates models' capability to determine whether a hypothesis can be logically inferred from a given premise. Below are examples from the RTE task, showcasing both entailment and non-entailment relationships. Generate at least one new example for each label.\n\nExample dataset :\n\n`,
    'ddd': `ddd`,
};
const data_examples_dict={
    "imdb": "text: I watched the movie in a preview and I really loved it. The cast is excellent and the plot is sometimes absolutely hilarious. Another highlight of the movie is definitely the music, which hopefully will be released soon. I recommend it to everyone who likes the British humour and especially to all musicians. Go and see. It's great.\nlabel: positive\n\ntext: I saw this movie not knowing anything about it before hand. The plot was terrible with large gaps of information missing. The movie didn't have the 'battle of wits' feel to me. The actors just spewed out mouthful's of nonsense, at times causing me to gnash my teeth in agony as they droned on and on. The plot was predictable except for the stomach sickening homo erotic scene at the end (I'm not homophobic but made me physically sick to my stomach), even the ending was predictable. And you could tell the detective was Jude Law in a costume, everything from the fake accent, terrible dental work, costume shop facial hair, everything pointed to it being a disguise. The whole movie just felt like wasted time out of my life. This movie had the feel of a puppet show with Jude Law and Michael Caine as puppets and the house as the window to view the show, really boooring in my opinion.\nlabel: negative",
    'boolq':`Question: are there any original members of the little river band\nPassage: Little River Band have undergone numerous personnel changes, with over 30 members since their formation. None of the musicians now performing as Little River Band are original members, nor did they contribute to the success the band had in the 1970s. In the 1980s, members included John Farnham, David Hirschfelder, Stephen Housden, Wayne Nelson and Steve Prestwich. Currently the line-up is Nelson with Rich Herring, Greg Hind, Chris Marion and Ryan Ricks. Two former members have died, Barry Sullivan in October 2003 (aged 57) and Steve Prestwich in January 2011 (aged 56).\nAnswer: False\n\nQuestion: is the department of education a government agency\nPassage: The Department for Education (DfE) is a department of Her Majesty's Government responsible for child protection, education (compulsory, further and higher education), apprenticeships and wider skills in England. The DfE is also responsible for women and equalities policy.\nAnswer: True`,
    'piqa':`piqa ex`,
    'wnli':`Premise: Always before, Larry had helped Dad with his work. But he could not help him now, for Dad said that his boss at the railroad company would not want anyone but him to work in the office.\nHypothesis: Larry's boss at the railroad company would not want anyone but him to work in the office.\nLabel: not entailment\n\nPremise: Even before they reached town, they could hear a sound like corn popping. Dora asked what it was, and Dad said it was firecrackers.\nHypothesis: Dora asked what the sound was.\nLabel: entailment`,
    'rte':`Premise: Mr. Balasingham will return to his London home and then move on to Sri Lanka in early October to consult with LTTE leader Vilupillai Prabhakaran, diplomats said.\nHypothesis: Vilupillai Prabhakaran is a diplomat.\nLabel: not entailment\n\nPremise: Springsteen performed 27 songs Monday night, including most of the 12 new songs from his latest release, 'Devils and Dust', which was recorded without the E Street Band.\nHypothesis: Springsteen introduced some of the 12 new songs off his latest release, 'Devils andDust', which was recorded without the E Street Band.\nLabel: entailment`,
    'ddd': `ddd`,
};
const val_dict={
    "imdb": `IMDB is a large movie review dataset, a dataset for binary sentiment classification. The data should be evaluated to ensure that it has appropriate labels for binary sentiment classification and meets criteria such as clarity, grammatical accuracy, and contextual accuracy. If the data is found to be inappropriate, it should be deleted. Write a one-sentence comment about the evaluation, pre-add a “reason” key, and pre-add a “status” key to indicate “delete” if the data should be deleted or “appropriate” if the data is appropriate. The output only requires “reason” and “status” and must be in JSON format.\n\nInput data:\n\n`,
    // "imdb": `IMDB is a large movie review dataset, a dataset for binary sentiment classification. The data should be evaluated to ensure that it has appropriate labels for binary sentiment classification and meets criteria such as clarity, grammatical accuracy, and contextual accuracy. Write a one-sentence comment about the evaluation, and indicate whether the data should be deleted or the data is appropriate, adding the word "delete" or "appropriate" in the end.\n\nInput data:\n\n`,
    'boolq':`BoolQ is a large dataset for binary classification tasks, specifically determining whether the answer to a given question is true or false based on information provided in a passage. The data should be evaluated to ensure that it has appropriate labels for binary classification and meets criteria such as clarity, grammatical accuracy, and contextual accuracy. Based on the given passage, the questions and answers must absolutely match the facts. If the data is found to be inappropriate, it should be deleted. Write a one-sentence comment about the evaluation, pre-add a “reason” key, and pre-add a “status” key to indicate “delete” if the data should be deleted or “appropriate” if the data is appropriate. The output only requires “reason” and “status” and must be in JSON format.\n\nInput data:\n\n`,
    'piqa':`piqa ex`,
    'wnli':`The Winograd Schema Challenge assesses a system's capability to determine whether a sentence or clause beginning with a pronoun accurately refers to the entity mentioned earlier in the text. This evaluation focuses on the system's proficiency in interpreting pronouns within the given context. Below are example datasets illustrating this concept. Each dataset entry should be evaluated to ensure that it meets the following criteria: clarity, variety, grammatical correctness, and contextual correctness. If the data is found to be inappropriate, it should be deleted. Add a “status” key to the dictionary, labeling it “Delete” if the data should be deleted, or “Appropriate” if the data is appropriate. The output only requires “reason” and “status” and must be in JSON format.\n\ninput data :\n\n`,
    'rte':`In the GLUE benchmark, the Recognizing Textual Entailment (RTE) task evaluates models' capability to determine whether a hypothesis can be logically inferred from a given premise. Below are data from the RTE task. Each dataset entry should be evaluated to ensure that it meets the following criteria: clarity, variety, grammatical correctness, and contextual correctness. If the data is found to be inappropriate, it should be deleted. Add a “status” key to the dictionary, labeling it “Delete” if the data should be deleted, or “Appropriate” if the data is appropriate. The output only requires “reason” and “status” and must be in JSON format.\n\ninput data :\n\n`,
    'ddd': `ddd`,
}

const sample_new={
    'imdb':`"The Dark Knight is a gripping and intense thriller that showcases the battle between Batman and the Joker. Heath Ledger's legendary performance as the Joker is chilling and unforgettable. The movie explores complex themes such as chaos, morality, and the thin line between hero and villain. Christian Bale's portrayal of Batman is brooding and powerful, capturing the internal struggles of the caped crusader. The Dark Knight is filled with twisty plot twists, jaw-dropping action sequences, and a haunting soundtrack that adds to the suspense. This film is a masterpiece of superhero cinema that leaves a lasting impact on its viewers.", "label": "positive"}`
};
const sample_final={
    'imdb': `"The Dark Knight is a gripping and intense thriller that showcases the battle between Batman and the Joker. Heath Ledger's legendary performance as the Joker is chilling and unforgettable. The movie explores complex themes such as chaos, morality, and the thin line between hero and villain. Christian Bale's portrayal of Batman is brooding and powerful, capturing the internal struggles of the caped crusader. The Dark Knight is filled with twisty plot twists, jaw-dropping action sequences, and a haunting soundtrack that adds to the suspense. This film is a masterpiece of superhero cinema that leaves a lasting impact on its viewers.", "label": "positive", "reason": "The text provides a clear, grammatically accurate, and contextually appropriate review with a positive sentiment, which matches the provided label.", "status": "appropriate"}`
};

const keys_dict={
    'imdb':['text','label'],
    'boolq':["Question", "Passage", "Answer" ],
    'rte':["Premise","Hypothesis", "Label"],
    'piqa':["goal", "sol1", "sol2", "label"],
    'wnli':["Premise","Hypothesis", "Label"],
};
function parsing(dataset, raw_txt_arr){
    let keys=keys_dict[dataset];
    const regex = new RegExp(":\\s*(.*?)\\s*" + keys.join(":\\s*(.*?)\\s*") + ":\\s*(.*?)$");
    let gens = [];
    raw_txt_arr.forEach(txt => {
        let match=txt.match(regex);
        if (match){
            let temp={};
            keys.forEach((key, idx) => {
                temp[key] = match[idx + 1].trim();
            });
            gens.push(temp);
        }
    });
    console.log(gens);
    return gens;
};

const generateText3 = async (prompt) => {
    let json_prompt={'txt':prompt};
    // console.log(prompt);
    const response = await fetch('/get_response_3', {
        method: 'POST',
        body: JSON.stringify(json_prompt),
        headers: {
            'Content-Type': 'application/json'
        }
    });
    const result=await response.text();
    // console.log(result);
    return result;
};
const generateText4 = async (prompt) => {
    let json_prompt={'txt':prompt};
    // console.log(prompt);
    let response = await fetch('/get_response_4', {
        method: 'POST',
        body: JSON.stringify(json_prompt),
        headers: {
            'Content-Type': 'application/json'
        }
    });
    const result=await response.text();
    // console.log(result);
    return result;
};
// 대기 구현용
function sleep(time) {
    if(time<=0) return; // new Promise((r) => setTimeout(r, 1))
    return new Promise((r) => setTimeout(r, time));
}
//
const data_desc_1=`    IMDB: 영화 리뷰에 대한 감성 분석
    BoolQ: 문장의 참 거짓 판단
    PiQA: 문장의 인과관계 또는 질문에 대한 상식 데이터
    WNLI: 두 문장간의 관계 추론
    RTE: 두 문장의 함의 관계 추론`;

document.addEventListener('DOMContentLoaded', function() {
    // document.getElementById('promptBtn').style.display = 'none';
    document.getElementById('outputDisplay').style.display = 'none';
    document.getElementById('finalPromptBtn').style.display = 'none';
    document.getElementById('finalPromptDisplay').style.display = 'none';
    document.getElementById('nextStepBtn').style.display = 'none';
    document.getElementById('dataDesc1').innerText=data_desc_1;
});
document.getElementById('loadDataBtn').addEventListener('click', function() {
    const dataset = document.getElementById('dataset').value;
    let dataText=`데이터셋을 선택하세요`;
    dataText = data_examples_dict[dataset];

    document.getElementById('dataText').innerHTML = dataText;
    // document.getElementById('dataDisplay').classList.remove('hidden');
    // document.getElementById('promptBtn').style.display = 'inline-block';
    document.getElementById('promptBtn').disabled=false;
    document.getElementById('promptBtn').classList.remove("btn-secondary");
    document.getElementById('promptBtn').classList.add("btn-primary");
    document.getElementById('outputDisplay').style.display = 'none';
    document.getElementById('finalPromptDisplay').style.display = 'none';
    document.getElementById('nextStepBtn').style.display = 'none';
});


document.getElementById('promptBtn').addEventListener('click', function() {
    const dataset = document.getElementById('dataset').value;
    const outputText = first_prompts_dict[dataset];
    document.getElementById('outputText').innerHTML = outputText;
    document.getElementById('outputDisplay').style.display = 'block';
    document.getElementById('finalPromptBtn').style.display = 'inline-block';
    document.getElementById('finalPromptDisplay').style.display = 'none';
    document.getElementById('nextStepBtn').style.display = 'none';
});

document.getElementById('finalPromptBtn').addEventListener('click', function() {
    document.getElementById('finalPromptText').innerText = document.getElementById('outputText').value+data_examples_dict[document.getElementById('dataset').value];
    document.getElementById('finalPromptDisplay').style.display = 'block';
    document.getElementById('nextStepBtn').style.display = 'inline-block';
});

document.getElementById('nextStepBtn').addEventListener('click', async function() { // 데이터 생성(gpt3)
    let prompt=document.getElementById('finalPromptText').innerHTML;
    let result=generateText3(prompt);
    console.log(result);

    document.getElementById('output-exampleText').innerText=`출력 중`;
    document.getElementById('outputExample').classList.remove('hidden');
    document.getElementById('validateOutputBtn').classList.remove('hidden');

    const loading = (async function(){
        document.getElementById('nextStepBtn').classList.remove("btn-primary");
        document.getElementById('nextStepBtn').classList.add("btn-secondary");
        document.getElementById('nextStepBtn').disabled=true;

        document.getElementById('nextStepBtn').innerText=`출력 중`;

        let i=0;
        const time=2;
        for (i=0; i<time*4; i++){
            await sleep(250);
            if (i%5!=4){
                document.getElementById('nextStepBtn').innerText +=` .`;
            }
            else{
                document.getElementById('nextStepBtn').innerText=`출력 중`;
            }
        }
    })();
    document.getElementById('valid-prompt').innerText=val_dict[document.getElementById('dataset').value];
    document.getElementById('output-exampleText').innerText = await result;

    document.getElementById('nextStepBtn').classList.remove("btn-secondary");
    document.getElementById('nextStepBtn').classList.add("btn-primary");
    document.getElementById('nextStepBtn').disabled=false;
    const tmp = await loading;
    document.getElementById('nextStepBtn').innerText='4. 프롬프트로 데이터 증강';

    document.getElementById('outputExample').classList.remove('hidden');
    document.getElementById('validateOutputBtn').classList.remove('hidden');

    document.getElementById('validateOutputBtn').classList.remove("btn-secondary");
    document.getElementById('validateOutputBtn').classList.add("btn-primary");
    document.getElementById('validateOutputBtn').disabled=false;

});
//
document.getElementById('validateOutputBtn').addEventListener('click', async function() { //데이터 검수(gpt4)
    let val_prompt=val_dict[document.getElementById('dataset').value];
    console.log(val_prompt);
    let prompt=val_prompt+document.getElementById('output-exampleText').innerText;
    console.log(prompt);
    let result=generateText4(prompt);
    const loading = (async function(){
        document.getElementById('validateOutputBtn').classList.remove("btn-primary");
        document.getElementById('validateOutputBtn').classList.add("btn-secondary");
        document.getElementById('validateOutputBtn').disabled=true;

        document.getElementById('validateOutputBtn').innerText=`ChatGPT4로 검증 중`;
        let i=0;
        const time=2;
        for (i=0; i<time*4; i++){
            await sleep(250);
            if (i%5!=4){
                document.getElementById('validateOutputBtn').innerText +=` .`;
            }
            else{
                document.getElementById('validateOutputBtn').innerText=`ChatGPT4로 검증 중`;
            }
        }
    })();
    let result_json= await result;
    // console.log(result_json);
    // let result_txt=result_json[];
    result_txt=result_json;

    document.getElementById('finalOutputExampleText').innerText = result_txt;
    document.getElementById('validateOutputBtn').classList.remove("btn-secondary");
    document.getElementById('validateOutputBtn').classList.add("btn-primary");
    document.getElementById('validateOutputBtn').disabled=false;
    const tmp = await loading;
    document.getElementById('validateOutputBtn').innerText=`5. ChatGPT4로 데이터 검증`;

    document.getElementById('finalOutputExample').classList.remove('hidden');

});
///
