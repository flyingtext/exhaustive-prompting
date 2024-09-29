from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
import requests
import pandas as pd
import json

def get_response_from_ollama(question):
    req = requests.post('http://192.168.0.112:11434/api/chat', data=json.dumps({
        "model": "gemma2:2b-instruct-fp16",
        "messages": [
            {
                'role': 'system',
                'content': 'Must answer with format that user asked. Do not say anthing other than the asked.'
            }
            {
            "role": "user",
            "content": question
            }
        ],
        'stream': False
    }))
    return json.loads(req.text)['message']['content']

ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")

df = pd.DataFrame(ds['test'])

for n, row in df.iterrows() :
    question = row['question']
    choices = row['choices']
    answer_key = row['answerKey']

    prob_texts = choices['text']
    prob_labels = choices['label']

    correct = prob_labels.index(answer_key)

    req = ''

    print('Questioning')

    question_rewrite = get_response_from_ollama('''Rewrite the following question in aspect of specifying what, where, when, why, who, how. Question: ''' + question)

    print(question_rewrite)

    req = req + question_rewrite + '\n\n'

    choices_rewrite = []

    for i, c in enumerate(choices) :
        choice_rewrite = get_response_from_ollama('''Rewrite the following selection choice for arbitrary question in aspect of specifying what, where, when, why, who, how. Selection choice: ''' + c)
        print(choice_rewrite)

        choices_rewrite.append(choice_rewrite)

        req = req + '[Choice ' + str(i) + '] ' + choice_rewrite + '\n\n'

    res = get_response_from_ollama(req + '\n\n Answer format : [Choice (between 1 to ' + str(len(choices)) + ')]')

    print(res)

