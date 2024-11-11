

import os
import json
import csv
import pandas as pd
import google.generativeai as genai

from pprint import pprint


api_key = ""

genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-1.0-pro-latest')


def generate(model, prompt):
    response = model.generate_content(prompt)
    return response.text

file = open('variations.csv',encoding="utf8")


df = pd.read_csv(file)

for tweet,number in df[["corrected","nr"]].itertuples(index=False):
    print(number)
    # print(tweet)
    prompt = '''You are the automatic evaluator of natural language explanations.
                Instructions:

                You are required to carefully read and understand the natural language explanation provided to you.
                After comprehending the given explanation, you are asked to rate it on a scale of 1 to 7 based on three criteria: informativeness, clarity, and effectiveness.
                Your ratings should range from 1 (indicating the lowest score) to 7 (indicating the highest score).
                Definitions:
                Informativeness: How relevant is the information in an explanation?
                Clarity: How clear is the meaning of an explanation?
                Effectiveness: How effective is an explanation in serving its intended purpose or achieving its goals?

                Please answer as honestly as possible. Provide just scores.
    ''' +  'Explanation: ' + str(tweet) + '''
    Informativeness:
    Clarity:
    Effectiveness:
    '''
    # print(prompt)
    response = generate(model,prompt)
    # print(number)
    print(response)
    print('\n')
        # gen.append(response)
        # df1['results'] = response.choices[0].message.content

    
# df1.to_csv("resuklts_openai.csv", sep=",")
