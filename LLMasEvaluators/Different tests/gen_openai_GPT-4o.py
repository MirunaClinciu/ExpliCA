
import os
import json
import csv
import pandas as pd
from openai import OpenAI

client = OpenAI(
  api_key=''
)

def generate(prompt):
    response = client.chat.completions.create(
    model="gpt-4o-2024-05-13",
    messages=[
        {"role": "system", 
         "content": prompt},
    ],
    temperature=1)

    return response


file = open('variations.csv',encoding="utf8")


df = pd.read_csv(file)


for tweet,number in df[["corrected","nr"]].itertuples(index=False):
    print(number)
    # print(tweet)
    prompt = '''
                You are the automatic evaluator  of natural language explanations.

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
    response = generate(prompt)
    # print(number)
    print(response.choices[0].message.content)
    print('\n')
        # gen.append(response)
        # df1['results'] = response.choices[0].message.content

    
# df1.to_csv("resuklts_openai.csv", sep=",")
