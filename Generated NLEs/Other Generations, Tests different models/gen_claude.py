

import os
import json
import csv
import pandas as pd


from anthropic import Anthropic

client = Anthropic(api_key="")




def generate(prompt):
    response= client.beta.tools.messages.create(
        model="claude-3-opus-20240229",
        system="You are the automatic evaluator of natural language explanations.",
        max_tokens=1000,
        temperature=1,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response


file = open('variations.csv',encoding="utf8")


df = pd.read_csv(file)


for tweet,number in df[["corrected","nr"]].itertuples(index=False):
    print(number)
    # print(tweet)
    prompt = '''
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
    print(response.content)
    print('\n')
        # gen.append(response)
        # df1['results'] = response.choices[0].message.content

    
# df1.to_csv("resuklts_openai.csv", sep=",")
