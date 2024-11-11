

import os
import json
import csv
import pandas as pd


from anthropic import Anthropic

client = Anthropic(api_key="")




def generate(prompt):
    response= client.beta.tools.messages.create(
        model="claude-3-opus-20240229",
        system="You are the generator of causal natural language explanations.",
        max_tokens=100,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response


file = open('variations.csv',encoding="utf8")


df = pd.read_csv(file)


for tweet,number,t1,t2,t3 in df[["corrected","nr", "T1","T2","T3"]].itertuples(index=False):
    print(number)
    # print(tweet)
    prompt = '''
    Create a causal natural language explanation based on the given concept-set and the human reference.''' + '\n' + 'Human reference: ' + str(tweet) + '\n' +  'Concept-set: ' + '[' + str(t1) + ', ' + str(t2) + ', ' + str(t3) + ']'
    print(prompt)
    response = generate(prompt)
    # print(number)
    print('Generated'+ '\n')
    print(response.content)
    print('\n')
        # gen.append(response)
        # df1['results'] = response.choices[0].message.content

    
# df1.to_csv("resuklts_openai.csv", sep=",")
