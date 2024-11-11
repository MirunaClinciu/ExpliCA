

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
        temperature=1,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response


file = open('variations.csv',encoding="utf8")


df = pd.read_csv(file)


# Prepare a list to store the generations
generations = []


for tweet,number,t1,t2,t3 in df[["corrected","nr", "T1","T2","T3"]].itertuples(index=False):
    print(number)
    # print(tweet)
    prompt = '''
    Create a causal natural language explanation based on the given concept-set.''' +  '\n' +  'Concept-set: ' + '[' + str(t1) + ', ' + str(t2) + ', ' + str(t3) + ']'
    print(prompt)
    response = generate(prompt)
    # print(number)
    print('Generated'+ '\n')
    print(response.content)
    print('\n')
    response = generate(prompt)
    
    # Append the generated explanation to the list
    generations.append(response)

# Save the generations to a new CSV file
output_df = pd.DataFrame(generations, columns=['Generated Explanation'])
output_df.to_csv('generated_explanations_claude.csv', index=False)

print("Generated explanations have been saved to 'generated_explanations.csv'.")
