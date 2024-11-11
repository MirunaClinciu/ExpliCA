import os
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
        # max_tokens=120,
        # max_tokens=50,
        max_tokens=20,

        temperature=1
    )

    # Access the content of the response correctly
    return response.choices[0].message.content

# Read the CSV file
df = pd.read_csv('variations.csv', encoding="utf8")

# Prepare a list to store the generations
generations = []

# Loop through each row and generate explanations
for tweet, number, t1, t2, t3 in df[["corrected", "nr", "T1", "T2", "T3"]].itertuples(index=False):
    prompt = '''
    Create a causal natural language explanation based on the given concept-set.''' +  '\n' +  'Concept-set: ' + '[' + str(t1) + ', ' + str(t2) + ', ' + str(t3) + ']'
    response = generate(prompt)
    
    # Append the generated explanation to the list
    generations.append(response)

# Save the generations to a new CSV file
output_df = pd.DataFrame(generations, columns=['Generated Explanation'])
# output_df.to_csv('generated_explanations_gpt-4o-2024-05-13.csv', index=False)
# output_df.to_csv('generated_explanations_gpt-4o-maxtokens120.csv', index=False)
# output_df.to_csv('generated_explanations_gpt-4o-maxtokens50.csv', index=False)
output_df.to_csv('generated_explanations_gpt-4o-maxtokens20.csv', index=False)

print("Generated explanations have been saved to 'generated_explanations.csv'.")
