import os
import csv
import pandas as pd
from openai import OpenAI

# Load the API key from an environment variable for security
client = OpenAI(
  api_key=''
)

def generate(prompt):
    try:
        response = client.chat.completions.create(
           # model="gpt-4-turbo",
            model="gpt-4o",

            messages=[
                {
                    "role": "system", 
                    "content": "Generate a brief, complete causal explanation, ideally around 20 tokens."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7
        )
        text = response.choices[0].message.content.strip()
        # Truncate the response if it exceeds approximately 20 tokens
        truncated_text = ' '.join(text.split()[:20])
        return truncated_text
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

# Read the CSV file
df = pd.read_csv('variations.csv', encoding="utf8")

# Prepare a list to store the generations
generations = []

# Loop through each row and generate explanations
for tweet, number, t1, t2, t3 in df[["corrected", "nr", "T1", "T2", "T3"]].itertuples(index=False):
    # Define the prompt with the specific format
    prompt = (
        "Create a brief and complete causal explanation based on the following concept-set:\n"
        f"Concept-set: [{t1}, {t2}, {t3}]"
    )
    
    # Generate response
    response = generate(prompt)
    
    # Append the generated explanation to the list with tweet as additional context
    generations.append({
        "Tweet": tweet,
        "Generated Explanation": response
    })

# Save the generations to a new CSV file
output_df = pd.DataFrame(generations)
output_df.to_csv('prompt3_openai_gpt4o.csv', index=False)

print("Generated explanations have been saved to 'prompt3_openai_gpt4o.csv'.")
