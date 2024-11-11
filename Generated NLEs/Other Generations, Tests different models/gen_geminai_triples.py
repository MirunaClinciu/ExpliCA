import os
import pandas as pd
import google.generativeai as genai

# Configure the API key for the Gemini model
api_key = ""
genai.configure(api_key=api_key)

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.0-pro-latest')

def generate(model, prompt):
    # Generate content with temperature set to 1
    response = model.generate_content(prompt)
    return response.text

# Read the CSV file
df = pd.read_csv('variations.csv', encoding="utf8")

# Prepare a list to store the generations
generations = []

# Loop through each row and generate explanations
for tweet, number, t1, t2, t3 in df[["corrected", "nr", "T1", "T2", "T3"]].itertuples(index=False):
    prompt = f'''
    Create a causal natural language explanation based on the given concept-set.
    Concept-set: [{str(t1)}, {str(t2)}, {str(t3)}]'''
    
    # Generate the response using the model
    response = generate(model, prompt)
    
    # Append the generated explanation to the list
    generations.append(response)

# Save the generations to a new CSV file
output_df = pd.DataFrame(generations, columns=['Generated Explanation'])
output_df.to_csv('generated_explanations_gemini.csv', index=False)

print("Generated explanations have been saved to 'generated_explanations_gemini.csv'.")
