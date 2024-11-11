import os
import csv
import pandas as pd
from anthropic import Anthropic
import time

# Initialize the API client with your API key
client = Anthropic(api_key="")

def generate(prompt: str) -> str:
    """
    Generate explanation using Claude API with error handling and rate limiting.
    """
    try:
        # Create the message using the correct API structure
        message = client.messages.create(
       
            model="claude-3-opus-20240229",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=100,  # Specify the maximum token limit
            temperature=0.7
        )
        
        # Access the response content correctly
        return message.content
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return ""
    finally:
        # Add rate limiting
        time.sleep(1)  # Wait 1 second between requests

# Three prompts for generating explanations
prompts = [
    lambda tweet, t1, t2, t3: (
        f"Create a brief and complete causal explanation based on the following concept-set:\n"
        f"Human reference: {tweet}\n"
        f"Concept-set: [{t1}, {t2}, {t3}]"
    ),
    lambda tweet, t1, t2, t3: (
        "Create a causal natural language explanation based on the given concept-set.\n"
        f"Concept-set: [{t1}, {t2}, {t3}]"
    ),
    lambda tweet, t1, t2, t3: (
        "Generate a brief, complete causal explanation, ideally around 20 tokens.\n"
        f"Create a brief and complete causal explanation based on the following concept-set:\n"
        f"Concept-set: [{t1}, {t2}, {t3}]"
    )
]

try:
    # Load the CSV data
    df = pd.read_csv('variations.csv', encoding="utf8")

    # Generate explanations and save them
    for idx, prompt_func in enumerate(prompts, start=1):
        generations = []
        total_rows = len(df)
        
        for row_idx, (tweet, number, t1, t2, t3) in enumerate(df[["corrected", "nr", "T1", "T2", "T3"]].itertuples(index=False), 1):
            print(f"Processing row {row_idx}/{total_rows} for prompt {idx}")
            
            prompt = prompt_func(tweet, t1, t2, t3)
            response = generate(prompt)
            generations.append({
                "Tweet": tweet,
                "Generated Explanation": response,
                "Prompt": f"Prompt {idx}"
            })

            # Print the generated explanation for monitoring
            print(f"Generated explanation: {response}\n")

        # Save each generation set to a separate CSV file
        output_df = pd.DataFrame(generations)
        output_file = f'generated_explanations_prompt{idx}.csv'
        output_df.to_csv(output_file, index=False)
        print(f"Generated explanations have been saved to '{output_file}'.")

except FileNotFoundError:
    print("Error: variations.csv file not found")
except pd.errors.EmptyDataError:
    print("Error: variations.csv is empty")
except Exception as e:
    print(f"An unexpected error occurred: {str(e)}")
