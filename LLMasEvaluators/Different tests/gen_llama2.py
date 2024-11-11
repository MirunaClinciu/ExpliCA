from transformers import AutoTokenizer
import transformers
import torch
# from huggingface_hub import login
# login()

import pandas as pd
from transformers import AutoModel


from transformers import AutoModel

access_token = ""


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModel.from_pretrained(model_id, token=access_token)


def llama(sentence):
    # model = "meta-llama/Llama-2-7b-chat-hf"
    # model = AutoModel.from_pretrained( "meta-llama/Llama-2-7b-chat-hf", token="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")

    # tokenizer = AutoTokenizer.from_pretrained(model)
    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model=model,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    # )

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
    )
    sequences = pipeline(
        sentence,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )
    for seq in sequences:
        return seq['generated_text']



file = open('variations.csv',encoding="utf8")

# df = pd.read_csv(file)

df = pd.read_csv(file)

df1 =  df[['corrected','T1', 'T2', 'T3']]

df1['combined'] = df1.apply(lambda x: list([x['T1'],
                                        x['T2'],
                                        x['T3']]),axis=1)   
df1['variations'] = df.apply(lambda x: list([x['V1'],
                                        x['V2'],
                                        x['V3'],
                                        x['V4'],
                                        x['V5'],
                                        x['V6'],
                                        x['V7'],
                                        x['V8'],
                                        x['V9']
                                        ]),axis=1)   
for tweet in df1['corrected']:
    for triple in df1['combined']:
        for variation in df1['variations']:


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

            Please answer as honestly as possible.


            ''' +  'Explanations: ' + str(tweet) 

    # print(prompt)
    print(llama(prompt))


    
