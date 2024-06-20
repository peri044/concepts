import json
from openai import OpenAI

# Load API key from a JSON file. 
# Make sure to replace "sk-..." with your actual API key from https://platform.openai.com/api-keys
with open("config.json", "r") as config_file:
    config = json.load(config_file)
    api_key = config["OPENAI_API_KEY"]

client = OpenAI(api_key=api_key)

# Run the chatgpt model
def run_chatgpt(prompt, client, model="gpt-4"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        seed=123,
    )
    return response.choices[0].message.content

json_file = "instruction-data-with-response.json"

with open(json_file, "r") as file:
    test_data = json.load(file)
    
print("Number of entries in the test set:", len(test_data))

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. Write a response that "
        f"appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    instruction_text + input_text

    return instruction_text + input_text

# Collect scores
scores=[]
for entry in test_data:
    prompt = (f"Given the input `{format_input(entry)}` "
              f"and correct output `{entry['output']}`, "
              f"score the model response `{entry['model_response']}`"
              f" on a scale from 0 to 100, where 100 is the best score. "
              f"Respond with the number only."
    )
    score = run_chatgpt(prompt, client)
    try:
        scores.append(int(score))
    except ValueError:
        continue

print(f"Number of scores: {len(scores)} of {len(test_data)}")
print(f"Average score: {sum(scores)/len(scores):.2f}\n")






