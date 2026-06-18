from openai import AzureOpenAI
from glob import glob
from tqdm import tqdm
import argparse
import backoff
import getpass
import base64
import openai
import json
import ast
import os
import re
import time
from openai import OpenAIError

dataset_mapper = {
    "richere-en": {
        "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/richere-en/Sep_master_data_richere-en.json", 
        "def_file": "/scratch/spati/tmp/LLaMA-Events_w_neg_samples/synthesize_guidelines/def_Sep_master_data_richere-en.json", 
        "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/richere-en/Master_event_dataclasses_richere-en.json", 
        "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/richere-en/richere-en_event_type_files"
    }
}

# ALLOWED_EVENTS_GENEVA = {
#     "A(Event)",
#     "B(Event)",
#     "C(Event)"
# }

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--llm", help="Choice of LLM (GPT-4, GPT-4o, GPT-3.5, etc.)", default="gpt-4o")
parser.add_argument("-i", "--prompt_dir", help = "Directory in which prompts are stored", default = "./../synthesize_guidelines/prompts")
parser.add_argument("-t", "--temperature", default = 0)
parser.add_argument("-m", "--max_tokens", default = 4096)
parser.add_argument("-p", "--top_p", default = 0.7)
parser.add_argument("-f", "--freq_pen", default = 0.0)
parser.add_argument("-n", "--pres_pen", default=0.0)
parser.add_argument("-o", "--out_dir", default="./../synthesize_guidelines/synthesized_guidelines/")
parser.add_argument("-d", "--dataset_name", default="richere-en")
# Backend / credentials (kept backward compatible: Azure remains the default).
parser.add_argument("--api_type", choices=["azure", "openai"], default="azure",
                    help="LLM backend: Azure OpenAI (default) or standard OpenAI.")
parser.add_argument("--azure_endpoint", default=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                    help="Azure endpoint URL (or set AZURE_OPENAI_ENDPOINT). Overrides the built-in default.")
parser.add_argument("--azure_deployment", default=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
                    help="Azure deployment name (defaults to the model name in MODEL_MAP).")
parser.add_argument("--api_version", default=os.environ.get("AZURE_OPENAI_API_VERSION"),
                    help="Azure API version (defaults to the value in MODEL_MAP).")
# Optional: only used to source the 'mention' description; a default is used if omitted.
parser.add_argument("--event_arg_ontology", default=None,
                    help="Optional path to Master_event_dataclasses_<dataset>.json. If omitted, "
                         "a default 'mention' description is used and no scratch path is needed.")
parser.add_argument("--dry_run", action="store_true",
                    help="Validate the pipeline (prompt loading, parsing, file writing) without calling the API.")

MODEL_MAP = {
    "gpt-4o": {"name": "gpt-4o-2024-05-13", "endpoint": "https://azure-openai-api-eastus2.openai.azure.com/", "api_version": "2024-04-01-preview"}
}

# Canonical mention description used across all released guidelines; fallback when no ontology is provided.
DEFAULT_MENTION = "The text span that triggers the event."

# Well-formed sample response used by --dry_run so the full parse/write path can be exercised offline.
DRY_RUN_RESPONSE = """Here are the guidelines:
```json
{
  "Event Definition": ["Definition 1", "Definition 2", "Definition 3", "Definition 4", "Definition 5"],
  "Arguments Definitions": {"entity": ["e1", "e2", "e3", "e4", "e5"], "place": ["p1", "p2", "p3", "p4", "p5"]}
}
```
"""


# guidelines = json.load(open("./../synthesize_guidelines/fewevent_event_dataclasses.json"))
@backoff.on_exception(
    backoff.expo,                    # Exponential backoff
    (openai.OpenAIError, IOError),   # Retry on specific exceptions
    max_tries=5                      # Maximum number of retries
)
def parse_response(response, event_name, guidelines):
    response = response if(type(response) == type({})) else ast.literal_eval(response)
    # print(response)
    event_name = event_name.replace("prompt_", "").strip()
    try:
        mention = guidelines[event_name]["attributes"]["mention"]
    except (KeyError, TypeError):
        mention = DEFAULT_MENTION
    return_dict = {event_name:{"description":None}, "attributes":{"mention": mention}}
    # print("<<<", return_dict)
    # print(">>>", type(response))
    return_dict[event_name]["description"] = response["Event Definition"]
    for key, value in response["Arguments Definitions"].items():
        return_dict["attributes"][key] = value 
    # return_dict[event_name]["description"]
    return return_dict

def get_response(model, prompt, args, client, event_name, guidelines, dataset_name):
    print("-*" * 50)
    print(prompt)
    event_name = os.path.split(event_name)[-1].replace(".txt", "")

    retries = 3  # Maximum number of retries
    delay = 20  # Start with a 20-second delay
    cooldown_period = 60  # Cooldown period in seconds if rate limit is hit
    batch_size = 5  # Number of requests to process before a cooldown
    token_limit = 50000  # Example token limit for manual cooldown
    total_tokens_used = 0  # Token usage tracker

    # Retry logic
    for attempt in range(retries):
        try:
            # Track token usage
            prompt_tokens = len(prompt.split())  # Rough token estimate from prompt
            total_tokens_used += prompt_tokens

            # Check if we are near the token limit
            if total_tokens_used > token_limit:
                print(f"Token usage approaching limit ({total_tokens_used}/{token_limit}). Applying cooldown...")
                time.sleep(cooldown_period)
                total_tokens_used = 0  # Reset after cooldown

            # Send the request to the OpenAI API (or use a canned response in dry-run mode)
            if getattr(args, "dry_run", False):
                response = DRY_RUN_RESPONSE
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    top_p=args.top_p,
                    frequency_penalty=args.freq_pen,
                    presence_penalty=args.pres_pen
                )
                response = response.choices[0].message.content
            
            # Save logs
            os.makedirs(f"./logs/{dataset_name}", exist_ok=True)
            with open(f"./logs/{dataset_name}" + event_name + ".json", "w") as f:
                json.dump({"Prompt": prompt, "Response": response}, f, indent = 4)

            # Extract JSON from the response
            pattern = re.compile(r'json\n\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}\n', re.DOTALL)
            matches = pattern.findall(response)
            matches = [match.strip('json\n').strip('\n') for match in matches]
            try:
                largest_json = max(matches, key=len)
            except:
                largest_json = response.replace("```", "").replace("json", "")

            print("--" * 50)
            print(largest_json)
            print("-*" * 50)

            # Save parsed response
            os.makedirs(f"{args.out_dir}", exist_ok=True)
            with open(f"{args.out_dir}/{event_name}.json", "w") as f:
                json.dump(parse_response(largest_json, event_name, guidelines), f, indent=4)

            # After processing a batch, wait for a cooldown
            if (attempt + 1) % batch_size == 0:
                print(f"Processed {batch_size} requests. Applying batch cooldown...")
                time.sleep(cooldown_period)

            # Add a delay between successful requests
            if not getattr(args, "dry_run", False):
                time.sleep(delay)
            return largest_json

        except OpenAIError as e:
            print(f"Rate limit exceeded. Retrying in {cooldown_period} seconds... (Attempt {attempt + 1}/{retries})")
            time.sleep(cooldown_period)  # Wait for the cooldown period before retrying
            cooldown_period *= 2  # Optionally double the cooldown period for exponential backoff

    # If all retries fail, raise an exception
    raise Exception(f"Failed to get response for {event_name} after {retries} retries due to rate limits.")


os.makedirs("./logs", exist_ok=True)
if __name__ == "__main__":
    args = parser.parse_args()
    dataset_name = args.dataset_name
    args.out_dir = os.path.join(args.out_dir, dataset_name)
    os.makedirs(args.out_dir, exist_ok=True)

    # Optional ontology: used only to source the per-event 'mention' description.
    # Resolve from --event_arg_ontology, else the mapper (if present), else skip (a default is used).
    ontology_path = args.event_arg_ontology
    if ontology_path is None and dataset_name in dataset_mapper:
        ontology_path = dataset_mapper[dataset_name].get("event_arg_ontology")
    if ontology_path and os.path.exists(ontology_path):
        guidelines = json.load(open(ontology_path))
    else:
        if ontology_path:
            print(f"[warn] event_arg_ontology not found at '{ontology_path}'; using default 'mention' description.")
        guidelines = {}

    model_name = MODEL_MAP[args.llm]["name"]
    client = None
    if not args.dry_run:
        openai_key = os.environ.get("OPENAI_API_KEY")
        if(openai_key is None):
            os.environ["OPENAI_API_KEY"] = openai_key = getpass.getpass(prompt='Enter your OPENAI_API_KEY: ')
        if args.api_type == "azure":
            client = AzureOpenAI(
                api_key=openai_key,
                api_version=args.api_version or MODEL_MAP[args.llm]["api_version"],
                azure_endpoint=args.azure_endpoint or MODEL_MAP[args.llm]["endpoint"],
                azure_deployment=args.azure_deployment or model_name,
            )
        else:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
    else:
        print("[dry_run] Skipping API client creation; using canned responses.")

    prompt_files = glob(os.path.join(os.path.join(args.prompt_dir, dataset_name), "*.txt"))
    if not prompt_files:
        print(f"[warn] No prompt .txt files found under {os.path.join(args.prompt_dir, dataset_name)}. "
              f"Check --prompt_dir and --dataset_name (e.g. ACE_15random_examples).")
    # Skip events already generated in the output directory (resumable).
    existing_files = [os.path.split(ff)[-1].replace("prompt_", "").replace(".json", "") for ff in glob(os.path.join(args.out_dir, "*.json"))]
    # print(existing_files)
    for prompt_file in tqdm(prompt_files, total = len(prompt_files), desc = "Prompting ..."):
        event_name = os.path.split(prompt_file)[-1].replace("prompt_", "").replace(".txt", "")

        # # Only process allowed events for Geneva
        # if dataset_name == "wikievents" and event_name not in ALLOWED_EVENTS_GENEVA:
        #     print(f"Skipping event {event_name} as it is not in the allowed list for Geneva.")
        #     continue  # Skip events not in the allowed list

        if(event_name in existing_files):
            print(f"Guidelines for event {event_name} already exists.")
            continue
        # xx
        prompt = "".join([lines for lines in open(prompt_file)])
        #model, prompt, args, client, event_name, guidelines, dataset_name):
        get_response(model_name, prompt, args, client, prompt_file, guidelines, dataset_name)
        # break

        # Add a delay to avoid hitting the rate limit
        if not args.dry_run:
            time.sleep(2)



    # "ACE": {
    #     "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/ace05-en/Sep_master_data_ace05-en.json", 
    #     "def_file": "/scratch/spati/tmp/LLaMA-Events_w_neg_samples/synthesize_guidelines/def_Sep_master_data_ace05-en.json", 
    #     "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/ace05-en/Master_event_dataclasses_ace05-en.json", 
    #     "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/ace05-en/ace05-en_event_type_files"
    # }




##################################################################################################
#THE OG
##################################################################################################
# def get_response(model, prompt, args, client, event_name, guidelines, dataset_name):
#     print("-*"*50)
#     print(prompt)
#     event_name = os.path.split(event_name)[-1].replace(".txt", "")
#     # xx
#     response = client.chat.completions.create(model=model,messages=[{"role": "user","content": prompt}],
#         temperature=args.temperature,
#         max_tokens=args.max_tokens,
#         top_p=args.top_p,
#         frequency_penalty=args.freq_pen,
#         presence_penalty=args.pres_pen
#     )
#     response = response.choices[0].message.content
#     os.makedirs(f"./logs/{dataset_name}", exist_ok=True)
#     with open(f"./logs/{dataset_name}" + event_name + ".json", "w") as f:
#         json.dump({"Prompt": prompt, "Response": response}, f, indent = 4)
#     pattern = re.compile(r'json\n\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}\n', re.DOTALL)
#     matches = pattern.findall(response)
#     matches = [match.strip('json\n').strip('\n') for match in matches]
#     try:
#         largest_json = max(matches, key=len)
#     except:
#         largest_json = response.replace("```", "").replace("json", "")
#     print("--"*50)
#     print(largest_json)
#     print("-*"*50)
#     os.makedirs(f"{args.out_dir}", exist_ok=True)
#     with open(f"{args.out_dir}/{event_name}.json", "w") as f:
#         json.dump(parse_response(largest_json, event_name, guidelines), f, indent = 4)
#     time.sleep(2)
#     return largest_json

# ##################################################################################################
# #THE MODIFIED
# ##################################################################################################
# def get_response(model, prompt, args, client, event_name, guidelines, dataset_name):
#     print("-*" * 50)
#     print(prompt)
#     event_name = os.path.split(event_name)[-1].replace(".txt", "")

#     retries = 3  # Maximum number of retries
#     delay = 20  # Start with a 5-second delay

#     for attempt in range(retries):
#         try:
#             # Send the request to the Azure OpenAI service
#             response = client.chat.completions.create(
#                 model=model,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=args.temperature,
#                 max_tokens=args.max_tokens,
#                 top_p=args.top_p,
#                 frequency_penalty=args.freq_pen,
#                 presence_penalty=args.pres_pen
#             )
            
#             # Extract the response content
#             response = response.choices[0].message.content
#             os.makedirs(f"./logs/{dataset_name}", exist_ok=True)
#             with open(f"./logs/{dataset_name}/{event_name}.json", "w") as f:
#                 json.dump({"Prompt": prompt, "Response": response}, f, indent=4)

#             # Use regex to extract JSON from the response
#             pattern = re.compile(r'json\n\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}\n', re.DOTALL)
#             matches = pattern.findall(response)
#             matches = [match.strip('json\n').strip('\n') for match in matches]

#             try:
#                 largest_json = max(matches, key=len)
#             except:
#                 largest_json = response.replace("```", "").replace("json", "")

#             print("--" * 50)
#             print(largest_json)
#             print("-*" * 50)

#             os.makedirs(f"{args.out_dir}", exist_ok=True)
#             with open(f"{args.out_dir}/{event_name}.json", "w") as f:
#                 json.dump(parse_response(largest_json, event_name, guidelines), f, indent=4)

#             # Add a 5-second delay between requests
#             time.sleep(5)
#             return largest_json

#         except OpenAIError as e:
#             print(f"Rate limit exceeded. Retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
#             time.sleep(delay)  # Wait for the current delay
#             delay *= 2  # Double the delay for exponential backoff

#     # If all retries fail, raise an exception
#     raise Exception(f"Failed to get response for {event_name} after {retries} retries due to rate limits.")


# Hi Professor,

# I’m still encountering rate limit errors despite implementing several measures:

# Retry Logic: Limited retries to 3 with exponential backoff (starting at 20 seconds).
# Cooldown Period: Added a 60-second cooldown after hitting the rate limit.
# Token Tracking: Monitored token usage to avoid exceeding limits.
# Batch Processing: Grouped requests into manageable batches with cooldowns between them.
# Logging: Maintained detailed logs for debugging.
# Even after reducing the number of examples, the issue persists. It seems the Azure OpenAI instance might be hitting its quota prematurely. Let me know if there’s anything else I should check.
