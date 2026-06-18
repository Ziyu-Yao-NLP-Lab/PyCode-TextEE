from tabulate import tabulate
from tqdm import tqdm
import argparse
import json
import re
import os
import random

MAX_SAMPLES = 10

dataset_mapper = {
    "ACE": {
        "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/ace05-en/Sep_master_data_ace05-en.json", 
        "def_file": "/scratch/spati/tmp/LLaMA-Events_w_neg_samples/synthesize_guidelines/def_Sep_master_data_ace05-en.json", 
        "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/ace05-en/Master_event_dataclasses_ace05-en.json", 
        "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/ace05-en/ace05-en_event_type_files"
    },
    "richere-en": {
        "master_file": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/richere-en/Sep_master_data_richere-en.json", 
        "def_file": "/scratch/spati/tmp/LLaMA-Events_w_neg_samples/synthesize_guidelines/def_Sep_master_data_richere-en.json", 
        "event_arg_ontology": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/richere-en/Master_event_dataclasses_richere-en.json", 
        "example_dir": "/scratch/spati/tmp/NLP_Research_Work/TextEE/a_final_preprocessing/synthesize_guidelines/guideline_generation_data/richere-en/richere-en_event_type_files"
    }
}

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset_name", help="Dataset Name (used for the output prompts/<dataset_name> subdir)", default="richere-en")
# New-dataset mode: derive everything (event types, argument lists, positive/negative example
# pools) from a single TextEE-style master JSONL, so no uncommitted def_file/ontology/example_dir
# are required. Each line: {"text": ..., "event_mentions": {"<Event(Parent)>": {"results": [ {...} ]}}}.
parser.add_argument("--master_file", default=None,
                    help="Path to a TextEE-style master JSONL. If given, def_file/event_arg_ontology/"
                         "example_dir are derived from it and are not needed.")
# Explicit-paths mode (legacy): override the hardcoded dataset_mapper entries.
parser.add_argument("--def_file", default=None, help="Override def_file path.")
parser.add_argument("--event_arg_ontology", default=None,
                    help="Override ontology path (optional; argument names are taken from def_file if absent).")
parser.add_argument("--example_dir", default=None, help="Override per-event example_dir path.")
parser.add_argument("--output_dir", default="prompts", help="Where prompt subdirs are written (default: ./prompts).")
parser.add_argument("--neg_strategy", choices=["sibling", "random"], default="sibling",
                    help="Negative sampling: 'sibling' (Guideline-PS) or 'random' across event types (Guideline-PN).")
parser.add_argument("--num_negatives", type=int, default=10, help="Target number of negative examples.")
parser.add_argument("--examples_per_event", type=int, default=1, help="Negative examples drawn per selected event type.")
parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES, help="Max positive examples per event type.")
parser.add_argument("--seed", type=int, default=1337, help="Random seed (makes negative sampling reproducible).")


PROMPT = """You are an expert in annotating NLP datasets for event extraction.Your task is to generate precise and detailed annotation guidelines for the event type [###Event_Type###]

### Input Format ###
```
Event Schema:
Event Name and its parent class
Arguments:
Arguments separated by new lines. If there are no arguments None will be given.

Examples:
Examples provided to illustrate the event type and arguments.
```

### Instructions ###
1. Identify and List All Unique Arguments: 
   - Carefully review the schema to identify all arguments relevant to the event type.
   - Please remember that the examples may not cover all the arguments in the list. In some cases, you may not have arguments at all, in such cases, you can have an empty list for arguments. 
2. Define the Event Type: Write 5 clear and specific definitions, starting with "The event is triggered by ...":
   - Include example triggers.
   - Highlight key characteristics and scope of the event.
   - Compare and contrast with closely related events using provided negative examples.
   - Explain how triggers and outcomes differ for similar event types.
3. Define Each Argument:** For each argument, provide 5 definitions with detailed examples, starting with "Examples are ...":
   - Explain the role and importance of each argument.
   - Include domain knowledge and address edge cases to clarify ambiguities.
4. Focus on Distinctions: Use positive examples to describe the event, and negative examples to clarify what the event is not. Explicitly state differences using phrases like:
   - "Unlike [Related Event Type], this event does not ..."
   - "Triggers such as [Trigger] are indicative of [Related Event Type], not this event type."
5. Structured Output: Present the output in the following JSON format:
   ```json
   {
     "Event Definition": [
       "Definition 1",
       "Definition 2",
       "Definition 3",
       "Definition 4",
       "Definition 5"
     ],
     "Arguments Definitions": {
       "Argument1": [
         "Definition 1",
         "Definition 2",
         "Definition 3",
         "Definition 4",
         "Definition 5"
       ],
       "Argument2": [
         "Definition 1",
         "Definition 2",
         "Definition 3",
         "Definition 4",
         "Definition 5"
       ]
       // Add more arguments if applicable
     }
   }
   ```

### Output Requirements ###
- Use detailed yet concise language for event and argument definitions.
- Incorporate diverse and domain-relevant examples for each definition.
- Avoid copying examples directly from provided data, create unique variations.

"""
def create_prompt(event_type, positive_examples, negative_examples, display_name, all_guidelines):
    demos = f"\n"
    schema_information = all_guidelines[event_type]
    schema_information_desc = f"Event Schema:\n{event_type.replace('(', ' which is a child event type of super class ').replace(')', '')}\n"
    schema_information_desc += "Arguments:\n"

    # Use a separate counter to correctly number arguments
    arg_counter = 1
    for key in schema_information["attributes"].keys():
        if key == "mention":
            continue
        schema_information_desc += f"Argument {arg_counter} -> {key}\n"
        arg_counter += 1

    # Add positive examples
    demos += "### The below examples are positive examples, as they match the Event Type being annotated: ###\n\n"
    for idx, example in enumerate(positive_examples):
        text = example["text"]
        event_mentions = example["event_mentions"][event_type]["results"]
        for event_mention in event_mentions:
            trigger = event_mention["mention"]
            arguments = [(key, value) for key, value in event_mention.items() if key != "mention"]
            demos += f"Example {idx + 1}\n"
            demos += f"#### Event Type ####\n{event_type}\n"
            demos += f"### Input Text ###\n{text.strip()}\n"
            demos += f"### Event Trigger ###\n{trigger.strip()}\n"
            demos += "### Event Arguments ###\n"
            for arg_key, arg_val in arguments:
                demos += f"For argument \"{arg_key}\" extracted spans {arg_val}\n"
            demos += "\n"

    # Add negative examples
    # demos += "### The below examples are negative examples, as they are from different Event Types for contrast: ###\n\n"
    demos += "### The following examples are negative examples, as they illustrate different event types provided for contrast and differentiation: ###\n\n"
    for idx, example in enumerate(negative_examples, start=len(positive_examples) + 1):
        negative_event_type = next(iter(example["event_mentions"]))
        text = example["text"]
        event_mentions = example["event_mentions"][negative_event_type]["results"]
        for event_mention in event_mentions:
            trigger = event_mention["mention"]
            arguments = [(key, value) for key, value in event_mention.items() if key != "mention"]
            demos += f"Example {idx}\n"
            demos += f"#### Event Type ####\n{negative_event_type}\n"
            demos += f"### Input Text ###\n{text.strip()}\n"
            demos += f"### Event Trigger ###\n{trigger.strip()}\n"
            demos += "### Event Arguments ###\n"
            for arg_key, arg_val in arguments:
                demos += f"For argument \"{arg_key}\" extracted spans {arg_val}\n"
            demos += "\n"

    prompt = PROMPT.replace("[###Event_Type###]", event_type.replace("(", " which is a child event type of super class ").replace(")", ""))
    prompt += schema_information_desc.strip() + "\n\n" + demos.strip()
    return prompt

def create_examples(event_type_dict, event_type, full_argument_list):
    print("****"*50)
    print("Positive Examples")
    def extract_field_types(results):
        field_types = set()
        for result in results:
            field_types.update(result.keys())
        return field_types

    sorted_data = sorted(event_type_dict, key=lambda x: len(x["event_mentions"][event_type]["results"]), reverse=True)

    all_fields = set(full_argument_list) - set(["mention"]) # Assuming 'full_argument_list' contains all fields required
    covered_fields = set()
    selected_examples = []

    while covered_fields != all_fields and len(selected_examples) < MAX_SAMPLES:
        # print(covered_fields)
        # print(all_fields)
        # print('-'*100)
        best_example = None
        best_new_fields = set()

        for example in sorted_data:
            if example in selected_examples:
                continue  # Skip examples already added
            results = example["event_mentions"][event_type]["results"]
            # print(results)
            # xx
            fields = extract_field_types(results)- set(["mention"])
            # print(fields)
            # xx
            new_fields = fields - covered_fields
            if new_fields and len(new_fields) > len(best_new_fields):
                best_example = example
                best_new_fields = new_fields

        if best_example:
            selected_examples.append(best_example)
            covered_fields.update(best_new_fields)
    
    if len(selected_examples) < MAX_SAMPLES:
        remaining_examples = [ex for ex in sorted_data if ex not in selected_examples]
        remaining_slots = MAX_SAMPLES - len(selected_examples)
        selected_examples.extend(remaining_examples[:remaining_slots])

    # formatted_examples = tabulate(selected_examples, headers="keys")
    print(f"Finished event type {event_type} with {len(selected_examples)} samples. Arguments were: {full_argument_list}. Total {len(selected_examples)} number of samples")
    print(f"All positive examples look like: {selected_examples}")
    return selected_examples#formatted_examples

#############################################################################################################################################################
# ############# This below is the function to add negative samples such that if I want to add 10 samples then I will from 4 different event types ###################
#############################################################################################################################################################
# def create_negative_examples(all_event_types, current_event_type, dataset_name, num_negatives):
#     print("****" * 50)
#     print("Negative Examples")

#     # Exclude the current event type and 'None' from the pool of event types to select negatives
#     selectable_event_types = [et for et in all_event_types if et != current_event_type and et != "None"]
#     print(f"Selectable event types for negatives: {selectable_event_types}")

#     negative_examples = []
#     num_types = min(4, len(selectable_event_types))  # Determine the number of event types to select
#     num_selected_per_type = max(1, num_negatives // num_types)  # Ensure at least one negative per type

#     print(f"Number of negatives to select per type: {num_selected_per_type}")

#     # Randomly select different event types to fetch negatives from
#     selected_event_types = random.sample(selectable_event_types, num_types)
#     print(f"Selected event types for negatives: {selected_event_types}")

#     # Function to count attributes in an example's results section
#     def count_attributes_in_results(example, event_type):
#         results = example['event_mentions'][event_type]['results']
#         return max(len(result.keys()) for result in results) if results else 0

#     # Fetch examples from each of the selected event types
#     for event_type in selected_event_types:
#         file_path = f"{dataset_mapper[dataset_name]['example_dir']}/{event_type}.json"
#         try:
#             with open(file_path, 'r') as file:
#                 examples = [json.loads(line) for line in file]
                
#                 # Sort examples by the number of attributes in the results section
#                 examples_sorted = sorted(
#                     examples,
#                     key=lambda ex: count_attributes_in_results(ex, event_type),
#                     reverse=True
#                 )

#                 print(f"Loaded and sorted {len(examples_sorted)} examples from {event_type}")

#                 # Select top examples based on attribute coverage
#                 num_to_select = min(num_selected_per_type, len(examples_sorted))
#                 selected_samples = examples_sorted[:num_to_select]
#                 negative_examples.extend(selected_samples)

#                 print(f"Selected {len(selected_samples)} negatives from {event_type}")
#                 for sample in selected_samples:
#                     attr_count = count_attributes_in_results(sample, event_type)
#                     print(f"Example with {attr_count} attributes: {sample['text']}")
#         except FileNotFoundError:
#             print(f"Warning: File not found for event type {event_type} at {file_path}")

#     # print("Final negative examples list:")
#     # for idx, example in enumerate(negative_examples):
#     #     event_type_key = next(iter(example['event_mentions']))
#     #     attr_count = count_attributes_in_results(example, event_type_key)
#     #     print(f"Negative Example {idx + 1} ({event_type_key} with {attr_count} attributes): {example['text']}")
#     # print("-----" * 50)
#     print("Negative examples list:")
#     print(negative_examples)  # Directly print the list of all negative examples

#     print("****" * 50)
#     return negative_examples


# ########################################################Guideline-PN################################################################################################
# # ############# This below is the function to add negative samples such that I add n example from 15 randomly choosen event types ###################
# ############################################################################################################################################################
# def create_negative_examples(all_event_types, current_event_type, dataset_name, num_negatives, examples_per_event=1):
#     print("****" * 50)
#     print("Negative Examples")

#     # Exclude the current event type and 'None' from the pool of event types to select negatives
#     selectable_event_types = [et for et in all_event_types if et != current_event_type and et != "None"]
#     print(f"Selectable event types for negatives: {selectable_event_types}")

#     # Randomly select up to 15 event types (or fewer if there are less than 15 available)
#     selected_event_types = random.sample(selectable_event_types, min(15, len(selectable_event_types)))
#     print(f"Randomly selected event types for negatives: {selected_event_types}")

#     negative_examples = []

#     # Function to count attributes in an example's results section
#     def count_attributes_in_results(example, event_type):
#         results = example['event_mentions'][event_type]['results']
#         return max(len(result.keys()) for result in results) if results else 0

#     # Fetch examples from each of the selected event types
#     for event_type in selected_event_types:
#         file_path = f"{dataset_mapper[dataset_name]['example_dir']}/{event_type}.json"
#         try:
#             with open(file_path, 'r') as file:
#                 examples = [json.loads(line) for line in file]
                
#                 # Sort examples by the number of attributes in the results section
#                 examples_sorted = sorted(
#                     examples,
#                     key=lambda ex: count_attributes_in_results(ex, event_type),
#                     reverse=True
#                 )

#                 print(f"Loaded and sorted {len(examples_sorted)} examples from {event_type}")

#                 # Select up to `examples_per_event` examples from this event type
#                 num_to_select = min(examples_per_event, len(examples_sorted))
#                 selected_samples = examples_sorted[:num_to_select]
#                 negative_examples.extend(selected_samples)

#                 print(f"Selected {len(selected_samples)} negatives from {event_type}")
#                 for sample in selected_samples:
#                     attr_count = count_attributes_in_results(sample, event_type)
#                     print(f"Example with {attr_count} attributes: {sample['text']}")
#         except FileNotFoundError:
#             print(f"Warning: File not found for event type {event_type} at {file_path}")

#     print("Negative examples list:")
#     print(negative_examples)  # Directly print the list of all negative examples

#     print("****" * 50)
#     return negative_examples

#############################################################Guideline-PS#########################################################################################
############ This below is the function to add negative samples such that I add 1 examples each from only the sibling event types  ###################
#############################################################################################################################################################
def _load_event_examples(event_type, example_pools=None, example_dir=None):
    """Return the example list for an event type from in-memory pools or from example_dir files."""
    if example_pools is not None:
        return list(example_pools.get(event_type, []))
    file_path = f"{example_dir}/{event_type}.json"
    try:
        with open(file_path, 'r') as file:
            return [json.loads(line) for line in file]
    except FileNotFoundError:
        print(f"Warning: File not found for event type {event_type} at {file_path}")
        return []


def create_negative_examples(all_event_types, current_event_type, num_negatives, examples_per_event=1,
                             strategy="sibling", example_pools=None, example_dir=None):
    print("****" * 50)
    print(f"Negative Examples (strategy={strategy})")

    if strategy == "sibling":
        # Guideline-PS: negatives are sibling event types sharing the same superclass.
        current_superclass = current_event_type.split('(')[-1].rstrip(')')
        candidate_event_types = [
            et for et in all_event_types
            if et != current_event_type and et != "None"
            and et.split('(')[-1].rstrip(')') == current_superclass
        ]
        print(f"Sibling event types for negatives: {candidate_event_types}")
    else:
        # Guideline-PN: negatives are randomly drawn across all other event types.
        candidate_event_types = [et for et in all_event_types if et != current_event_type and et != "None"]

    # Randomly select up to 15 event types (seeded in __main__ for reproducibility).
    selected_event_types = random.sample(candidate_event_types, min(15, len(candidate_event_types)))
    print(f"Selected event types for negatives: {selected_event_types}")

    negative_examples = []

    def count_attributes_in_results(example, event_type):
        results = example['event_mentions'][event_type]['results']
        return max(len(result.keys()) for result in results) if results else 0

    for event_type in selected_event_types:
        examples = _load_event_examples(event_type, example_pools=example_pools, example_dir=example_dir)
        if not examples:
            continue
        examples_sorted = sorted(
            examples,
            key=lambda ex: count_attributes_in_results(ex, event_type),
            reverse=True
        )
        num_to_select = min(examples_per_event, len(examples_sorted))
        negative_examples.extend(examples_sorted[:num_to_select])

    print("****" * 50)
    return negative_examples

def _display_name(event_type):
    name = event_type.split("(")[0]
    parent = event_type[event_type.find("(") + 1:event_type.find(")")] if "(" in event_type else ""
    spaced = re.sub(r"(\w)([A-Z])", r"\1 \2", name)
    parent_spaced = re.sub(r"(\w)([A-Z])", r"\1 \2", parent)
    return f"Event Type: {spaced}\nParent Event Type: {parent_spaced}"


def build_from_master(master_file):
    """Derive (def-like event_type_dict, ontology-like all_guidelines, per-event example_pools)
    from a single TextEE-style master JSONL, so no separate def_file/ontology/example_dir are needed."""
    event_type_dict, example_pools, all_guidelines = {}, {}, {}
    with open(master_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            jsn = json.loads(line)
            for event_type, payload in jsn.get("event_mentions", {}).items():
                if event_type == "None":
                    continue
                example_pools.setdefault(event_type, []).append(jsn)
                entry = event_type_dict.setdefault(event_type, {"DisplayName": _display_name(event_type), "Arguments": set()})
                for result in payload.get("results", []):
                    for k in result:
                        if k != "mention":
                            entry["Arguments"].add(k)
    for et, entry in event_type_dict.items():
        entry["Arguments"] = sorted(entry["Arguments"])
        all_guidelines[et] = {"attributes": {"mention": ""}}
        for a in entry["Arguments"]:
            all_guidelines[et]["attributes"][a] = ""
    return event_type_dict, all_guidelines, example_pools


if __name__ == "__main__":
    args = parser.parse_args()
    random.seed(args.seed)
    MAX_SAMPLES = args.max_samples  # used by create_examples (module global)
    dataset_name = args.dataset_name

    example_pools = None
    example_dir = None
    if args.master_file:
        event_type_dict, all_guidelines, example_pools = build_from_master(args.master_file)
    else:
        entry = dataset_mapper.get(dataset_name, {})
        def_file = args.def_file or entry.get("def_file")
        ontology_path = args.event_arg_ontology or entry.get("event_arg_ontology")
        example_dir = args.example_dir or entry.get("example_dir")
        if not def_file or not os.path.exists(def_file):
            raise SystemExit(f"def_file not found: {def_file}. Pass --master_file (recommended) or --def_file.")
        with open(def_file) as f:
            event_type_dict = json.load(f)
        if ontology_path and os.path.exists(ontology_path):
            all_guidelines = json.load(open(ontology_path))
        else:
            # Ontology only contributes argument names; derive them from def_file when it is absent.
            all_guidelines = {}
            for et, e in event_type_dict.items():
                if et == "None":
                    continue
                all_guidelines[et] = {"attributes": {"mention": ""}}
                for a in e.get("Arguments", []):
                    if a != "mention":
                        all_guidelines[et]["attributes"][a] = ""

    all_event_types = list(event_type_dict.keys())
    out_base = os.path.join(args.output_dir, dataset_name)
    os.makedirs(out_base, exist_ok=True)

    for event_type in event_type_dict:
        if event_type == "None":
            continue
        full_argument_list = event_type_dict[event_type]["Arguments"]
        if example_pools is not None:
            positive_pool = example_pools.get(event_type, [])
        else:
            positive_pool = _load_event_examples(event_type, example_dir=example_dir)
        examples = create_examples(positive_pool, event_type, full_argument_list)
        negative_examples = create_negative_examples(
            all_event_types, event_type,
            num_negatives=args.num_negatives, examples_per_event=args.examples_per_event,
            strategy=args.neg_strategy, example_pools=example_pools, example_dir=example_dir,
        )
        prompt = create_prompt(event_type, positive_examples=examples, negative_examples=negative_examples,
                               display_name=event_type_dict[event_type].get("DisplayName", ""),
                               all_guidelines=all_guidelines)
        with open(os.path.join(out_base, "prompt_" + event_type + ".txt"), "w") as f:
            f.write(prompt)
        print(f"Wrote prompt for {event_type}")
    print(f"Done. {sum(1 for et in event_type_dict if et != 'None')} prompts written to {out_base}/")