# 🧠 Instruction Tuning with Annotation Guidelines for Event Extraction (Findings of ACL 2025)
> Efficient and extensible Event Extraction with Code Prompts and Annotation Guidelines — built on top of [TextEE](https://github.com/ej0cl6/TextEE). 

This repository includes code for:
- `PyCode-TextEE`: Tools to obtain code prompts for 15 event extraction datasets supported by TextEE.
- `Instruction Tuning with Guidelines`: Source code to reproduce [our work on utlizing code prompts and annotation guidelines for Event Extraction](https://arxiv.org/abs/2502.16377). Please navigate to the directory `instruction_tuning_with_guidelines_ACL_2025` for the source code.

If you find our work helpful, please cite our work:
```
@inproceedings{srivastava-etal-2025-instruction,
    title = "Instruction-Tuning {LLM}s for Event Extraction with Annotation Guidelines",
    author = "Srivastava, Saurabh  and
      Pati, Sweta  and
      Yao, Ziyu",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.677/",
    pages = "13055--13071",
    ISBN = "979-8-89176-256-5",
    abstract = "In this work, we study the effect of annotation guidelines{--}textual descriptions of event types and arguments, when instruction-tuning large language models for event extraction. We conducted a series of experiments with both human-provided and machine-generated guidelines in both full- and low-data settings. Our results demonstrate the promise of annotation guidelines when there is a decent amount of training data and highlight its effectiveness in improving cross-schema generalization and low-frequency event-type performance."
}
```
---

<div align="center">

[![🛠️ Updates](https://img.shields.io/badge/🛠️%20Updates-Click%20Here-informational?style=flat-square)](#updates) •
[![📂 Datasets](https://img.shields.io/badge/📂%20Datasets-15%20Supported-brightgreen?style=flat-square)](#datasets) •
[![⚙️ Environment](https://img.shields.io/badge/⚙️%20Environment-Setup-orange?style=flat-square)](#environment) •
[![🚀 Running](https://img.shields.io/badge/🚀%20Running-Instructions-blueviolet?style=flat-square)](#running) •
[![🌐 Website](https://img.shields.io/badge/🌐%20Website-Demo%20Soon-lightgrey?style=flat-square)](#website) •
[![📄 Paper](https://img.shields.io/badge/📄%20arXiv-2502.16377-b31b1b?style=flat-square)](https://arxiv.org/abs/2502.16377)


</div>

---

**Authors**:  
Saurabh Srivastava, Sweta Pati, Ziyu Yao

---

## 🧩 Introduction

**PyCode-TextEE** extends [TextEE](https://github.com/ej0cl6/TextEE), bringing **event extraction into the era of prompt-based large language models**.

While **TextEE** standardizes 10+ event extraction datasets into a unified JSON format—making them reproducible and comparable—**PyCode-TextEE** takes the next leap:

> ✨ We transform TextEE-formatted data into **code-style prompts**—a format that is both readable and executable by LLMs and ideal for structured evaluation. In addition, we annotate the code-prompts with annotation guidelines. Below, we provide an example of code prompt and how we integrate annotation guidelines within them:

### What are Code Prompts and Annotation Guidelines?
- `Code prompting` is a technique that enhances reasoning abilities in text+code LLMs by transforming natural language (NL) tasks into code representations. Instead of executing the code, the model uses it as a structured input format to reason and generate answers. *The labels such event classes and arguments are represented as Python classes, and the guidelines or instructions are introduced as docstrings.* The model start generating after the `result =`  line.

- `Annotation Guidelines` involve defining how to identify and classify events and their arguments within a text or other data. These guidelines help ensure consistency and quality in the annotation process, which is crucial for training machine learning models for event extraction. The performance of current SoTA models heavily depends
on the quantity of human-annotated data, as the model learns the guidelines from these examples. 

⚠️ Note that not all datasets release their annotation guidelines. We provide code to generate these annotation guidelines automatically using a few training samples.

#### An example for a code prompt with annotation guidelines is shown below:
```python
# This is an event extraction task where the goal is to extract structured events from the text. A structured event contains an event trigger word, an event type, the arguments participating in the event, and their roles in the event. For each different event type, please output the extracted information from the text into python-style dictionaries where the first key will be 'mention' with the value of the event trigger. Next, please output the arguments and their roles following the same format. The event type definitions and their argument roles are defined next.

# Here are the event definitions:

@dataclass
class Meet(ContactEvent):
    """A 'Meet(ContactEvent)' is triggered by interactions where individuals or groups come together for a specific purpose, either physically or virtually. This event involves direct interaction, distinguishing it from remote communication events like 'PhoneWrite'. It encompasses formal and informal gatherings such as diplomatic talks, business meetings, press conferences, and forums, but excludes casual or unplanned encounters."""
    mention: str  # The text span that triggers the event.
    entity: List  # Entities are individuals, groups, organizations, or countries participating in the meeting. They represent the participants involved in the event.
    place: List  # The place is the location where the meeting occurs, providing context for the event. It can be a city, building, specific venue, or virtual platform. 

# This is the text to analyze
text = "The meeting concluded with the delegates voting by show of hands to meet again in 10 days."

result = [
    Meet(mention='meeting', entity=['delegates'], time=[], place=[]), 
    Meet(mention='meet', entity=['delegates'], time=['10 days'], place=[])
]
```
> PyCode-TextEE transforms EE datasets into the above format which have shown to perform well with LLMs. For more details, please refer to our paper [Instruction-Tuning LLMs for Event Extraction with Annotation Guidelines](https://arxiv.org/abs/2502.16377).
---

### 🚀 What’s New in PyCode-TextEE?

- **CodePrompt Format Conversion**  
  We convert event structures (event triggers, arguments—if available) into Python-like prompts (e.g., `Attack(mention="...", attacker=[...], target=[...])`) to help LLMs handle structured outputs.

- **Annotation Guideline Generation**
  While annotation guidelines have helped LLMs achieve SOTA results for EE, previous approaches assume that these guidelines are made available which is not always true. We take the next steps in generating these guidelines automatically from a few training samples. 

- **Plug-and-Play with TextEE**  
  Directly load standardized datasets from TextEE and transform them with one command into training-ready CodePrompts.

- **Evaluation Toolkit for Prompted LLMs**  
  We provide exact-match evaluation utilities that compute **precision, recall, and F1 scores** over structured LLM outputs.

- **Code to Reproduce LLaMAEvents**  
  Includes all data transformations and training scripts used for our paper on utilizing code prompts and annotation guidelines. Code for that will live in `LLaMAEvents/`.

<a id="updates"></a>
## 🛠️ Updates

- **April 23, 2025** — We release **PyCode-TextEE**, a modular framework for converting standardized event extraction datasets (via TextEE) into code-style prompts, along with exact-match evaluation scripts.  
  Feel free to reach out if you’d like to contribute your **models**, **datasets**, or ideas!

<a id="datasets"></a>
## 📂 Supported Datasets

We support **15 datasets** for Event Detection (ED), Event Argument Extraction (EAE), and End-to-End (E2E) Event Extraction.  All are converted into **code-style prompts** and support evaluation using our exact-match metric suite.  

The table below also shows whether annotation **guidelines** are included for each dataset.

<div align="center">
<table>
<thead>
<tr>
  <th><strong>Dataset</strong></th>
  <th><strong>Task(s)</strong></th>
  <th><strong>Paper Title</strong></th>
  <th><strong>Source</strong></th>
  <th><strong>Guidelines</strong></th>
</tr>
</thead>
<tbody>
<tr>
  <td><code>ACE05</code></td>
  <td>ED, EAE, E2E</td>
  <td>The Automatic Content Extraction (ACE) Program</td>
  <td><a href="https://www.ldc.upenn.edu/">LDC</a></td>
  <td>🔘</td>
</tr>
<tr>
  <td><code>ERE</code></td>
  <td>ED, EAE, E2E</td>
  <td>From Light to Rich ERE</td>
  <td><a href="https://www.ldc.upenn.edu/">LDC</a></td>
  <td>🔘</td>
</tr>
<tr>
  <td><code>MLEE</code></td>
  <td>ED, EAE, E2E</td>
  <td>Biological Event Extraction</td>
  <td><a href="https://academic.oup.com/bioinformatics/article/28/18/i575/245077">Bioinformatics</a></td>
  <td>⚪️</td>
</tr>
<tr>
  <td><code>Genia2011</code></td>
  <td>ED, EAE, E2E</td>
  <td>Genia Event Task (2011)</td>
  <td><a href="https://www.aclweb.org/anthology/W11-1801/">BioNLP 2011</a></td>
  <td>⚪️</td>
</tr>
<tr>
  <td><code>Genia2013</code></td>
  <td>ED, EAE, E2E</td>
  <td>Genia Event Task (2013)</td>
  <td><a href="https://www.aclweb.org/anthology/W13-5201/">BioNLP 2013</a></td>
  <td>⚪️</td>
</tr>
<tr>
  <td><code>M2E2</code></td>
  <td>ED, EAE, E2E</td>
  <td>Cross-media Structured Common Space</td>
  <td><a href="https://aclanthology.org/2020.acl-main.188/">ACL 2020</a></td>
  <td>⚪️</td>
</tr>
<tr>
  <td><code>CASIE</code></td>
  <td>ED, EAE, E2E</td>
  <td>CASIE: Cybersecurity Event Extraction</td>
  <td><a href="https://ojs.aaai.org/index.php/AAAI/article/view/6155">AAAI 2020</a></td>
  <td>⚪️</td>
</tr>
<tr>
  <td><code>PHEE</code></td>
  <td>ED, EAE, E2E</td>
  <td>Pharmacovigilance Event Extraction</td>
  <td><a href="https://aclanthology.org/2022.emnlp-main.343/">EMNLP 2022</a></td>
  <td>⚪️</td>
</tr>
<tr>
  <td><code>MEE</code></td>
  <td>ED</td>
  <td>Multilingual Event Extraction</td>
  <td><a href="https://aclanthology.org/2022.emnlp-main.428/">EMNLP 2022</a></td>
  <td>⚪️</td>
</tr>
<tr>
  <td><code>FewEvent</code></td>
  <td>ED</td>
  <td>Few-Shot Event Detection</td>
  <td><a href="https://dl.acm.org/doi/10.1145/3336191.3371962">WSDM 2020</a></td>
  <td>⚪️</td>
</tr>
<tr>
  <td><code>MAVEN</code></td>
  <td>ED</td>
  <td>Massive General-Domain ED</td>
  <td><a href="https://aclanthology.org/2020.emnlp-main.154/">EMNLP 2020</a></td>
  <td>⚪️</td>
</tr>
<tr>
  <td><code>SPPED</code></td>
  <td>ED</td>
  <td>ED from Social Media for Epidemic Prediction</td>
  <td><a href="https://aclanthology.org/2024.naacl-main.172/">NAACL 2024</a></td>
  <td>⚪️</td>
</tr>
<tr>
  <td><code>MUC-4</code></td>
  <td>EAE</td>
  <td>Fourth Message Understanding Conference</td>
  <td><a href="https://www-nlpir.nist.gov/related_projects/muc/">MUC 1992</a></td>
  <td>⚪️</td>
</tr>
<tr>
  <td><code>RAMS</code></td>
  <td>EAE</td>
  <td>Multi-Sentence Argument Linking</td>
  <td><a href="https://aclanthology.org/2020.acl-main.743/">ACL 2020</a></td>
  <td>🔘</td>
</tr>
<tr>
  <td><code>WikiEvents</code></td>
  <td>EAE</td>
  <td>Conditional Generation for Doc-level EAE</td>
  <td><a href="https://aclanthology.org/2021.naacl-main.417/">NAACL 2021</a></td>
  <td>🔘</td>
</tr>
<tr>
  <td><code>GENEVA</code></td>
  <td>EAE</td>
  <td>Benchmarking Generalizability for EAE</td>
  <td><a href="https://aclanthology.org/2023.acl-long.794/">ACL 2023</a></td>
  <td>🔘</td>
</tr>
</tbody>
</table>


</div>

<a id="environment"></a>
## ⚙️ Environment

Although there is no need of any additional package to run PyCode-TextEE, we recommend using **Python 3.9+** with a clean virtual environment (e.g., via `venv` or `conda`).

### 🔹 Install Dependencies
```bash
# Clone the repo
git clone https://github.com/yourname/PyCode-TextEE.git
cd PyCode-TextEE

# Create a virtual environment (optional)
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install requirements (optional)
pip install -r requirements.txt
```

### 🔹 Core Dependencies
> These are the minimal dependencies to run the code.
- `datasets`  
- `openai`  # (used for guideline generation)
- `wandb` # (optional for experiment tracking)  

### ⚠️ Note
Some datasets (e.g., ACE, ERE) require **LDC license** to access raw files.  We provide code for preprocessing them, but not the data itself.

<a id="running"></a>
## 🚀 Running the Code

Below is a step-by-step guide to run PyCode-TextEE.  
Our pipeline is divided into 4 main stages:

---
### ✣ Step 0 — Obtaining TextEE Format Dataset
Our code accepts data formatted after TextEE pre-processing. Please follow the instructions in `data` directory from the [TextEE repo](https://github.com/ej0cl6/TextEE/tree/main).

Make sure after running TextEE, you have data saved in the following structure:
#### 📂 Expected dataset layout:
```
<your_dataset_dir>/
├── ace05-en/
│   ├── split1/
│   │   ├── train.json
│   │   ├── dev.json
│   │   └── test.json
│   └── ...
├── casie/
└── ...
```

### 🔹 Step 1 — [Optional] Generate Code Schema

If you're working with custom datasets (or want to regenerate schemas for the 15 supported ones), you'll first convert them into **TextEE format** and generate the corresponding **Python-style event definitions**.

📁 Directory structure:
```
PyCode-TextEE/
├── code_schema_generation/
│   ├── generate_schema.py
│   ├── init_prompts/            # Contains per-dataset class schemas (*.txt)
│   ├── python_event_defs/       # Python classes for eval (dataset-wise + all_ee_definitions.py)
│   ├── mapper.json              # Maps cleaned names → class names
│   └── schema.json              # All cleaned event/arg schemas
```

🛠 To generate schema:
```bash
cd code_schema_generation
python generate_schema.py --dataset_folder <your_dataset_dir>
```
👾 Example output schema (for ACE05 `Attack` event):
```python
@dataclass
class Attack(ConflictEvent):
    mention: str
    target: List
    victim: List
    attacker: List
    instrument: List
    place: List
    agent: List
```

**Note**: We’ve already generated schema for all 15 supported datasets. This step is only required for new datasets.

---
### 🔹 Step 2: Generating Annotation Guidelines from a few Training Samples
While code prompts convert EE datasets into a structured format, annotating the schema with guidelines helps LLMs understand event and argument definitions. As shown in [our paper](https://arxiv.org/abs/2502.16377), annotated schema with these guidelines help us achive SOTA results with LLaMA-3-8.1B. However, not all datasets release these annotation guidelines and we address this in our paper by proposing 5 different ways to generate this guidelines. Specifically, we generate guidelines using following 5 variants discussed below:
- **Guideline-P**: Uses training samples from an event class e to generate guidelines. We denote such instances as positive samples in our approach.
- **Guideline-PN**: In addition to positive training samples, we also utilize 15 negative samples from different event classes to generate guidelines.
- **Guideline-PS**: We designate sibling event classes in event schema as negative samples and utilize them to generate guidelines.
- **Guideline-PN-Int and Guideline-PS-Int**: We create two more variants that Integrate the 5 diverse guideline samples from GuidelinePN and Guideline-PS into a comprehensive one,
respectively.

**Note**: We’ve already the synethesized guidelines and available human guidelines in directory `guideline_generation/synthesize_guidelines/synthesized_guidelines`

To (re)generate the guidelines, or to generate them for a new dataset, run the pipeline below from
the `guideline_generation` directory:
```bash
cd guideline_generation

# (0) Credentials. Azure OpenAI is the default; pass --api_type openai for standard OpenAI.
export OPENAI_API_KEY="<your key>"
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"   # Azure only

# (1) Convert TextEE processed data into the guideline "master" JSONL (the first hop).
python utils/textee_to_master.py -d <dataset_name> -i <TextEE/processed_data> \
    --split split1 --files train.json -o ./synthesize_guidelines/master_<dataset_name>.jsonl

# (2) Build the per-event prompts from the master file.
#     --neg_strategy sibling -> Guideline-PS, random -> Guideline-PN; add --examples_per_event 0 for
#     positive-only Guideline-P. Negative sampling is seeded (--seed) for reproducibility.
python synthesize_guidelines/synthesize_guidelines_w_neg_samples.py \
    --master_file ./synthesize_guidelines/master_<dataset_name>.jsonl \
    --dataset_name <prompt_set> --neg_strategy sibling --seed 1337 \
    --output_dir ./synthesize_guidelines/prompts

# (3) Generate the P / PN / PS guidelines from those prompts (add --dry_run to test without the API).
python prompting/prompt_llms.py --dataset_name <prompt_set> \
    --prompt_dir ./synthesize_guidelines/prompts \
    --out_dir ./synthesize_guidelines/synthesized_guidelines/

# (4) [Optional] Integrated variants (Guideline-PN-Int / PS-Int): build the consolidation prompts
#     from the generated guidelines, then prompt the LLM with them.
python synthesize_guidelines/synthesize_adv_guideline_PN.py \
    --input_dir ./synthesize_guidelines/synthesized_guidelines/<prompt_set> \
    --output_dir ./synthesize_guidelines/prompts/<prompt_set>_INT
python prompting/prompt_llm_adv_guidelines.py --dataset_name <prompt_set>_INT \
    --prompt_dir ./synthesize_guidelines/prompts \
    --out_dir ./synthesize_guidelines/synthesized_guidelines/

cd ..  # back to the repo root
```
where `<dataset_name>` is a TextEE dataset (e.g. `ace05-en`) and `<prompt_set>` is any name you pick
for the generated prompt/guideline subdir (e.g. `ace05_PS`). Pass `--dry_run` to the two `prompt_*`
scripts to validate the full pipeline offline (no API key or spend).
### 📘 Guideline File Format
After above code execution, the guidelines will be stored in the file `<output_file>`. Please make sure that your guideline file looks like:

```json
{
  "EventName1": {
    "description": [
      "One possible definition.",
      "Another variation of the same."
    ],
    "attributes": {
      "mention": "Trigger span of the event.",
      "arg_1": ["One definition for arg_1", "another definition for arg_1"]
    }
  }
}
```

This enables *randomized sampling* during conversion to avoid overfitting to one phrasing—an approach highlighted in our paper.

---

### 🔹 Step 3: Obtaining Code Prompts
We first need to make sure that python event definitions are in current environment to verify code prompts.
```bash
cd python_event_defs # this directory is already included in the code or can be generated using Step 1. You can find it in "PyCode-TextEE/code_schema_generation/python_event_defs"
export PYCODE_HOME=$(pwd)
export PYTHONPATH=$PYCODE_HOME:$PYCODE_HOME:$PYTHONPATH
cd ../../ # redirect the terminal to PyCode home directory
```
Run the following:

```bash
cd code_prompts
python prepare_dataset.py \
    --input_dir <your_dataset_directory> \
    --dataset_name <dataset_name> \
    --annotate_schema <True/False> \ #if unspecified, the schema will be left unannotated because the flag defaults to False.
    --guideline_file <guideline_file> \ #if unspecified, the guidelines will be generated automatically as specified in Step-2.
    --add_negative_samples <True/False> \ #used to reproduce our LLaMAEvents results.
    --output_dir ./processed_code_prompts/
```

---

### ⚙️ Argument Descriptions

| Argument               | Description |
|------------------------|-------------|
| `--input_dir`          | Path to TextEE-formatted JSONs (default: `../../TextEE/processed_data`) |
| `--dataset_name`       | Name of the dataset to process (e.g., `ace05-en`) |
| `--annotate_schema`    | Add class docstrings and inline comments using guidelines (default: `False`) |
| `--guideline_file`     | Guideline JSON file for schema annotation (required if `annotate_schema=True`) |
| `--add_negative_samples` | Add negative examples to training set (default: `False`) |
| `--output_dir`         | Where to save the converted code prompts (default: `./processed_code_prompts/`) |

---

### 🧬 Annotated Prompt Example (with Guidelines)

When `--annotate_schema=True`, we generate prompts like:

```python
@dataclass
class Event(ParentEvent):
    """the event definition"""
    mention: str  # Event trigger definition
    arg_1: List   # Definition of argument 1
    arg_2: List   # Definition of argument 2
```

This format supports **LLM-compatible** structure learning and improves interpretability.

---

### 💡 Tip
⓵ Skip `--guideline_file` and `--annotate_schema` if you're only interested in raw code prompts. If `annotate_schema` is True but the `guideline_file` is unspecified or not found, Step 2 will be executed automatically to produce `guideline_file`.

⓶ Use `--add_negative_samples` if you want to add negative sample per instance similar to [DEGREE](https://github.com/PlusLabNLP/DEGREE).

---

### 🔹 Step 4: Training Models
To train the model, you can use the following scripts with LLaMA models as default, simply run:
```bash
cd training_scripts
python train_completion.py # train a chat completion model with LLaMA-3.1-8B as backbone
```

You can also run following command to resume training from a checkpoint:
```bash
python resume_from_ckpt.py # please specify the checkpoint directory in the script. By default, it will download and run LLaMA-3.1-8B
```

## 🧪 Evaluation

Once you've trained your model to generate Python-style event prompts, you can use our evaluation suite in `code_evaluation/` to compute standard **precision, recall, and F1 scores** via exact-match comparison of predicted and gold structured outputs.

### 📁 Directory Overview
```
code_evaluation/
├── all_ee_definitions.py     # Event classes copied from schema generation (Step 1)
├── event_scorer.py           # 🔥 Main evaluation logic
├── utils_typing.py           # (Attribution to GoLLIE — type helper module)
```

---

### 📊 `event_scorer.py`: Evaluation in a Nutshell

The core script compares model-generated code prompts with gold ones using Python object introspection.

#### ✅ Key Features:
- Extracts arguments from predicted and gold event objects
- Computes **micro/macro F1** across all examples
- Identifies:
  - **Trigger-level mismatches**
  - **Argument-level hallucinations**
- Logs detailed stats (TP / FP / FN per role)

#### 🎯 Core Functions:
- `compute_f1(...)`: calculates precision, recall, and F1 from match counts
- `extract_objects(...)`: extracts fields except for `mention` to compare arguments
- `micro_ed_scores`: calculate micro f1 score on Event Detection task
- `micro_eae_scores`: calculate micro f1 score on Event Argument Extraction task
- `micro_e2e_scores`: calculate micro f1 score on End-to-End Event Extraction task
- `log_hallucinations_and_mismatches(...)`: logs mismatches like hallucinated roles

---

### 🧪 Run the Demo Evaluation

We provide a ready-to-run example in:

```
demo/e2e_demo.json
```

This file contains three illustrative cases:
 \- ✅ One fully correct prediction - 🟡 One partially correct - ❌ One incorrect

To run the evaluation:

```bash
cd code_evaluation
python event_scorer.py --input_file ./../demo/e2e_demo.json
```

