# SwiftGuard
##### *Emphasizing both speed and protection* 
SwiftGuard is an algorithm designed to protect LLMs from malicious prompts attempting to 'jailbreak' the system. 
With the implemenation of SwiftGuard, LLMs can be protected from being tricked into responding with illegal or harmful information, at little
to no cost to the typical user. 

## Setup

### **Step 1: Creating Enviorment**
```bash
conda create -n swiftguard python=3.10
conda activate swiftguard
```

### **Step 2: Installing packages**
```bash
cd spd
cd jailbreakbench
pip install jailbreakbench
```

### **Step 3: Installing additional packages**
```bash
cd ..
cd ..
pip install -r requirements.txt
```

### **Step 4: Set OPENAI API KEY**
```bash
setx OPENAI_API_KEY "your_api_key_here"
```
Close out of the terminal, and open a new one.
Go to the SwiftGuard directory
### **Step 5: Running classifier**
```bash
conda activate swiftguard
python classify_jailbreak.py
```
Enter the prompt you want to classify as jailbroken or not. Model used is GPT-4o-mini.

## Current Progress
This repository currently has a few notebooks outlining progress made so far in the creation of SwiftGuard
- [Prompt Dataset](#prompt-dataset)
- [Prompt Exploration](#prompt-exploration)
- [Transforming](#transforming)
- [LLM Classifier](#llm-classifier)

## Prompt Dataset
This notebook used our PAIR algorithm to locate prompts that succcessfully jailbreak our target LLM. These prompts will be used throughout this repository to
show what to look out for in a malicious prompt.

## Prompt Exploration
This notebook contains basic EDA, exploring semantic and syntantic features of the above prompt dataset. 

## Transforming
This notebook contains functions useful to apply transformations on existing prompts via an LLM.

## LLM Classifier
This notebook contains a preliminary classifier which sends prompts down an LLM to weed out obviously benign ones. 
This classifier will determine which prompts get sent down the pipeline to further check its status. 


### NOTE: 
All jupyter notebooks can be accessed and ran locally with a togetherAI API key. To get more sucessfully jailbroken prompts, prompt dataset can be run again. Transforming and LLM Classifier both contain reproducable functions with built-in instructions. 
