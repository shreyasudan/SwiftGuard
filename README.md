# SwiftGuard
##### *Emphasizing both speed and protection* 
SwiftGuard is an algorithm designed to protect LLMs from malicious prompts attempting to 'jailbreak' the system. 
With the implemenation of SwiftGuard, LLMs can be protected from being tricked into responding with illegal or harmful information, at little
to no cost to the typical user. 

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
This notebook contains a preliminary classifier which sends prompts down an LLM to weed out immediately benign ones. 
This classifier will determine which prompts get sent down the pipeline to further check its status. 
