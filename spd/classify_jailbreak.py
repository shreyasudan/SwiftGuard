print("🚀 Script started. Waiting for input...")


import os
import sys
import pickle
import numpy as np
import argparse
from scipy.special import softmax

# Ensure Python can find `spd/` (where SPD.main_spd is located)
sys.path.append(os.path.abspath("spd"))

# Import calculate_logits from SPD.main_spd
from SPD.main_spd import calculate_logits
from sklearn.svm import SVC


with open("models/svm_model.pkl", "wb") as f:
    pickle.dump(clf, f)


# Constants
MODEL_NAME = "gpt-4o-mini"
WRITE_PATH = "logits/temp"  # Temporary path to store generated logits
R = 25  # Number of token places to consider
K = 5   # Number of candidate logit values


def get_logits_from_prompt(prompt):
    """Generate logits for a given prompt using calculate_logits."""
    all_prompts = {MODEL_NAME: {0: {"prompt": prompt}}}  # Format required by calculate_logits
    logits = calculate_logits(WRITE_PATH, all_prompts, llm_provider="litellm")

    # Process logits: Apply softmax and take negative log probabilities
    probabilities = softmax(logits, axis=2)
    values = -np.log(probabilities)

    # Reshape to match training format
    values = values[:, :R, :K].reshape(1, R * K)  # Shape: (1, R*K)

    return values


def classify_prompt(prompt):
    """Classify the prompt as Jailbroken (1) or Benign (0)."""
    logits = get_logits_from_prompt(prompt)
    prediction = clf.predict(logits)

    if prediction[0] == 1:
        print("🚨 The prompt is classified as **JAILBROKEN**.")
    else:
        print("✅ The prompt is classified as **BENIGN**.")



if __name__ == "__main__":
    print("🔍 Jailbreak Classifier Running...")
    print("Type a prompt to classify or type 'exit' to quit.\n")

    while True:
        user_prompt = input("Enter prompt: ")

        if user_prompt.lower() == "exit":
            print("👋 Exiting...")
            break

        classify_prompt(user_prompt)