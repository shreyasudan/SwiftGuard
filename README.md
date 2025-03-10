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
