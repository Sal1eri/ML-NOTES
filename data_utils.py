# Function to process different dataset formats
def process_qa_dataset(examples, question_col, answer_col):
    """Process Q&A datasets into chat format"""
    processed = []
    
    for question, answer in zip(examples[question_col], examples[answer_col]):
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        processed.append(messages)
    
    return {"messages": processed}

def process_instruction_dataset(examples):
    """Process instruction-following datasets"""
    processed = []
    
    for instruction, response in zip(examples["instruction"], examples["response"]):
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]
        processed.append(messages)
    
    return {"messages": processed}

# Example: Process GSM8K math dataset
print("=== PROCESSING GSM8K DATASET ===\n")

gsm8k = load_dataset("openai/gsm8k", "main", split="train[:100]")  # Small subset for demo
print(f"Original GSM8K example: {gsm8k[0]}")

# Convert to chat format
def process_gsm8k(examples):
    processed = []
    for question, answer in zip(examples["question"], examples["answer"]):
        messages = [
            {"role": "system", "content": "You are a math tutor. Solve problems step by step."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        processed.append(messages)
    return {"messages": processed}

gsm8k_processed = gsm8k.map(process_gsm8k, batched=True, remove_columns=gsm8k.column_names)
print(f"Processed example: {gsm8k_processed[0]}")