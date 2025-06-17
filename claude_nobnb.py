#!/usr/bin/env python3
"""
Simple Fine-tune Llama 3.1 8B with LoRA on Alpaca dataset
Avoids bitsandbytes and triton dependencies entirely
Fixed tokenization and batching issues
"""

import json
import torch
import os
import sys

# Set environment variables early to avoid issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Import PEFT after setting environment variables
try:
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install: pip install peft datasets transformers accelerate")
    sys.exit(1)

# Configuration
MODEL_PATH = "/storage2/fs1/dt-summer-corp/Active/common/users/c.daedalus/llamas/llama3.1-8b-sagemaker-pretrained"
ALPACA_DATA_PATH = "/storage2/fs1/dt-summer-corp/Active/common/users/c.daedalus/instruction_tune/alpaca_data.json"
OUTPUT_DIR = "./llama-3.1-8b-alpaca-lora"
MAX_LENGTH = 512
# Tokenizer settings
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def load_alpaca_data(file_path):
    """Load and format Alpaca dataset"""
    print(f"Loading Alpaca data from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data:
        # Format: instruction + input (if exists) + output
        if item.get('input', '').strip():
            text = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}<|endoftext|>"
        else:
            text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}<|endoftext|>"
        
        formatted_data.append({"text": text})
    
    print(f"Loaded {len(formatted_data)} examples")
    return formatted_data

def tokenize_function(examples, tokenizer):
    """Tokenize the dataset with proper padding and truncation"""
    # Tokenize with padding and truncation
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,  # Enable padding
        max_length=MAX_LENGTH,
        return_overflowing_tokens=False,
        return_tensors=None,  # Keep as lists for now
    )
    
    # Validate token IDs are within vocabulary range
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # Ensure labels are the same as input_ids for causal LM
    tokenized["labels"] = []
    for input_ids in tokenized["input_ids"]:
        # Validate token IDs
        for token_id in input_ids:
            if token_id >= vocab_size:
                print(f"Warning: Token ID {token_id} >= vocab_size {vocab_size}")
                print(f"This will cause CUDA indexing errors!")
        
        # Create labels, replacing pad tokens with -100 (ignored in loss)
        labels = input_ids.copy()
        labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]
        tokenized["labels"].append(labels)
    
    return tokenized

def main():
    print("=== Llama 3.1 8B LoRA Fine-tuning ===")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU (will be very slow)")
    else:
        print(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Handle padding token properly for Llama
    if tokenizer.pad_token is None:
        # For Llama models, use eos_token as pad_token but don't add new tokens
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Set pad token to EOS token (ID: {tokenizer.eos_token_id})")
    
    # Alternative: If pad_token_id is still out of bounds, use a safe token
    if tokenizer.pad_token_id >= tokenizer.vocab_size:
        print(f"Pad token ID {tokenizer.pad_token_id} is out of bounds, using EOS token ID instead")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    
    # Critical: Ensure tokenizer vocabulary matches model
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    print(f"Pad token ID: {tokenizer.pad_token_id}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    
    # Test tokenization with a simple example
    test_text = "Hello world"
    test_tokens = tokenizer(test_text)
    max_token_id = max(test_tokens['input_ids'])
    print(f"Test tokenization - Max token ID: {max_token_id}")
    
    # Check if any token IDs are out of bounds
    if max_token_id >= tokenizer.vocab_size:
        print(f"ERROR: Token ID {max_token_id} >= vocab_size {tokenizer.vocab_size}")
        print("This will cause CUDA indexing errors!")
        return
    
    print("Tokenizer validation passed!")
    
    print("Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("Model loaded successfully")
        
        # Verify model and tokenizer compatibility
        model_vocab_size = model.get_input_embeddings().num_embeddings
        print(f"Model vocabulary size: {model_vocab_size}")
        
        if tokenizer.vocab_size != model_vocab_size:
            print(f"WARNING: Tokenizer vocab size ({tokenizer.vocab_size}) != Model vocab size ({model_vocab_size})")
            print("This mismatch will cause CUDA indexing errors!")
            
        # Resize token embeddings if needed
        if tokenizer.vocab_size > model_vocab_size:
            print("Resizing model token embeddings to match tokenizer...")
            model.resize_token_embeddings(tokenizer.vocab_size)
            print(f"Model embeddings resized to {model.get_input_embeddings().num_embeddings}")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    print("\nSetting up LoRA configuration...")
    
    # Simple LoRA configuration - avoid advanced features that might cause issues
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # Smaller rank to reduce memory usage
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj",
            "gate_proj",
            "down_proj",
        ],  # Fewer modules to avoid issues
        bias="none",
    )
    
    print("Applying LoRA to model...")
    try:
        model = get_peft_model(model, lora_config)
        print("LoRA applied successfully")
        model.print_trainable_parameters()
    except Exception as e:
        print(f"Error applying LoRA: {e}")
        print("This might be due to library compatibility issues.")
        return
    
    print("\nLoading and preprocessing dataset...")
    
    # Load Alpaca data
    try:
        alpaca_data = load_alpaca_data(ALPACA_DATA_PATH)
        dataset = Dataset.from_list(alpaca_data)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Take a smaller subset for testing if dataset is large
    if len(dataset) > 5000:
        dataset = dataset.select(range(5000))
        print(f"Using subset of 5000 examples for faster training")
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    try:
        tokenized_dataset = dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
            batch_size=100  # Process in smaller batches
        )
        print("Tokenization completed successfully")
    except Exception as e:
        print(f"Error tokenizing dataset: {e}")
        return
    
    # Split dataset (90% train, 10% eval)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Custom data collator for better handling
    class CustomDataCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
        
        def __call__(self, features):
            # Extract input_ids and labels
            input_ids = [torch.tensor(f["input_ids"]) for f in features]
            labels = [torch.tensor(f["labels"]) for f in features]
            attention_mask = [torch.tensor(f["attention_mask"]) for f in features]
            
            # Pad sequences to the same length
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=-100
            )
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                attention_mask, batch_first=True, padding_value=0
            )
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
    
    # Use custom data collator
    data_collator = CustomDataCollator(tokenizer)
    
    print("\nSetting up training arguments...")
    
    # Conservative training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,  # Start with 1 epoch for testing
        per_device_train_batch_size=1,  # Reduced batch size
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Effective batch size = 1 * 16 = 16
        warmup_steps=50,
        learning_rate=5e-5,  # Conservative learning rate
        fp16=torch.cuda.is_available(),  # Only use fp16 if CUDA available
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to=None,  # No logging to external services
        dataloader_num_workers=0,  # Disable multiprocessing for stability
    )
    
    print("Initializing trainer...")
    
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        print("Trainer initialized successfully")
    except Exception as e:
        print(f"Error initializing trainer: {e}")
        return
    
    print("\n=== Starting Training ===")
    
    try:
        trainer.train()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        print("You might need to reduce batch size or check GPU memory")
        return
    
    print("Saving model...")
    try:
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Model saved to {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error saving model: {e}")
        return
    
    # Test the fine-tuned model
    print("\n=== Testing Fine-tuned Model ===")
    test_inference(model, tokenizer)

def test_inference(model, tokenizer):
    """Test the fine-tuned model with a sample prompt"""
    test_prompt = "### Instruction:\nWhat is the capital of France?\n\n### Response:\n"
    
    print(f"Test prompt: {test_prompt}")
    
    try:
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        # Move inputs to same device as model if needed
        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\nModel response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
    except Exception as e:
        print(f"Error during inference test: {e}")

if __name__ == "__main__":
    # Final check for required files
    if not os.path.exists(ALPACA_DATA_PATH):
        print(f"Error: {ALPACA_DATA_PATH} not found!")
        print("Please ensure the alpaca_data.json file is in the current directory")
        sys.exit(1)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path {MODEL_PATH} not found!")
        print("Please update MODEL_PATH in the script")
        sys.exit(1)
    
    main()