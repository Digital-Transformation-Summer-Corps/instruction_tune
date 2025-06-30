# INFERENCE FROM SUMMARY

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
import argparse
import os

# LIST OF SUMMARIES
summaries = {
    "1": "The first summary",
}

# Concatenate summaries into a single string
summaries_str = "\n".join([f"{title}: \n{summary}\n\n" for title, summary in enumerate(summaries)])

# SYSTEM PROMPT
CONTEXT_RETRIEVAL_TEMPLATE = f"Based on this list of summaries: {summaries_str}, list the titles of the summaries most relevant to the question: {{question}}. Do not list more than 3 summaries. Please list the titles in a comma separated list, e.g., 'title1, title2, title3'."

SYSTEM_PROMPT_TEMPLATE = f"You are a helpful assistant that can answer questions about the following context: {{context}}. Please answer the question: {{question}}."

# Parse command line arguments
parser = argparse.ArgumentParser(description="Interactive LLaMA inference with optional LoRA adapters")
parser.add_argument("--use-lora", action="store_true", help="Use LoRA fine-tuned adapters")
args = parser.parse_args()

# Model paths
base_model_path = "/storage2/fs1/dt-summer-corp/Active/common/users/c.daedalus/llamas/llama3.1-8b-sagemaker-pretrained"  # Base model
adapter_path = "/storage2/fs1/dt-summer-corp/Active/common/users/c.daedalus/llamas/llama3.18b-sagemaker-pretrained-alpaca-ft"  # LoRA adapters

print("Loading tokenizer from base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,  # Use half precision to save memory
    device_map="auto",  # Automatically distribute across GPUs if available
    trust_remote_code=True
)

if args.use_lora:
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("Model with LoRA adapters loaded successfully! Type 'end' to quit.")
else:
    model = base_model
    print("Using base model (no LoRA adapters). Type 'end' to quit.")

print("-" * 50)

# Interactive loop
while True:
    # Get user input
    prompt = input("\nYour question: ").strip()
    
    # Check if user wants to end the conversation
    if prompt.lower() == "end":
        print("Goodbye!")
        break
    
    # Skip empty inputs
    if not prompt:
        print("Please enter a question or type 'end' to quit.")
        continue
    
    # Generate response
    try:
        # Retrieve the most relevant summaries
        context_retrieval_prompt = CONTEXT_RETRIEVAL_TEMPLATE.format(question=prompt)
        tokenized_context_retrieval_prompt = tokenizer(context_retrieval_prompt, return_tensors="pt")
        tokenized_context_retrieval_prompt = {k: v.to("cuda") for k, v in tokenized_context_retrieval_prompt.items()}

        with torch.no_grad():
            system_outputs = model.generate(
                tokenized_context_retrieval_prompt["input_ids"],
                max_length=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=tokenized_context_retrieval_prompt.get("attention_mask")
            )
        
        # Decode and print system response
        response = tokenizer.decode(system_outputs[0], skip_special_tokens=True)
        # Remove the original prompt from the response
        response = response[len(context_retrieval_prompt):].strip()
        print(f"\nRetrieving the following pages:")
        titles = response.split(",")
        titles = [title.strip() for title in titles]
        for title in titles:
            print(f"- {title}")
        print("-" * 50)

        context = "\n"
        for title in titles:
            with open(os.path.join("RIS_docs/", title), "r") as f:
                context += f"{f.read()}\n\n"

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context, question=prompt)

        inputs = tokenizer(system_prompt, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=inputs.get("attention_mask")
            )
        
        # Decode and print response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt from the response
        response = response[len(prompt):].strip()
        
        print(f"\nResponse: {response}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error generating response: {e}")
        print("Please try again or type 'end' to quit.")