# INFERENCE FROM SUMMARY

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
import argparse
import os

# LIST OF SUMMARIES
summaries = {
    "Compute1 Quickstart": "This guide provides a hands-on overview for users accessing the Compute1 HPC platform. It covers connecting via SSH and VPN, using the Open OnDemand (OOD) web portal for launching applications (like Jupyter or MATLAB), and navigating job sessions. It also explains how to use the RIS Desktop environment via THPC modules, customize Docker containers with the noVNC image, and optionally connect through a local VNC viewer. It includes practical instructions on configuring storage paths, setting job parameters (like CPUs, memory, GPUs), and launching interactive sessions within the OOD interface.",
    "Compute2 Quickstart": "This document serves as a practical starting guide for the newer Compute2 platform, which uses Slurm instead of LSF. It walks users through SSH access, VPN setup, and command-line interaction. It introduces the Slurm job scheduler with example usage of srun, sbatch, squeue, and scancel, including options for requesting GPUs, memory, and using Docker containers via Pyxis. It also includes guidance on job arrays, GPU testing with PyTorch, and container storage mounting. The content is oriented toward command-line workflows and assumes moderate familiarity with HPC environments.",
    "Docker Basics: Building, Tagging, & Pushing A Custom Docker Image": "This step-by-step tutorial walks through building a custom Docker image with Python, R, and Git. It explains best practices like choosing a minimal base image, installing dependencies with apt-get, importing code via Git or COPY, and tagging images. It provides command examples for building, running, and pushing Docker containers. The tutorial emphasizes testing locally before deploying on Compute1, and ends with instructions on running Docker containers via the LSF job scheduler.",
}

# Concatenate summaries into a single string
summaries_str = "\n".join([f"Title: {title} \n Summary:{summary}\n\n" for title, summary in summaries.items()])

# SYSTEM PROMPT
CONTEXT_RETRIEVAL_TEMPLATE = f"What is the title of the summary that is most relevant to the query '{{question}}' in the following summaries: {summaries_str}. It is incredibly important that you only return the title of a single summary, not the summary itself, or multiple titles."

SYSTEM_PROMPT_TEMPLATE = f"You are a helpful assistant that can answer questions about the following context: {{context}}. Please answer the question: {{question}}."

# Parse command line arguments
# parser = argparse.ArgumentParser(description="Interactive LLaMA inference with optional LoRA adapters")
# parser.add_argument("--use-lora", action="store_true", help="Use LoRA fine-tuned adapters")
# args = parser.parse_args()

# Model paths
base_model_path = "/storage2/fs1/dt-summer-corp/Active/common/users/c.daedalus/llamas/llama-3.1-8b-instruct"  # Base model
# adapter_path = "/storage2/fs1/dt-summer-corp/Active/common/users/c.daedalus/llamas/llama3.18b-sagemaker-pretrained-alpaca-ft"  # LoRA adapters

print("Loading tokenizer from base model...")
if os.path.exists(base_model_path) and len(os.listdir(base_model_path)) > 0: 
    print("Loading tokenizer from local path...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
else:
    print("Downloading tokenizer from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

print("Loading base model...")
if os.path.exists(base_model_path) and len(os.listdir(base_model_path)) > 0: 
    print("Loading base model from local path...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,  # Use half precision to save memory
        device_map="auto",  # Automatically distribute across GPUs if available
        trust_remote_code=True
    )
else:
    print("Downloading model from HuggingFace...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16,  # Use half precision to save memory
        device_map="auto",  # Automatically distribute across GPUs if available
        trust_remote_code=True
    )
    
    # Save the model and tokenizer to the local path for future use
    print(f"Saving model to {base_model_path} for future use...")
    os.makedirs(base_model_path, exist_ok=True)
    base_model.save_pretrained(base_model_path)
    tokenizer.save_pretrained(base_model_path)
    print("Model saved successfully!")

# if args.use_lora:
#     print("Loading LoRA adapters...")
#     model = PeftModel.from_pretrained(base_model, adapter_path)
#     print("Model with LoRA adapters loaded successfully! Type 'end' to quit.")
# else:
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
                max_length=1000,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=tokenized_context_retrieval_prompt.get("attention_mask")
            )
        
        # Decode and print system response
        response = tokenizer.decode(system_outputs[0], skip_special_tokens=True)
        # Remove the original prompt from the response
        response = response[len(context_retrieval_prompt):].strip()
        print(f"Model response to context retrieval prompt: {response}\n")
        print(f"\nRetrieving the following pages:")
        # titles = response.split(",")
        # titles = [title.strip() for title in titles]
        title = response.strip()
        # for title in titles:
        #     print(f"- {title}")
        print(f"- {title}")
        print("-" * 50)

        context = "\n"
        # for title in titles:
        #     with open(os.path.join("RIS_docs/", title), "r") as f:
        #         context += f"{f.read()}\n\n"
        with open(os.path.join("RIS_docs/", title), "r") as f:
            context += f"{f.read()}\n\n"

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context, question=prompt)

        inputs = tokenizer(system_prompt, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=3500,
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