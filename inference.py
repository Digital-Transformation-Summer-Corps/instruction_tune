from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel

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

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Model with LoRA adapters loaded successfully! Type 'end' to quit.")
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
        inputs = tokenizer(prompt, return_tensors="pt")
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