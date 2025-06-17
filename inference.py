from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer and model
model_path = "D:\dt-summer-corp\llamas\llama3.1-8b-sagemaker-pretrained"  # Replace with your actual path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Use half precision to save memory
    device_map="auto",  # Automatically distribute across GPUs if available
    trust_remote_code=True
)

# Example inference
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")
inputs.to("cuda")
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_length=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    outputs.to("cpu")
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)