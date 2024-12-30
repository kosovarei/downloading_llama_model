from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi

# Path to the current folder
save_path = "./"  # The folder where the model was saved

# Load the model and tokenizer from the same folder
model = AutoModelForCausalLM.from_pretrained(save_path)
tokenizer = AutoTokenizer.from_pretrained(save_path)

# Test the model by generating text
prompt = "Create a Terraform configuration to deploy an AWS EC2 instance."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=150, num_return_sequences=1)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
