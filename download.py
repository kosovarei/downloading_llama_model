from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer from Hugging Face
model_name = "meta-llama/CodeLlama-13b-Instruct-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Specify the path where you want to save the model and tokenizer
save_path = "./"

# Save the model and tokenizer to the specified path
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# Print a message indicating where the model and tokenizer were saved
print(f"Model and tokenizer saved to {save_path}")
