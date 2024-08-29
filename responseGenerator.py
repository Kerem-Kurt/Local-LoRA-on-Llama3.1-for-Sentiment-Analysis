import bitsandbytes as bnb
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig)
import torch

#Log in via huggingface-cli

base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Set up 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

# Load the model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    torch_dtype="float16",
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)


def generate_text(prompt, max_length=50):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate the output using the model
    outputs = model.generate(**inputs, max_new_tokens=max_length)

    # Decode the output tokens to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

while True:
    prompt = input()
    if prompt == "EXIT":
        break
    output_text = generate_text(prompt)
    print(output_text)

