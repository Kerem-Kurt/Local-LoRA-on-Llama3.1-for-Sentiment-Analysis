## Hugging Face Login
from tokenLogging import log_in

# Managin Dataset
from getDataset import get_dataset
from dividingDataset import divide_dataset

# Predict and Evaluate
import predictEvaluate

import wandb
import bitsandbytes as bnb
from datasets import Dataset
from trl import SFTTrainer
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          pipeline,
                          logging)

from peft import LoraConfig

# Install CUDA to your conda environment before downloading the requirements
# conda install nvidia/label/cuda-11.8.0::cuda
# Then
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

log_in()

df = get_dataset("takala/financial_phrasebank")

X_train, X_test, X_eval, y_true = divide_dataset(df,0.8,0.1)

# Convert to datasets
train_data = Dataset.from_pandas(X_train[["text"]])
eval_data = Dataset.from_pandas(X_eval[["text"]])


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


y_pred = predictEvaluate.predict(X_test, model, tokenizer)
predictEvaluate.evaluate(y_true, y_pred)


print("y_true (first 10 elements):", y_true[:10])

print("y_pred (first 10 elements):", y_pred[:10])


# Accuracy: 0.881
# Accuracy for label positive: 0.924
# Accuracy for label negative: 0.960
# Accuracy for label neutral: 0.846

# Classification Report:
#               precision    recall  f1-score   support

#     positive       0.76      0.92      0.84        66
#     negative       0.92      0.96      0.94        25
#      neutral       0.95      0.85      0.89       136

#     accuracy                           0.88       227
#    macro avg       0.88      0.91      0.89       227
# weighted avg       0.89      0.88      0.88       227


# Confusion Matrix:
# [[ 61   0   5]
#  [  0  24   1]
#  [ 19   2 115]]


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit 
    lora_module_names = set()
    
    for name, module in model.named_modules():
        if isinstance(module, cls): 
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    # If 'lm_head' is in the set, remove it (often needed for 16-bit models)
    if 'lm_head' in lora_module_names:  
        lora_module_names.remove('lm_head')
    
    return list(lora_module_names)

modules = find_all_linear_names(model)

print("Found all linear models")


output_dir="Llama-3.1-fine-tuned-model"

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules,  # Retaining the modules found using find_all_linear_names
)

training_arguments = TrainingArguments(
    output_dir=output_dir,                    # Directory to save the model
    num_train_epochs=1,                       # Number of training epochs
    per_device_train_batch_size=1,            # Batch size per device during training
    gradient_accumulation_steps=8,            # Number of steps before performing a backward/update pass
    gradient_checkpointing=True,              # Use gradient checkpointing to save memory
    optim="paged_adamw_32bit",                # Optimizer set for 32-bit AdamW (efficient memory usage)
    logging_steps=1,                         
    learning_rate=2e-4,                       # Learning rate, based on QLoRA paper (https://arxiv.org/abs/2305.14314 )
    weight_decay=0.001,
    fp16=True,                                # Using fp16 as the model was loaded in fp16 (float16) precision
    bf16=False,                               
    max_grad_norm=0.3,                        # Max gradient norm based on QLoRA paper
    max_steps=-1,
    warmup_ratio=0.03,                        # Warmup ratio based on QLoRA paper
    group_by_length=False,
    lr_scheduler_type="cosine",               # Cosine learning rate scheduler
    report_to="wandb",                        # Report metrics to W&B
    eval_strategy="steps",                    # Save checkpoint every few steps rather than epochs
    eval_steps=int(0.2 * len(train_data)),    # Eval steps based on a portion of the training data
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    dataset_text_field="text",               # The field name in the dataset containing the text
    tokenizer=tokenizer,
    max_seq_length=512,                      # Max sequence length for padding/truncation
    packing=False,                           # No packing of multiple examples into one input sequence
    dataset_kwargs={
        "add_special_tokens": False,         # Do not add special tokens, based on your previous settings
        "append_concat_token": False,        # Do not append concatenation token
    }
)

print("Training Arguments done")
tokenizer.pad_token = tokenizer.eos_token


trainer.train()

print("Trained")

wandb.finish()
model.config.use_cache = True
print("Wandb finished")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("Saved the fine tuned model")

y_pred = predictEvaluate.predict(X_test, model, tokenizer)
predictEvaluate.evaluate(y_true, y_pred)