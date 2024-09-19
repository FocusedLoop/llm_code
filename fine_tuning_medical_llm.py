import torch
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline)

# Loading the model
llama_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path = "aboonaji/llama2finetune-v2",
                                                   quantization_config = BitsAndBytesConfig(load_in_4bit = True,
                                                                                            bnb_4bit_compute_dtype = getattr(torch, "float16"),
                                                                                            bnb_4bit_quant_type = "nf4"))

llama_model.config.use_cache = False
llama_model.config.pretraining_tp = 1

# Loading the tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = "aboonaji/llama2finetune-v2", trust_remote_code = True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

# Setting the training arguments
traing_arguments = TrainingArguments(output_dir = "./results", per_device_train_batch_size = 4, max_steps = 100)

# Creating the supervised fine-tunning trainer
llama_sft_trainer = SFTTrainer(model = llama_model,
                               args = traing_arguments,
                               train_dataset = load_dataset(path = "aboonaji/wiki_medical_terms_llam2_format", split = "train"),
                               tokenizer = llama_tokenizer,
                               peft_config = LoraConfig(task_type = "CAUSAL_LM", r = 64, lora_alpha = 16, lora_dropout = 0.1),
                               dataset_text_field = "text")

# Training the model
llama_sft_trainer.train()

# Chatting with the model
user_prompt = "please tell me about epilepsy" # Input prompt here
text_generation_pipeline = pipeline(task = "text-generation", model = llama_model, tokenizer = llama_tokenizer, max_length = 300)
model_answer = text_generation_pipeline(f"<s>[INST] {user_prompt} [/INST]")
print(model_answer[0]['generated_text'])