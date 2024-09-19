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
traing_arguments = TrainingArguments(output_dir = "llms/results", per_device_train_batch_size = 4, max_steps = 100)

# Creating the supervised fine-tunning trainer
llama_sft_trainer = SFTTrainer(model = llama_model,
                               args = traing_arguments,
                               train_dataset = load_dataset(path = "aboonaji/wiki_medical_terms_llam2_format", split = "train"),
                               tokenizer = llama_tokenizer,
                               peft_config = LoraConfig(task_type = "CAUSAL_LM", r = 64, lora_alpha = 16, lora_dropout = 0.1),
                               dataset_text_field = "text")

# Training the model
llama_sft_trainer.train()

# Saving the model
torch.save(llama_model.state_dict(), "llms/trained_medical_llama_model_state_dict.pth")
llama_tokenizer.save_pretrained("llms/trained_medical_llama_tokenizer")
