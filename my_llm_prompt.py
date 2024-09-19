import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, pipeline)

# Load the tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained("llms/trained_medical_llama_tokenizer")

# Load the model
llama_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="aboonaji/llama2finetune-v2",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=getattr(torch, "float16"),
        bnb_4bit_quant_type="nf4"
    )
)

# Load the saved state_dict
state_dict = torch.load("llms/trained_medical_llama_model_state_dict.pth", weights_only=True)

# Filter out unexpected keys
filtered_state_dict = {k: v for k, v in state_dict.items() if k in llama_model.state_dict()}

# Load the filtered state_dict into the model
llama_model.load_state_dict(filtered_state_dict, strict=False)

# Chatting with the model
max_text_length = 300
text_generation_pipeline = pipeline(task = "text-generation", model = llama_model, tokenizer = llama_tokenizer, max_length = max_text_length)

# Generation configuration
generation_config = GenerationConfig.from_pretrained("aboonaji/llama2finetune-v2")
generation_config.max_length = max_text_length

while True:
    user_prompt = input("Enter a prompt: ")
    if user_prompt.lower() == "exit":
        break
    
    responce = text_generation_pipeline(f"<s>[INST] {user_prompt} [/INST]", generation_config=generation_config)
    model_answer = responce[0]['generated_text']
    print(model_answer)

print("Exiting chat.")