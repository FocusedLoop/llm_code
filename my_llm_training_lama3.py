import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "unsloth/llama-3-8b-bnb-4bit"

# Load the pretrained model using to_empty to handle meta tensors
llama_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
    device_map="auto",  # Automatically map model layers to available devices
    quantization_config=BitsAndBytesConfig(  # Ensure the model is loaded in 4-bit precision
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    ),
    ignore_mismatched_sizes=True,
)

# Use to_empty() before moving the model to the GPU
llama_model.to_empty()

# Now move the model to the GPU
llama_model = llama_model.to("cuda")

# Load the tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(model_id)

# Create the text generation pipeline
text_generation_pipeline = transformers.pipeline(
    "text-generation",
    model=llama_model,
    tokenizer=llama_tokenizer,
    device=0,  # Use GPU (device 0)
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

# Prepare the prompt using the tokenizer's template
prompt = llama_tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

terminators = [
    llama_tokenizer.eos_token_id,
    llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = text_generation_pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])

# https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing#scrollTo=2eSvM9zX_2d3