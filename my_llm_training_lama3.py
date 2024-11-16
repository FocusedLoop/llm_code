from transformers import AutoModelForCausalLM, TrainingArguments, AutoTokenizer, Trainer, BitsAndBytesConfig, pipeline
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

model_id = r"C:\Users\Joshua\Desktop\python_ai\llm_code\llama31\Llama-3.1-8B-Instruct"
dataset = ""

# Loading model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
) # INSPECT MORE

llm_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_id,
    quantization_config=quantization_config
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Training arguments
strat = "steps"
step_amount = 100
training_args = TrainingArguments(output_dir="./test_trainer")
training_args.set_dataloader(train_batch_size=8, eval_batch_size=32)
training_args.set_evaluate(strategy=strat, steps=step_amount)
training_args.set_logging(strategy=strat, steps=step_amount)

# Fine tunning
# Optimize training
peft_config = LoraConfig(
    task_type = "CAUSAL_LM", 
    r = 64, lora_alpha = 16, 
    lora_dropout = 0.1
)
llm_model = get_peft_model(llm_model, peft_config)

# Start Training with dataset
finetune_dataset = load_dataset(path=dataset, split="train") # GO OVER

trainer = Trainer(
    model=llm_model,
    args=training_args,
    train_dataset=finetune_dataset,
    tokenizer=tokenizer
)

# Pipeline for taking in inputs
pipeline = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)


# Prompting
while True:
    user_prompt = input("What's your question?\n")
    if user_prompt == "exit":
        break
    messages = [
    {"role": "system", "content": (
            "You are a highly knowledgeable mathematician and engineering chatbot. "
            "Your purpose is to assist users by providing detailed, scientifically accurate explanations of mathematical and engineering concepts. "
            "Use formal scientific language and precise terminology, ensuring your responses are comprehensive and thorough. "
            "Reference relevant theories, equations, and examples as needed, but keep your explanations strictly text-based. "
            "Organize your answers clearly, and aim to enhance understanding by elucidating complex concepts without oversimplifying them."
        )},
    {"role": "user", "content": user_prompt}]

    outputs = pipeline(messages,max_new_tokens=256)
    responce = outputs[0]["generated_text"]
    print(responce)

print("Exiting...")