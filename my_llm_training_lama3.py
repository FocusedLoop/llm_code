from transformers import AutoModelForCausalLM, TrainingArguments, AutoTokenizer, BitsAndBytesConfig, pipeline
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

model_id = r"C:\Users\Joshua\Desktop\python_ai\llm_code\llama31\Llama-3.1-8B-Instruct"
dataset = "Shekswess/llama3_medical_meadow_wikidoc_instruct_dataset"

# Loading model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, #float16
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
training_arguments = TrainingArguments(
    output_dir="./test_trainer",
    num_train_epochs=4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# Fine tunning
# Optimize training
peft_config = LoraConfig(
    task_type = "CAUSAL_LM", 
    r = 64, #64
    lora_alpha = 16, #16
    lora_dropout = 0.1 #0.1
)
#llm_model = get_peft_model(llm_model, peft_config)

# Start Training with dataset
finetune_dataset = load_dataset(path=dataset, split="train") # GO OVER

def formatting_func(example):
    output_texts = []
    for i in range(len(example['input'])):
        text = f"### Input: {example['input'][i]}\n ### Output: {example['output'][i]}\n ### Instruction: {example['instruction'][i]}\n ### Prompt: {example['prompt'][i]}"
        output_texts.append(text)
    return output_texts


llm_model_trainer = SFTTrainer(
    model=llm_model,
    args=training_arguments,
    train_dataset=load_dataset(path = "Shekswess/llama3_medical_meadow_wikidoc_instruct_dataset", split = "train"),
    tokenizer=tokenizer,
    peft_config = peft_config,
    formatting_func=formatting_func,
)

# Start training
latest_checkpoint = "./test_trainer/checkpoint-10000"

if latest_checkpoint:
    print(f"Resuming training from checkpoint: {latest_checkpoint}")
    llm_model_trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    print("Starting training from scratch")
    llm_model_trainer.train()

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
    # messages = [
    # {"role": "system", "content": (
    #     "You are a highly knowledgeable mathematician and engineering chatbot. "
    #         "Your purpose is to assist users by providing detailed, scientifically accurate explanations of mathematical and engineering concepts. "
    #         "Use formal scientific language and precise terminology, ensuring your responses are comprehensive and thorough. "
    #         "Reference relevant theories, equations, and examples as needed, but keep your explanations strictly text-based. "
    #         "Organize your answers clearly, and aim to enhance understanding by elucidating complex concepts without oversimplifying them."
    #     )},
    # {"role": "user", "content": user_prompt}]

    outputs = pipeline(user_prompt,max_new_tokens=256)
    responce = outputs[0]["generated_text"]
    print(responce)

print("Exiting...")