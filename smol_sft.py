from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
imoport torch
# Initialize experiment tracking

# Load SmolLM3 base model
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM3-3B-Base",device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B-Base")

# Load SmolTalk2 dataset
dataset = load_dataset("HuggingFaceTB/smoltalk2_everyday_convs_think")

# Configure training with Trackio integration
config = SFTConfig(
    output_dir="./smollm3-finetuned",
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    max_steps=1000,
    # report_to="trackio",  # Enable Trackio logging
)

# Train!
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=config,
)
trainer.train()