import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.image_utils import load_image
from datasets import features, load_dataset
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import argparse
from accelerate import PartialState


parser = argparse.ArgumentParser(description='Fine-tune SmolVLM model')
parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model')
parser.add_argument('--data_path', type=str, required=True, help='Path to the training data')
parser.add_argument('--model_checkpoint_path', type=str, required=True, help='Path to save the fine-tuned model')
parser.add_argument('--num_epochs', type=int, required=True, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate for training')

args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "CPU"

device_string = PartialState().process_index
print(device_string)

processor = AutoProcessor.from_pretrained(args.model_path)
model = AutoModelForVision2Seq.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16, device_map={'':device_string}
)
DEVICE = model.device

from datasets import Dataset, Sequence, Image as HFImageFeature
from PIL import Image
# import wandb
# wandb.login()
# wandb.init(
# project = "CS776-project",
# name = "SmolVLM-256M-massive-smolcot-7k"
# )

dataset = Dataset.load_from_disk(args.data_path)
print(f"size of data: {len(dataset)}")
split_dataset = dataset.train_test_split(test_size = 0.02)
train_data = split_dataset["train"]
eval_data = split_dataset["test"]

# sft trainer from huggingface
trainer = SFTTrainer(
    model = model,
    tokenizer = processor,
    data_collator = UnslothVisionDataCollator(model, processor), # Must use!
    train_dataset = train_data,
    eval_dataset = eval_data,
    # peft_config = lora_config,
    args = SFTConfig(
        # checkpointing
        use_liger_kernel=True,
        packing=False,
        save_strategy="steps", 
        save_steps=50, 
        save_total_limit=5, 
        load_best_model_at_end=True, g
        # If you want to evaluate and save based on the best model:
        eval_strategy="steps", 
        eval_steps=10, 
        metric_for_best_model="eval_loss", # Metric to use for determining the best model
        per_device_train_batch_size = 1, #1
        gradient_accumulation_steps = 32, #32
        warmup_steps = 5,
        # max_steps = 100,
        num_train_epochs = args.num_epochs, 
        learning_rate = args.learning_rate, 
        #fp16 = True,
        fp16 = False,
        bf16 = is_bf16_supported(),
        # bf16 = True,
        logging_steps = 1,
        optim = "paged_adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "log_outputs",
        # report_to = "wandb", # For Weights and Biases
        
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
        # gradient checkpointing
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs={"use_reentrant":False},
    ),
)

# better logging
trainer.args.save_strategy = "steps"
trainer.args.save_steps = 80 
trainer.args.load_best_model_at_end = True
trainer.args.metric_for_best_model = "loss"

trainer_stats = trainer.train()
#wandb.finish()

trainer.save_model(args.model_checkpoint_path)


