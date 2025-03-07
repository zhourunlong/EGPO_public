import torch

from peft import LoraConfig, TaskType

server_address = "http://10.158.48.114:5000/v1/classifications"

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=256,
    lora_alpha=512,
    lora_dropout=0.1,
)

def get_ds_config(batch_size,
                  micro_batch_size,
                  gradient_accumulation_steps,
                  learning_rate):
    return {
        "train_batch_size": batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW"
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_type": "linear",
                "warmup_max_lr": learning_rate,
                "total_num_steps": "auto",
            }
        },
        "zero_optimization": {
            "stage": 0
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 10000000000,
        "wall_clock_breakdown": False,
        "zero_allow_untested_optimizer": True
    }
