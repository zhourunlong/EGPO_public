import json
import os
import pdb
import time
import torch

from trl import NashMDTrainer
from trl.models.modeling_base import GeometricMixtureWrapper
from trl.models.utils import unwrap_model_for_generation

class NashMDPGTrainer(NashMDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _generate_completions(self, model, prompts):
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            model_output = unwrapped_model.generate(
                input_ids=prompts["input_ids"],
                attention_mask=prompts["attention_mask"],
                generation_config=self.generation_config,
            )

            ref_model = model if self.ref_model is None else self.ref_model
            with torch.no_grad(), \
                 unwrap_model_for_generation(
                     ref_model,
                     self.accelerator,
                     is_peft_model = self.ref_model is None
                 ) as unwrapped_ref_model:
                mixture_model = GeometricMixtureWrapper(
                    model=unwrapped_model,
                    ref_model=unwrapped_ref_model,
                    generation_config=self.generation_config,
                    mixture_coef=self.mixture_coef,
                    device=self.accelerator.device,
                )

                mixture_output = mixture_model.generate(
                    input_ids=prompts["input_ids"],
                    attention_mask=prompts["attention_mask"],
                    generation_config=self.generation_config,
                )

        return model_output, mixture_output