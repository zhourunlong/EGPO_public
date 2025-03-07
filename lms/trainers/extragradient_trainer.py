import jinja2
import os
import pdb
import textwrap
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from datasets import Dataset, IterableDataset
from transformers import (
    BaseImageProcessor,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import OptimizerNames
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.models.modeling_base import GeometricMixtureWrapper
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.judges import BasePairwiseJudge
from trl.trainer.online_dpo_trainer import OnlineDPOTrainer
from trl.trainer.utils import (
    SIMPLE_CHAT_TEMPLATE,
    empty_cache,
    generate_model_card,
    get_comet_experiment_url,
    get_reward,
    first_true_indices,
)
from typing import Any, Callable, Optional, Union

from trainers.extragradient_config import ExtragradientConfig
from utils import mylog, multiply_list

def can_reduce_variance(args):
    return args.y_yp_mixture_coef == args.ypp_mixture_coef \
        and args.y_yp_temperature == 1.0 \
        and args.y_yp_top_k == 0 \
        and args.y_yp_top_p == 1.0 \
        and args.y_yp_min_p == 0.0

class ExtragradientTrainer(OnlineDPOTrainer):
    r"""
    Initialize ExtragradientTrainer as a subclass of [`OnlineDPOTrainer`].

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForCausalLM`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        reward_model (`transformers.PreTrainedModel`):
            The reward model to score completions with, preferably an `AutoModelForSequenceClassification`.
        judge (`BasePairwiseJudge`):
            The judge to use for pairwise comparison of model completions.
        args (`ExtragradientConfig`):
            The Extragradient config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        processing_class (`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, *optional*):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.
        peft_config (`dict`):
            The peft config to use for training.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
        callbacks (`list[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
    """

    _tag_names = ["trl", "extra-gradient"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        reward_model: Union[PreTrainedModel, nn.Module, None] = None,
        judge: Optional[BasePairwiseJudge] = None,
        args: Optional[ExtragradientConfig] = None,
        data_collator: Optional[Callable] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, \
                  FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        peft_config: Optional[dict] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer,
                          torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor,
                                                          torch.Tensor],
                                                         torch.Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            ref_model=ref_model,
            reward_model=reward_model,
            judge=judge,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_class=processing_class,
            # for now, Extragradient Trainer can't use any reward model
            peft_config=peft_config,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        self.accumulated_steps = 0

        self.stats = {
            "kl": [],
            "prefs/chosen": [],
            "prefs/rejected": [],
            "prefs/margins": [],
            "rewards/chosen": [],
            "rewards/rejected": [],
            "rewards/margins": [],
            "rewards/accuracies": [],
            "logps/chosen": [],
            "logps/rejected": [],
            "contain_eos_token": [],
            "gen_len": [],
            "beta": [],
        }

        # override the generation config
        if self.args.max_seq_length != -1:
            self.generation_config = GenerationConfig(
                max_length=self.args.max_seq_length,
                do_sample=True,
                use_cache=False if self.args.gradient_checkpointing else True,
            )
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.args.max_new_tokens,
                do_sample=True,
                use_cache=False if self.args.gradient_checkpointing else True,
            )

    def _generate_completions(
        self,
        model,
        prompts,
        num_samples=1,
        mixture_coef=0,
        **kwargs
    ):
        """
        Args:
            - model: model to generate completions
            - prompts: dict of tensors of shape (batch_size, input_len)
            - override_top_k: top_k to use for generation
            - override_temperature: temperature to use for generation
            - num_samples: number of completions to generate
            - mixture_coef: mixture coefficient for geometric mixture model
                (1 - mixture_coef) * log (\pi_{\\theta}) + mixture_coef * log (\pi_{\\theta{ref}})
        
        Returns:
            - outputs: shape (batch_size, num_samples, seq_len)
        """
        if mixture_coef < 0 or mixture_coef > 1:
            raise ValueError("mixture_coef must be in [0, 1]")

        with torch.no_grad():
            ref_model = model if self.ref_model is None else self.ref_model
            if mixture_coef == 0:
                with unwrap_model_for_generation(
                    model,
                    self.accelerator
                ) as unwrapped_model:
                    outputs = unwrapped_model.generate(
                        input_ids=prompts["input_ids"],
                        attention_mask=prompts["attention_mask"],
                        generation_config=self.generation_config,
                        num_return_sequences=num_samples,
                        **kwargs
                    )
            elif mixture_coef == 1:
                with unwrap_model_for_generation(
                    ref_model,
                    self.accelerator,
                    is_peft_model=self.ref_model is None
                ) as unwrapped_ref_model:
                    outputs = unwrapped_ref_model.generate(
                        input_ids=prompts["input_ids"],
                        attention_mask=prompts["attention_mask"],
                        generation_config=self.generation_config,
                        num_return_sequences=num_samples,
                        **kwargs
                    )
            else:
                with unwrap_model_for_generation(
                    model,
                    self.accelerator
                ) as unwrapped_model, \
                unwrap_model_for_generation(
                    ref_model,
                    self.accelerator,
                    is_peft_model=self.ref_model is None
                )  as unwrapped_ref_model:
                    mixture_model = GeometricMixtureWrapper(
                        model=unwrapped_model,
                        ref_model=unwrapped_ref_model,
                        generation_config=self.generation_config,
                        mixture_coef=mixture_coef,
                        device=self.accelerator.device,
                    )
                    outputs = mixture_model.generate(
                        input_ids=prompts["input_ids"],
                        attention_mask=prompts["attention_mask"],
                        generation_config=self.generation_config,
                        num_return_sequences=num_samples,
                        **kwargs
                    )

        batch_size = prompts["input_ids"].shape[0]
        outputs = outputs.view(batch_size, num_samples, -1)
        
        return outputs

    def _process_completions(self, outputs, prompts):
        """
        Truncate by first EOS token and pad to the same length.

        Args:
            - outputs: shape (batch_size, num_samples, seq_len)
            - prompts: dict of tensors of shape (batch_size, input_len)
        
        Returns:
            - data: dict of tensors of shape (batch_size, num_samples, seq_len)
                `chunk_pos` is a tensor of shape (num_chunks,)
        """
        def truncate_right(
            input_ids: torch.Tensor, stop_token_id: int, pad_token_id: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Truncates the input tensor from the right side after the first 
            occurrence of the stop token.

            Args:
                input_ids: shape (batch_size, num_samples, seq_len)
                    The tensor containing the responses to be truncated
                stop_token_id (`int`):
                    The token ID representing the stop token where truncation 
                    occurs
                pad_token_id (`int`):
                    The token ID representing the pad token used to fill the 
                    truncated responses

            Returns:
                tuple:
                    - `output_ids`: shape (batch_size, num_samples, seq_len')
                        The truncated responses tensor with pad tokens filled 
                        after the stop token
                    - `mask`: shape (batch_size, num_samples, seq_len')
                        The mask tensor to indicate the padding tokens
            """
            trunc_idxs = first_true_indices(
                input_ids == stop_token_id
            ).unsqueeze(-1)
            new_size = [1] * (len(input_ids.size()) - 1) + [input_ids.shape[-1]]
            idxs = torch.arange(input_ids.shape[-1],
                                device=input_ids.device).view(*new_size)
            output_ids = torch.masked_fill(input_ids,
                                           idxs > trunc_idxs,
                                           pad_token_id)
            mask = torch.masked_fill(torch.ones_like(input_ids),
                                     idxs > trunc_idxs,
                                     0)
            return output_ids, mask

        input_length = prompts["input_ids"].shape[-1]
        input_ids = prompts["input_ids"].unsqueeze(1).expand(-1,
                                                             outputs.shape[1],
                                                             -1)
        attention_mask = \
            prompts["attention_mask"].unsqueeze(1).expand(-1,
                                                          outputs.shape[1],
                                                          -1)

        # Process model completions
        completion_ids = outputs[:, :, input_length:]
        completion_ids, completion_mask = truncate_right(
            completion_ids,
            self.processing_class.eos_token_id,
            self.processing_class.pad_token_id,
        )

        completion_length = completion_ids.shape[-1]
        total_chunk_length = completion_length - completion_length // 2
        chunk_num = self.args.prefix_chunk_num
        chunk_size = 1 + (total_chunk_length - 1) // chunk_num
        chunk_pos = completion_length // 2 + chunk_size \
            + chunk_size * torch.arange(chunk_num, device=completion_ids.device)
        chunk_pos = input_length + torch.clamp(chunk_pos, 0, completion_length)
        
        data = {
            "input_ids": torch.cat((input_ids, completion_ids),
                                   dim=-1),
            "attention_mask": torch.cat((attention_mask, 
                                         completion_mask),
                                        dim=-1),
            "raw": prompts["raw"],
            "chunk_pos": chunk_pos,
        }
        return data
    
    def sub_data(self, data, start, end):
        if data is None:
            return None
        return {
            "input_ids": data["input_ids"][start:end],
            "attention_mask": data["attention_mask"][start:end],
            "raw": data["raw"][start:end],
            "chunk_pos": data["chunk_pos"],
        }

    def _process_data(self, data, num_samples):
        """
        Args:
            - data: dict of tensors of shape (batch_size, k*num_samples, seq_len)
                `chunk_pos` is a tensor of shape (num_chunks,)
        Returns:
            - data: dict of tensors of shape (batch_size*num_samples, k, seq_len)
                `chunk_pos` is a tensor of shape (num_chunks,)
        """
        batch_size, _, seq_len = data["input_ids"].shape
        data["input_ids"] = data["input_ids"].view(batch_size * num_samples,
                                                   -1,
                                                   seq_len)
        data["attention_mask"] = data["attention_mask"].view(batch_size*num_samples, -1, seq_len)
        data["raw"] = multiply_list(data["raw"], num_samples)
        return data

    def _compute_rewards(self, data, ref_data, input_length):
        # TODO: add support for reward model
        raise NotImplementedError
    
        with torch.no_grad():
            _, scores, _ = get_reward(
                self.reward_model, data["input_ids"], self.processing_class.pad_token_id, input_length
            )
            _, ref_scores, _ = get_reward(
                self.reward_model, ref_data["input_ids"], self.processing_class.pad_token_id, input_length
            )

        # Apply EOS penalty if needed
        if self.args.missing_eos_penalty is not None:
            contain_eos = torch.any(data["input_ids"] == self.processing_class.eos_token_id, dim=-1)
            ref_contain_eos = torch.any(ref_data["input_ids"] == self.processing_class.eos_token_id, dim=-1)
            scores[~contain_eos] -= self.args.missing_eos_penalty
            ref_scores[~ref_contain_eos] -= self.args.missing_eos_penalty

        return scores, ref_scores

    def _compute_prefs(self, y_yp, ypp, input_length):
        """
        Args:
            - y_yp: dict of tensors of shape (batch_size, 2, seq_len)
                `chunk_pos` is a tensor of shape (num_chunks,)
                reference model completions (y, y')
            - ypp: dict of tensors of shape (batch_size, 2, seq_len)
                `chunk_pos` is a tensor of shape (num_chunks,)
                model completions (y''_1, y''_2)
                if None, then compute prefs between y and y'
            - input_length: length of the input
        
        Returns:
            - prefs: shape (batch_size, num_chunks, 2)
                (avg(P(y > y''_i)), avg(P(y' > y''_i))) if ypp is not None
                (1/2, P(y > y')) if ypp is None
        """
        num_chunks = y_yp["chunk_pos"].shape[-1]
        prompts = y_yp["raw"]
        device = y_yp["input_ids"].device
        completions = []

        prefs = []
        for i in range(num_chunks):
            y = self.processing_class.batch_decode(
                y_yp["input_ids"][:, 0, input_length:y_yp["chunk_pos"][i]],
                skip_special_tokens=True
            )
            y = [c.strip() for c in y]
            yp = self.processing_class.batch_decode(
                y_yp["input_ids"][:, 1, input_length:y_yp["chunk_pos"][i]],
                skip_special_tokens=True
            )
            yp = [c.strip() for c in yp]
            if ypp:
                ypp_1 = self.processing_class.batch_decode(
                    ypp["input_ids"][:, 0, input_length:ypp["chunk_pos"][i]],
                    skip_special_tokens=True
                )
                ypp_1 = [c.strip() for c in ypp_1]

                if ypp["input_ids"].shape[1] == 2:
                    ypp_2 = self.processing_class.batch_decode(
                        ypp["input_ids"][:, 1, input_length:ypp["chunk_pos"][i]],
                        skip_special_tokens=True
                    )
                    ypp_2 = [c.strip() for c in ypp_2]
                else:
                    ypp_2 = None

            if is_conversational({"prompt": prompts[0]}):
                environment = jinja2.Environment()
                template = environment.from_string(SIMPLE_CHAT_TEMPLATE)
                prompts = [template.render(messages=p) for p in prompts]

                y = [[{"role": "assistant", "content": c}] for c in y]
                y = [template.render(messages=c) for c in y]
                yp = [[{"role": "assistant", "content": c}] for c in yp]
                yp = [template.render(messages=c) for c in yp]
                if ypp:
                    ypp_1 = [[{"role": "assistant", "content": c}] for c in ypp_1]
                    ypp_1 = [template.render(messages=c) for c in ypp_1]
                    if ypp_2:
                        ypp_2 = [[{"role": "assistant", "content": c}] for c in ypp_2]
                        ypp_2 = [template.render(messages=c) for c in ypp_2]
            
            if ypp:
                completions += list(zip(y + yp, ypp_1 + ypp_1))
                if ypp_2:
                    completions += list(zip(y + yp, ypp_2 + ypp_2))
            else:
                completions += list(zip(y, yp))

        prefs = self.judge.judge(prompts * (len(completions) // len(prompts)),
                                 completions)

        prefs = torch.tensor(prefs).to(device)
        if ypp:
            if ypp_2:
                prefs = prefs.view(num_chunks, 2, 2, -1).permute(3, 0, 2, 1)
                prefs = prefs.mean(dim=3)
            else:
                prefs = prefs.view(num_chunks, 2, -1).permute(2, 0, 1)
        else:
            prefs = prefs.view(num_chunks, 1, -1).permute(2, 0, 1)
            # (batch_size, num_chunks, 1)
            prefs = torch.cat((prefs, 0.5 * torch.ones_like(prefs)), dim=-1)

        return prefs

    def _compute_logps(self, model, data, input_length):
        """
        Args:
            - model: model to compute logprobs
            - data: dict of tensors of shape (batch_size, num_samples, seq_len)
                `chunk_pos` is a tensor of shape (batch_size, num_chunks)
                model completions
            - input_length: length of the input
        
        Returns:
            - logps: shape (batch_size, num_samples, num_chunks)
                logprobs of model completions under the model
        """
        def compute_logprobs_for_data(model, input_ids, attention_mask):
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits[:, input_length - 1:-1]
            logprobs = F.log_softmax(logits, dim=-1)
            token_logprobs = torch.gather(
                logprobs,
                2,
                input_ids[:, input_length:].unsqueeze(-1)
            ).squeeze(-1)
            return token_logprobs
        
        input_ids = data["input_ids"].clone()
        attention_mask = data["attention_mask"].clone()

        batch_size, num_samples, seq_len = input_ids.shape

        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)

        logps = compute_logprobs_for_data(model, input_ids, attention_mask)
        padding_mask = attention_mask[:, input_length:] == 0
        logps = logps.masked_fill(padding_mask, 0.0).cumsum(dim=-1)

        logps = logps[:, data["chunk_pos"] - input_length - 1].view(batch_size,
                                                                    num_samples,
                                                                    -1)
        
        if self.args.normalize_logprobs:
            masks = attention_mask[:, input_length:].cumsum(dim=-1)
            masks = masks[:, data["chunk_pos"] - input_length - 1].view(
                batch_size,
                num_samples,
                -1
            )
            logps = logps / masks

        return logps

    def _compute_losses(self, logps, ref_logps, prefs):
        """
        Args:
            - logps: shape (batch_size, 2, num_chunks)
                (\log \pi_\\theta (y), \log \pi_\\theta (y')))
            - ref_logps: shape (batch_size, 2, num_chunks)
                (\log \pi_{\\theta_{ref}} (y), \log \pi_{\\theta_{ref}} (y'))
            - prefs: shape (batch_size, num_chunks, 2)
                (P(y > y''), P(y' > y''))
        Returns:
            - loss: IPO loss
            - kl: not exactly KL, no gradient
        """
        loss = (logps[:, 0] - logps[:, 1] - ref_logps[:, 0] + ref_logps[:, 1] \
                - (prefs[:, :, 0] - prefs[:, :, 1]) / self.beta) ** 2
        
        kl = 0.5 * torch.abs(ref_logps[:, 0] + ref_logps[:, 1] \
                             - logps[:, 0].detach() - logps[:, 1].detach())
        return loss.mean(), kl.mean()

    def _log_statistics(
        self,
        y_yp,
        ypp,
        logps,
        ref_logps,
        prefs,
        kl,
        input_length,
    ):
        def gather_mean(tensor):
            return self.accelerator.gather(tensor).mean().item()

        self.stats["kl"].append(gather_mean(kl))

        axis = torch.arange(prefs.shape[0], device=prefs.device)
        chosen_idx = (prefs[:, 1] > prefs[:, 0]).long()

        chosen_logps = logps[axis, chosen_idx]
        rejected_logps = logps[axis, 1 - chosen_idx]
        self.stats["logps/chosen"].append(gather_mean(chosen_logps))
        self.stats["logps/rejected"].append(gather_mean(rejected_logps))

        chosen_prefs = prefs[axis, chosen_idx]
        rejected_prefs = prefs[axis, 1 - chosen_idx]
        pref_margins = chosen_prefs - rejected_prefs
        self.stats["prefs/chosen"].append(gather_mean(chosen_prefs))
        self.stats["prefs/rejected"].append(gather_mean(rejected_prefs))
        self.stats["prefs/margins"].append(gather_mean(pref_margins))

        chosen_rewards = self.beta * (chosen_logps \
                                      - ref_logps[axis, chosen_idx])
        rejected_rewards = self.beta * (rejected_logps \
                                        - ref_logps[axis, 1 - chosen_idx])
        reward_margins = chosen_rewards - rejected_rewards
        self.stats["rewards/chosen"].append(gather_mean(chosen_rewards))
        self.stats["rewards/rejected"].append(gather_mean(rejected_rewards))
        self.stats["rewards/margins"].append(gather_mean(reward_margins))

        accuracies = (reward_margins > 0).float()
        self.stats["rewards/accuracies"].append(gather_mean(accuracies))

        if ypp is not None:
            eos = (ypp["input_ids"][:, :, input_length:] \
                   == self.processing_class.eos_token_id).any(dim=-1)
            masks = ypp["attention_mask"][:, :, input_length:].sum(dim=-1)
        else:
            eos = (y_yp["input_ids"][:, :, input_length:] \
                   == self.processing_class.eos_token_id).any(dim=-1)
            masks = y_yp["attention_mask"][:, :, input_length:].sum(dim=-1)
        self.stats["contain_eos_token"].append(gather_mean(eos.float()))
        self.stats["gen_len"].append(gather_mean(masks.float()))

        self.stats["beta"].append(self.beta)

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        self.accumulated_steps += 1
        if self.accumulated_steps == 1:
            self.batch_inputs = []
            self.batch_datas = []
        if self.accumulated_steps == self.args.gradient_accumulation_steps:
            self.accumulated_steps = 0
            is_last_minibatch = True
        else:
            is_last_minibatch = False

        # Apply chat template and tokenize the input
        batch_size = len(next(iter(inputs.values())))
        prompts = inputs["prompt"]

        inputs = [{k: v[i] for k, v in inputs.items()}
                  for i in range(batch_size)]
        inputs = [maybe_apply_chat_template(x, self.processing_class)
                  for x in inputs]
        
        inputs = [{
            "raw": p,
            **self.tokenize_row(x,
                                self.model.config.is_encoder_decoder,
                                self.processing_class)
        } for p, x in zip(prompts, inputs)]
        self.batch_inputs.extend(inputs)

        if not is_last_minibatch:
            return torch.tensor(0.0, device=self.args.device)
        
        batch_size = self.args.per_device_train_batch_size
        gen_batch_size = max(self.args.per_device_generate_batch_size,
                             batch_size)
        for i in range(0, len(self.batch_inputs), gen_batch_size):
            inputs = self.batch_inputs[i:i+gen_batch_size]
            cur_batch_size = len(inputs)
            inputs = self.data_collator(inputs)

            # prompt only
            inputs = self._prepare_inputs(inputs)
            input_length = inputs["prompt_input_ids"].shape[1]
            prompts = {
                "input_ids": inputs["prompt_input_ids"],
                "attention_mask": inputs["prompt_attention_mask"],
                "raw": inputs["raw"],
            }

            y_yp = self._generate_completions(
                model,
                prompts,
                num_samples=2 * self.args.samples_per_prompt,
                mixture_coef=self.args.y_yp_mixture_coef,
                temperature=self.args.y_yp_temperature,
                top_k=self.args.y_yp_top_k,
                top_p=self.args.y_yp_top_p,
                min_p=self.args.y_yp_min_p,
            )
            if can_reduce_variance(self.args):
                # reduce variance
                ypp = None
            else:
                # [*] y'' ~ \pi_{\theta}
                # (y''_1, y''_2) ~ \pi_{\theta}
                ypp = self._generate_completions(
                    model,
                    prompts,
                    # num_samples=2 * self.args.samples_per_prompt,
                    num_samples=self.args.samples_per_prompt,
                    mixture_coef=self.args.ypp_mixture_coef,
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,
                    min_p=0.0,
                )

            y_yp = self._process_completions(y_yp, prompts)
            self._process_data(y_yp, self.args.samples_per_prompt)
            if ypp is not None:
                ypp = self._process_completions(ypp, prompts)
                self._process_data(ypp, self.args.samples_per_prompt)

            # Compute rewards
            if self.reward_model is not None:
                # TODO: add support for reward model
                raise NotImplementedError
            else:
                prefs = self._compute_prefs(y_yp, ypp, input_length)
        
            for j in range(0, cur_batch_size, batch_size):
                self.batch_datas.append({
                    "y_yp": self.sub_data(y_yp, j, j + batch_size),
                    "ypp": self.sub_data(ypp, j, j + batch_size),
                    "input_length": input_length,
                    "prefs": prefs[j:j + batch_size],
                })
        
        del self.batch_inputs

        model.train()

        if self.args.estimate_extra_grad:
            if self.state.global_step % 2 == 1:
                # restore model and optimizer from pi^{(t+1/2)} to pi^{(t)}
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            param.data.copy_(self.model_backup[name])
                            param.grad = None
                self.optimizer.load_state_dict(self.optimizer_backup)
            else:
                # backup model and optimizer at pi^{(t)}
                self.model_backup = {name: param.data.clone().detach() for name, param in model.named_parameters() if param.requires_grad}
                self.optimizer_backup = [None] * self.args.local_rank \
                    + [self.optimizer.state_dict()]
            
        loss_sum = 0
        for minibatch in self.batch_datas:
            y_yp = minibatch["y_yp"]
            ypp = minibatch["ypp"]
            input_length = minibatch["input_length"]
            prefs = minibatch["prefs"]

            # Compute logprobs
            # log \pi_{\theta} (y), log \pi_{\theta} (y')
            logps = self._compute_logps(model, y_yp, input_length)
            # (batch_size, 2, num_chunks)
            # log \pi_{\theta_{ref}} (y), log \pi_{\theta_{ref}} (y')
            with torch.no_grad():
                if self.ref_model is None:
                    with model.disable_adapter():
                        ref_logps = self._compute_logps(model,
                                                        y_yp,
                                                        input_length)
                else:
                    ref_logps = self._compute_logps(self.ref_model,
                                                    y_yp,
                                                    input_length)
                # (batch_size, 2, num_chunks)
            
            # Compute loss
            loss, kl = self._compute_losses(
                logps,
                ref_logps,
                prefs,
            )

            # Log everything
            self._log_statistics(
                y_yp=y_yp,
                ypp=ypp,
                logps=logps.detach().permute(0, 2, 1).reshape(-1, 2),
                ref_logps=ref_logps.permute(0, 2, 1).reshape(-1, 2),
                prefs=prefs.reshape(-1, 2),
                kl=kl,
                input_length=input_length,
            )

            kwargs = {}
            # For LOMO optimizers you need to explicitly use the learning rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss, **kwargs)

            loss_sum += loss.detach() / self.args.gradient_accumulation_steps

        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            empty_cache()

        return loss_sum

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str`, *optional*, defaults to `None`):
                The name of the model.
            dataset_name (`str`, *optional*, defaults to `None`):
                The name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(""
        #     """\
        # @inproceedings{munos2024nash,
        #     title        = {Nash Learning from Human Feedback},
        #     author       = {R{\'{e}}mi Munos and Michal Valko and Daniele Calandriello and Mohammad Gheshlaghi Azar and Mark Rowland and Zhaohan Daniel Guo and Yunhao Tang and Matthieu Geist and Thomas Mesnard and C{\\^{o}}me Fiegel and Andrea Michi and Marco Selvi and Sertan Girgin and Nikola Momchev and Olivier Bachem and Daniel J. Mankowitz and Doina Precup and Bilal Piot},
        #     year         = 2024,
        #     booktitle    = {Forty-first International Conference on Machine Learning, {ICML} 2024, Vienna, Austria, July 21-27, 2024},
        #     publisher    = {OpenReview.net},
        #     url          = {https://openreview.net/forum?id=Y5AmNYiyCQ}
        # }"""
                                   )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="Extragradient",
            trainer_citation=citation,
            paper_title="TBA",
            paper_id="TBA",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
