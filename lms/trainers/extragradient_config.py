from dataclasses import dataclass, field

from trl.trainer.online_dpo_config import OnlineDPOConfig


@dataclass
class ExtragradientConfig(OnlineDPOConfig):
    r"""
    Configuration class for the [`ExtragradientTrainer`].

    Subclass of [`ExtragradientConfig`] we can use all its arguments and add the following:

    Parameters:
        y_yp_mixture_coef (`float`, *optional*, defaults to `1`):
            The mixture coefficient for sampling (y, y').
            (1 - mixture_coef) * log (\pi_{\\theta}) + mixture_coef * log (\pi_{\\theta{ref}})
        y_yp_temperature (`float`, *optional*, defaults to `1.0`):
            The temperature for sampling y and y'.
        y_yp_top_k (`int`, *optional*, defaults to `0`):
            The top-k value for sampling y and y'.
        y_yp_top_p (`float`, *optional*, defaults to `1.0`):
            The top-p value for sampling y and y'.
        y_yp_min_p (`float`, *optional*, defaults to `0.0`):
            The min-p value for sampling
        ypp_mixture_coef (`float`, *optional*, defaults to `0`):
            The mixture coefficient for sampling y''.
            (1 - mixture_coef) * log (\pi_{\\theta}) + mixture_coef * log (\pi_{\\theta{ref}})
        estimate_extra_grad (`bool`, *optional*, defaults to `True`):
            Whether to estimate the extra gradient.
            If set to `False`, then this trainer is equivalent to online IPO
        prefix_chunk_num (`int`, *optional*, defaults to `1`):
            The number of evaluating prefix chunks.
        samples_per_prompt (`int`, *optional*, defaults to `1`):
            The number of samples per prompt.
        per_device_generate_batch_size (`int`, *optional*, defaults to `-1`):
            The micro batch size for generating samples.
            If set to `-1`, then the micro batch size is equal to the train micro batch size.
        max_seq_length (`int`, *optional*, defaults to `-1`):
            The maximum sequence length.
            If set to `-1`, then `max_new_tokens` takes effect.
    """

    y_yp_mixture_coef: float = 1
    y_yp_temperature: float = 1.0
    y_yp_top_k: int = 0
    y_yp_top_p: float = 1.0
    y_yp_min_p: float = 0.0

    ypp_mixture_coef: float = 0
    
    estimate_extra_grad: bool = True
    normalize_logprobs: bool = True
    prefix_chunk_num: int = 1
    samples_per_prompt: int = 1
    per_device_generate_batch_size: int = -1
    
    max_seq_length: int = -1