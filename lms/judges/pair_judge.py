import pdb
import torch

from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import BasePairwiseJudge

from utils import pref_model_query, MODEL_ROOT

class PairJudge(BasePairwiseJudge):
    """
    LLM judge based on any AutoModelForSequenceClassification.
    """

    def __init__(self, model_name_or_path: str):
        """
        Initialize the PairJudge.

        Args:
            - model_name_or_path (`str`):
                Path to the model or model identifier from Hugging Face Hub.
            - device (`str`, *optional*, defaults to `cuda`):
                Device to use for the model.
        """
        self.device = Accelerator().device
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            cache_dir=MODEL_ROOT,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
    
    def judge(
        self,
        prompts: list[str],
        completions: list[list[str]],
        shuffle_order: bool = True,
        **kwargs,
    ) -> list[float]:
        """
        Args:
            prompts (`list[str]`):
                List of prompts to judge.
            completions (`list[list[str]]`):
                List of completion pairs (inner list len = 2) for each prompt.
            shuffle_order (`bool`, *optional*, defaults to `True`):
                Whether to shuffle the order of the completions to avoid positional bias.

        Returns:
            `list[float]`:
                Returns softmax probabilities for the first completion.

        Raises:
            `ValueError`:
                If the number of completions per prompt is not exactly 2.
        """
        for c in completions:
            if len(c) != 2:
                raise ValueError("PairJudge requires exactly 2 completions per prompt.")

        # Shuffle the order of the completions to avoid positional bias
        if shuffle_order:
            flip_mask = torch.randint(0, 2,
                                      (len(prompts),)).bool()
        else:
            flip_mask = torch.zeros(len(prompts)).bool()
        
        completions = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions)]

        # Build the input for the model
        inputs = []
        for prompt, c in zip(prompts, completions):
            inputs.append(pref_model_query(prompt,
                                           c[0],
                                           c[1],
                                           task="classification"))

        # Tokenize the inputs
        inputs = self.tokenizer(inputs,
                                truncation=True,
                                padding="longest")
        inputs = {k: torch.LongTensor(v).to(self.device)
                     if isinstance(v, (torch.Tensor, list)) else v
                  for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.model(**inputs)
        
        scores = torch.softmax(logits.logits, dim=-1).to("cpu")
        axis = torch.arange(scores.shape[0])
        scores = scores[axis, flip_mask.long()].tolist()

        return scores
