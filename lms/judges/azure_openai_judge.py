import os
import pdb
import torch

from azure.identity import AzureCliCredential, ChainedTokenCredential, DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from trl import BasePairwiseJudge

from utils import pref_model_query

class AzureOpenAIJudge(BasePairwiseJudge):
    def __init__(self, resource_name, api_version, model):
        self.azure_credential = ChainedTokenCredential(
            AzureCliCredential(),
            DefaultAzureCredential(
                exclude_cli_credential=True,
                exclude_environment_credential=True,
                exclude_shared_token_cache_credential=True,
                exclude_developer_cli_credential=True,
                exclude_powershell_credential=True,
                exclude_interactive_browser_credential=True,
                exclude_visual_studio_code_credentials=True,
                managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
            )
        )

        self.token_provider = get_bearer_token_provider(self.azure_credential,
            "https://cognitiveservices.azure.com/.default")
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=f"https://{resource_name}.openai.azure.com/",
            azure_ad_token_provider=self.token_provider,
        )

        self.model = model

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
                raise ValueError("AzureOpenAIJudge requires exactly 2 completions per prompt.")

        # Shuffle the order of the completions to avoid positional bias
        if shuffle_order:
            flip_mask = torch.randint(0, 2,
                                      (len(prompts),)).bool()
        else:
            flip_mask = torch.zeros(len(prompts)).bool()
        
        completions = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions)]

        scores = []
        for prompt, c in zip(prompts, completions):
            while True:
                try:
                    ret = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": pref_model_query(prompt,
                                                            c[0],
                                                            c[1],
                                                            task="generation"),
                            },
                        ],
                    )
                except Exception as e:
                    continue
                
                choice = ret.choices[0].message.content

                if choice in ["0", "1"]:
                    break
            
            scores.append([1 - float(choice), float(choice)])
        
        scores = torch.tensor(scores)
        axis = torch.arange(scores.shape[0])
        scores = scores[axis, flip_mask.long()].tolist()

        return scores
