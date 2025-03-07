import requests
import torch

from accelerate import Accelerator
from trl import BasePairwiseJudge

class LocalServerJudge(BasePairwiseJudge):
    def __init__(self, server_address: str):
        self.local_rank = Accelerator().local_process_index
        self.world_size = Accelerator().num_processes
        self.server_address = server_address
    
    def judge(
        self,
        prompts: list[str],
        completions: list[list[str]],
        shuffle_order: bool = True,
        **kwargs,
    ) -> list[float]:
        data = {
            "prompts": prompts,
            "completions": completions,
            "shuffle_order": shuffle_order,
        }
        gathered_data = [None] * self.world_size
        broadcast_results = [None] * self.world_size

        torch.distributed.all_gather_object(gathered_data, data)

        if self.local_rank == 0:
            all_data = {
                "prompts": sum([d["prompts"] for d in gathered_data], []),
                "completions": sum([d["completions"] for d in gathered_data], []),
                "shuffle_order": any([d["shuffle_order"] for d in gathered_data]),
            }

            while True:
                response = requests.post(self.server_address, json=all_data)
                if response.status_code == 200:
                    break
            results = response.json()["results"]
            
            idx = 0
            for i in range(self.world_size):
                broadcast_results[i] = results[idx:idx+len(gathered_data[i]["prompts"])]
            
        torch.distributed.broadcast_object_list(broadcast_results, 0)
        
        return broadcast_results[self.local_rank]
    