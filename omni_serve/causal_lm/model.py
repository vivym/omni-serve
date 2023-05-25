from typing import Any, Dict, Optional, List

import numpy as np
import torch
from ray import serve
from transformers import (
    AutoModelForCausalLM, PreTrainedModel, StoppingCriteria, StoppingCriteriaList
)

from omni_serve.utils.init import no_init


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids_list: List[List[int]]) -> None:
        super().__init__()

        self.stop_ids_list = torch.as_tensor(stop_ids_list, dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.stop_ids_list = self.stop_ids_list.to(input_ids.device)

        for stop_ids in self.stop_ids_list:
            if torch.all(stop_ids == input_ids[0][-stop_ids.shape[0] :]).item():
                return True
        return False


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 4, "num_gpus": 1},
    user_config={
        "model_name_or_path": "./weights/vicuna-7b-v1.1",
        "stop_ids_list": [[2]],
    }
)
class CausalLMDeployment:
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        stop_ids_list: Optional[List[List[int]]] = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path

        if stop_ids_list is not None:
            self.stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids_list)])

        if model_name_or_path is not None:
            self.model = self.init_model(model_name_or_path)

    def reconfigure(self, config: Dict[str, Any]):
        model_name_or_path = config.get("model_name_or_path", self.model_name_or_path)
        if model_name_or_path != self.model_name_or_path and model_name_or_path is not None:
            self.model = self.init_model(model_name_or_path)
            self.model_name_or_path = model_name_or_path

        stop_ids_list = config.get("stop_ids_list", None)
        if stop_ids_list is not None:
            self.stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids_list)])

    def init_model(self, model_name_or_path: str) -> PreTrainedModel:
        with no_init():
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
            )
        model.cuda()
        model.eval()
        return model

    async def generate(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        max_length: int = 2048,
        max_new_tokens: Optional[int] = None,
        max_time: Optional[float] = None,
        do_sample: bool = False,
        num_beams: int = 1,
        num_beam_groups: int = 1,
        penalty_alpha: Optional[float] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        typical_p: float = 1.0,
        epsilon_cutoff: float = 0.0,
        eta_cutoff: float = 0.0,
        diversity_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        encoder_repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        num_return_sequences: int = 1,
    ) -> np.ndarray:
        input_ids_tensor = torch.from_numpy(input_ids).cuda(non_blocking=True)
        attention_mask_tensor = torch.from_numpy(attention_mask).cuda(non_blocking=True)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                max_time=max_time,
                do_sample=do_sample,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                penalty_alpha=penalty_alpha,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                typical_p=typical_p,
                epsilon_cutoff=epsilon_cutoff,
                eta_cutoff=eta_cutoff,
                diversity_penalty=diversity_penalty,
                repetition_penalty=repetition_penalty,
                encoder_repetition_penalty=encoder_repetition_penalty,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                num_return_sequences=num_return_sequences,
                stopping_criteria=self.stopping_criteria,
            )

        outputs = outputs[:, input_ids_tensor.shape[1]:]

        return outputs.cpu().numpy()
