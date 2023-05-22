from typing import Dict, Optional

import numpy as np
import torch
from ray import serve
from transformers import Blip2VisionModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from omni_serve.utils.init import no_init


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 4, "num_gpus": 1},
)
class EVAViTDeployment:
    def __init__(
        self,
        model_name_or_path: str = "./weights/eva-01-vit",
        task_heads_path: Optional[Dict[str, str]] = None,
    ) -> None:
        with no_init():
            self.model = Blip2VisionModel.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
            )
        self.model.eval()
        self.model.cuda()

        if task_heads_path is None:
            task_heads_path = {}

        self.task_heads = {
            k: torch.jit.load(v).eval().cuda()
            for k, v in task_heads_path.items()
        }

    async def __call__(self, images: np.ndarray) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            image_tensors = torch.from_numpy(images).cuda(non_blocking=True)

            with torch.autocast(device_type="cuda"):
                outputs: BaseModelOutputWithPooling = self.model(image_tensors)
                return {
                    task_name: task_head(outputs.pooler_output).cpu().numpy()
                    for task_name, task_head in self.task_heads.items()
                }
