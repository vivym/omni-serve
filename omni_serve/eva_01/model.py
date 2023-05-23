from typing import Any, Dict, List, Optional

import numpy as np
import torch
from ray import serve
from transformers import Blip2VisionModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from omni_serve.utils.init import no_init


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 4, "num_gpus": 1},
    user_config={
        "model_name_or_path": "./weights/eva-01-vit",
        "task_head_paths": {},
    }
)
class EVAViTDeployment:
    def __init__(
        self,
        model_name_or_path: str = "./weights/eva-01-vit",
        task_head_paths: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path

        with no_init():
            self.model = Blip2VisionModel.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
            )
        self.model.eval()
        self.model.cuda()

        if task_head_paths is None:
            task_head_paths = {}

        self.task_heads = {
            k: torch.jit.load(v).eval().cuda()
            for k, v in task_head_paths.items()
        }

    def reconfigure(self, config: Dict[str, Any]):
        model_name_or_path = config.get("model_name_or_path", self.model_name_or_path)
        if model_name_or_path != self.model_name_or_path:
            with no_init():
                self.model = Blip2VisionModel.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch.float16,
                )
            self.model.eval()
            self.model.cuda()
            self.model_name_or_path = model_name_or_path

        task_head_paths = config.get("task_head_paths", {})
        del self.task_heads
        self.task_heads = {
            k: torch.jit.load(v).eval().cuda()
            for k, v in task_head_paths.items()
        }

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.05)
    async def batch_inference(
        self, images: List[np.ndarray]
    ) -> List[Dict[str, np.ndarray]]:
        batch_size = len(images)

        images_ndarray = np.stack(images, axis=0)
        images_tensor = torch.from_numpy(images_ndarray).cuda(non_blocking=True)

        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                outputs: BaseModelOutputWithPooling = self.model(images_tensor)
                task_results = {
                    task_name: task_head(outputs.pooler_output).cpu().numpy()
                    for task_name, task_head in self.task_heads.items()
                }

        return [
            {
                task_name: task_result[i]
                for task_name, task_result in task_results.items()
            }
            for i in range(batch_size)
        ]

    async def __call__(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        return await self.batch_inference(image)
