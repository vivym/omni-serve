from typing import Any, Dict, List

import numpy as np
from ray import serve
from transformers import BlipImageProcessor


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 2, "num_gpus": 0},
    user_config={"model_name_or_path": "./weights/eva-01-vit"},
)
class PreprocessorDeployment:
    def __init__(self, model_name_or_path: str = "./weights/eva-01-vit") -> None:
        self.model_name_or_path = model_name_or_path
        self.processor = BlipImageProcessor.from_pretrained(model_name_or_path)

    def reconfigure(self, config: Dict[str, Any]):
        model_name_or_path = config.get("model_name_or_path", self.model_name_or_path)
        if model_name_or_path != self.model_name_or_path:
            self.processor = BlipImageProcessor.from_pretrained(model_name_or_path)
            self.model_name_or_path = model_name_or_path

    @serve.batch(max_batch_size=4, batch_wait_timeout_s=0.01)
    async def batch_preprocess(self, images: List[np.ndarray]) -> List[np.ndarray]:
        return self.processor(images).pixel_values

    async def __call__(self, images: np.ndarray) -> np.ndarray:
        return await self.batch_preprocess(images)
