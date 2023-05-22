import numpy as np
from ray import serve
from transformers import BlipImageProcessor
from transformers.image_utils import ImageInput
from transformers.image_processing_utils import BatchFeature


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 2, "num_gpus": 0},
)
class PreprocessorDeployment:
    def __init__(self, model_name_or_path: str = "./weights/eva-01-vit") -> None:
        self.processor = BlipImageProcessor.from_pretrained(model_name_or_path)

    async def __call__(self, images: ImageInput) -> np.ndarray:
        return self.processor(images).pixel_values
