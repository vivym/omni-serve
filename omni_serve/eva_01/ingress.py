from typing import List

import numpy as np
import ray
from ray import serve
from ray.serve.dag import InputNode
from ray.serve.drivers import DAGDriver
from ray.serve.handle import RayServeDeploymentHandle
from ray.serve.http_adapters import image_to_ndarray
from starlette.exceptions import HTTPException

from .preprocessor import PreprocessorDeployment
from .model import EVAViTDeployment


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 4, "num_gpus": 0},
)
class EVAViTIngress:
    def __init__(
        self,
        preprocessor: RayServeDeploymentHandle,
        model: RayServeDeploymentHandle
    ) -> None:
        self.preprocessor = preprocessor
        self.model = model

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.05)
    async def batch_inference(self, images: List[np.ndarray]):
        images_ndarray_ref = await self.preprocessor.remote(images)
        images_ndarray = await images_ndarray_ref
        images_ndarray = np.stack(images_ndarray, axis=0)
        results_ref = await self.model.remote(images_ndarray)
        results = await results_ref
        return [results for _ in range(len(images))]

    async def inference(self, image: np.ndarray):
        return await self.batch_inference(image)


with InputNode() as request:
    preprocessor = PreprocessorDeployment.bind()

    model = EVAViTDeployment.bind()

    vit_ingress = EVAViTIngress.bind(preprocessor, model)

    vit_inference = vit_ingress.inference.bind(request)

    ingress = DAGDriver.bind(
        {"/eva-01/vit": vit_inference},
        http_adapter=image_to_ndarray,
    )
