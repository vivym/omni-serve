import time
from typing import List

import numpy as np
from ray import serve
from ray.serve.dag import InputNode
from ray.serve.drivers import DAGDriver
from ray.serve.handle import RayServeDeploymentHandle
from starlette.exceptions import HTTPException

from .chat_prompt_template import ChatPromptTemplateDeployment, ChatPrompts
from .model import CausalLMDeployment
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatResponse,
    FinishReason,
    Role,
    Usage,
)


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 2, "num_gpus": 0},
)
class CasusalLMTIngress:
    def __init__(
        self,
        chat_prompt_template: RayServeDeploymentHandle,
        model: RayServeDeploymentHandle
    ) -> None:
        self.chat_prompt_template = chat_prompt_template
        self.model = model

    async def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        max_length = request.max_tokens
        if max_length > 2048:
            max_length = 2048

        try:
            chat_promps_ref = await self.chat_prompt_template.format.remote(
                request.messages, max_length=max_length
            )
            chat_promps: ChatPrompts = await chat_promps_ref
            # TODO: padding left for causal lm
            chat_promps = ChatPrompts.collate([chat_promps], pad_to_multiple_of=1)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        try:
            outputs_ref = await self.model.generate.remote(
                input_ids=chat_promps.input_ids,
                attention_mask=chat_promps.attention_mask,
                do_sample=True,
                temperature=request.temperature,
                top_p=request.top_p,
                num_return_sequences=request.n,
            )
            outputs: np.ndarray = await outputs_ref
            num_completion_tokens = outputs.size
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        try:
            messages_ref = await self.chat_prompt_template.batch_decode.remote(outputs)
            messages: List[str] = await messages_ref
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        choices = [
            ChatResponse(
                index=i,
                message=ChatMessage(
                    role=Role.assistant,
                    content=msg,
                ),
                finish_reason=FinishReason.stop,
            )
            for i, msg in enumerate(messages)
        ]

        return ChatCompletionResponse(
            id="chat",
            created=int(time.time()),
            choices=choices,
            usage=Usage(
                prompt_tokens=sum(chat_promps.num_tokens),
                completion_tokens=num_completion_tokens,
            )
        )


with InputNode() as request:
    chat_prompt_template = ChatPromptTemplateDeployment.bind()

    model = CausalLMDeployment.bind()

    ingress = CasusalLMTIngress.bind(chat_prompt_template, model)

    chat_completions = ingress.chat_completions.bind(request)

    dag = DAGDriver.bind(
        {
            "/v1/chat/completions": chat_completions,
        },
        http_adapter=ChatCompletionRequest,
    )
