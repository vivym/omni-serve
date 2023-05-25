import copy
import re
import random
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
import torch
from ray import serve
from pydantic import BaseModel
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer

from .schemas import ChatMessage, Role as OpenAIRole


class Role(Enum):
    NONE = "NONE"
    HUMAN = "HUMAN"
    ASSISTANT = "ASSISTANT"


class Message(BaseModel):
    # Role of the speaker
    role: Role =  Role.NONE

    # Text of the message
    content: Optional[str] = None

    to_be_predicted: Optional[bool] = None

    def get_content(self) -> str:
        return self.content or ""


class HumanMessage(Message):
    role: Role = Role.HUMAN


class AssistantMessage(Message):
    role: Role = Role.ASSISTANT


class Conversation(BaseModel):
    human: HumanMessage

    assistant: AssistantMessage


@dataclass
class ChatPrompts:
    input_ids: Union[torch.Tensor, np.ndarray]

    attention_mask: Union[torch.Tensor, np.ndarray]

    num_tokens: Union[int, List[int]] = 0

    eos_token_id: int = 0

    is_batched: bool = False

    def __post_init__(self):
        if self.is_batched:
            assert isinstance(self.num_tokens, list)
        else:
            assert isinstance(self.num_tokens, int)

    @classmethod
    def collate(
        cls,
        batch: List["ChatPrompts"],
        pad_to_multiple_of: int = 1,
    ) -> "ChatPrompts":
        for data in batch:
            assert not data.is_batched, "Cannot collate batched samples."

        # Check eos_token_id
        eos_token_id = batch[0].eos_token_id
        assert all(data.eos_token_id == eos_token_id for data in batch), (
            "eos_token_id should be the same for all samples in a batch, but got:",
            [data.eos_token_id for data in batch]
        )

        batch_size = len(batch)

        max_input_ids = max(data.input_ids.shape[0] for data in batch)
        max_input_ids = (max_input_ids + (pad_to_multiple_of - 1)) // pad_to_multiple_of * pad_to_multiple_of

        if isinstance(batch[0].input_ids, np.ndarray):
            input_ids = np.full(
                (batch_size, max_input_ids), fill_value=eos_token_id, dtype=np.int64
            )
            attention_masks = np.zeros_like(input_ids)
        else:
            input_ids = torch.full(
                (batch_size, max_input_ids), fill_value=eos_token_id, dtype=torch.long
            )
            attention_masks = torch.zeros_like(input_ids)

        for i, data in enumerate(batch):
            input_ids[i, : data.input_ids.shape[0]] = data.input_ids
            attention_masks[i, : data.input_ids.shape[0]] = data.attention_mask

        return cls(
            input_ids=input_ids,
            attention_mask=attention_masks,
            num_tokens=[data.num_tokens for data in batch],
            eos_token_id=eos_token_id,
            is_batched=True,
        )


@dataclass
class ChatPromptTemplate:
    system_message: List[str] = field(default_factory=list)

    # Role name of the human user
    human_name: str = "Human"

    # Role name of the AI assistant
    assistant_name: str = "Assistant"

    # Few shot examples
    conversations: List[Conversation] = field(default_factory=list)

    conversation_template: str = "{human_name}: {human_content} ###\n{assistant_name}: {assistant_content} ###\n"

    tokenizer_name_or_path: str = "bigscience/bloomz-7b1"

    tokenizer: Optional[PreTrainedTokenizer] = None

    def __post_init__(self):
        if self.tokenizer is None:
            trust_remote_code = self.tokenizer_name_or_path in ["THUDM/chatglm-6b"]
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_path,
                padding_side="left",
                use_fast=False,
                trust_remote_code=trust_remote_code,
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def copy(self) -> "ChatPromptTemplate":
        return ChatPromptTemplate(
            system_message=copy.copy(self.system_message),
            human_name=self.human_name,
            assistant_name=self.assistant_name,
            conversations=copy.copy(self.conversations),
            conversation_template=self.conversation_template,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            tokenizer=self.tokenizer,
        )

    def text_processor(self, text: str) -> str:
        text = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            text.lower(),
        )
        text = re.sub(
            r"\s{2,}",
            " ",
            text,
        )
        text = text.rstrip("\n")
        text = text.strip(" ")

        return text

    def parse_conversations(self, conversations: dict) -> List[Conversation]:
        conversations: List[Conversation] = [
            Conversation.parse_obj(conv)
            for conv in conversations
        ]

        for conv in conversations:
            if conv.human.content is not None:
                conv.human.content = self.text_processor(conv.human.content)
            if conv.assistant.content is not None:
                conv.assistant.content = self.text_processor(conv.assistant.content)

        return conversations

    def format_conversation(
        self,
        conversation: Conversation,
        round: int,
    ) -> Tuple[Message, Message]:
        human_content = conversation.human.get_content()
        assistant_content = conversation.assistant.get_content()

        index = self.conversation_template.find("{assistant_content}")
        assert index >= 0, "conversation_template must contain '{assistant_content}'."

        template_1 = self.conversation_template[:index]
        text_1 = template_1.format(
            human_name=self.human_name,
            human_content=human_content,
            assistant_name=self.assistant_name,
            round=round,
        )

        # Remove trailing newline if assistant_text is empty
        if not assistant_content:
            text_1 = text_1.rstrip()

        message_1 = Message(content=text_1)

        if assistant_content:
            template_2 = self.conversation_template[index:]
            message_2 = Message(
                content=template_2.format(
                    human_name=self.human_name,
                    assistant_name=self.assistant_name,
                    assistant_content=assistant_content,
                ),
            )
        else:
            message_2 = Message(content="")

        return message_1, message_2

    def tokenize_and_append(
        self,
        message: Union[str, Message],
        max_length: int,
        input_ids: List[int],
    ) -> int:
        if max_length <= 0:
            return 0

        if isinstance(message, str):
            message = Message(content=message)

        content = message.get_content()

        if len(content) == 0:
            return 0

        num_remaining_tokens = max_length

        tokens: BatchEncoding = self.tokenizer(
            text=content,
            padding=False,
            truncation=True,
            max_length=num_remaining_tokens,
            add_special_tokens=False,
        )
        input_ids += tokens.input_ids
        num_tokens = len(tokens.input_ids)

        num_remaining_tokens -= num_tokens

        return max_length - num_remaining_tokens

    def format(
        self,
        conversations: Union[List[Conversation], List[dict]],
        max_length: int = 512,
        return_tensors: str = "pt",
    ) -> ChatPrompts:
        assert len(conversations) > 0, "At least one conversation is required."
        if isinstance(conversations[0], dict):
            conversations = self.parse_conversations(conversations)

        input_ids = [self.tokenizer.bos_token_id]
        num_remaining_tokens = max_length - 1

        tokenize_and_append = partial(
            self.tokenize_and_append,
            input_ids=input_ids,
        )

        if len(self.system_message) > 0:
            system_message = random.choice(self.system_message)
            num_remaining_tokens -= tokenize_and_append(
                system_message,
                max_length=num_remaining_tokens,
            )

        all_conversations = self.conversations + conversations
        for i, conversation in enumerate(all_conversations):
            message_1, message_2 = self.format_conversation(conversation, round=i)
            if message_2.to_be_predicted:
                assert i == len(all_conversations) - 1, (
                    "Only the last message of the last conversation can be to_be_predicted."
                )
            num_remaining_tokens -= tokenize_and_append(
                message_1, max_length=num_remaining_tokens
            )
            num_remaining_tokens -= tokenize_and_append(
                message_2, max_length=num_remaining_tokens
            )

        input_ids = np.asarray(input_ids, dtype=np.int64)
        attention_mask = np.ones_like(input_ids, dtype=np.int64)

        # To tensor
        if return_tensors == "pt":
            input_ids = torch.from_numpy(input_ids)
            attention_mask = torch.from_numpy(attention_mask)
        else:
            assert return_tensors == "np", (
                f"return_tensors must be 'pt' or 'np', but got {return_tensors}."
            )

        return ChatPrompts(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_tokens=input_ids.shape[0],
            eos_token_id=self.tokenizer.eos_token_id,
        )


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 4, "num_gpus": 0},
    user_config={
        "system_message": [
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        ],
        "human_name": "USER",
        "assistant_name": "ASSISTANT",
        "conversation_template": "{human_name}: {human_content} {assistant_name}: {assistant_content} </s>",
        "tokenizer_name_or_path": "./weights/vicuna-7b-v1.1",
    }
)
class ChatPromptTemplateDeployment:
    def __init__(self) -> None:
        self.chat_prompt_template: Optional[ChatPromptTemplate] = None

    def reconfigure(self, config: Dict[str, Any]):
        system_message = config.get("system_message", [])
        human_name = config.get("human_name", "Human")
        assistant_name = config.get("assistant_name", "Assistant")
        conversation_template = config.get(
            "conversation_template",
            "{human_name}: {human_content} {assistant_name}: {assistant_content} </s>",
        )
        tokenizer_name_or_path = config.get(
            "tokenizer_name_or_path", "./weights/vicuna-7b-v1.1"
        )

        self.chat_prompt_template = ChatPromptTemplate(
            system_message=system_message,
            human_name=human_name,
            assistant_name=assistant_name,
            conversation_template=conversation_template,
            tokenizer_name_or_path=tokenizer_name_or_path,
        )

    async def batch_decode(self, input_ids: np.ndarray) -> List[str]:
        return self.chat_prompt_template.tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    async def format(
        self,
        messages: List[ChatMessage],
        max_length: int = 2048,
    ):
        chat_prompt_template = self.chat_prompt_template.copy()

        conversations: List[Conversation] = []

        last_human_texts = []
        last_assistant_texts = []
        last_role = None

        for msg in messages:
            if msg.role == OpenAIRole.user:
                if last_role == OpenAIRole.assistant:
                    assert len(last_human_texts) > 0
                    conversations.append(Conversation(
                        human=HumanMessage(content="\n".join(last_human_texts)),
                        assistant=AssistantMessage(content="\n".join(last_assistant_texts)),
                    ))
                    last_human_texts = []
                    last_assistant_texts = []

                last_human_texts.append(msg.content)
            elif msg.role == Role.assistant:
                last_assistant_texts.append(msg.content)
            elif msg.role == Role.system:
                # TODO: add stop token
                chat_prompt_template.system_message.append(msg.content)
            else:
                raise ValueError(f"Invalid role {msg.role}")

            last_role = msg.role

        if len(last_human_texts) > 0 or len(last_assistant_texts) > 0:
            assert len(last_human_texts) > 0
            conversations.append(Conversation(
                human=HumanMessage(content="\n".join(last_human_texts)),
                assistant=AssistantMessage(content="\n".join(last_assistant_texts)),
            ))

        return chat_prompt_template.format(
            conversations=conversations,
            max_length=max_length,
            return_tensors="np",
        )
