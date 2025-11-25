from typing import Any, Dict, Iterator, List, Optional
import os
import streamlit as st
from openai import OpenAI
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
    ChatMessage,
    HumanMessage
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
import time

class AliyunLLM(BaseChatModel):
    """自定义阿里云聊天模型。"""

    model_name: str = None
    temperature: Optional[float] = 0
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 3
    api_key: Optional[str] = None

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """通过调用阿里云API从而响应输入。"""

        messages_dict = [_convert_message_to_dict(message) for message in messages]
        
        client = OpenAI(
            api_key=self.api_key or os.getenv("DASHSCOPE_API_KEY") or (st.secrets["DASHSCOPE_API_KEY"] if "DASHSCOPE_API_KEY" in st.secrets else None),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        start_time = time.time()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages_dict,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
            timeout=self.timeout,
            **kwargs
        )
        end_time = time.time()

        content = response.choices[0].message.content
        token_usage = response.usage
        
        message = AIMessage(
            content=content,
            additional_kwargs={},
            response_metadata={
                "time_in_seconds": end_time - start_time,
            },
            usage_metadata={
                "input_tokens": token_usage.prompt_tokens,
                "output_tokens": token_usage.completion_tokens,
                "total_tokens": token_usage.total_tokens,
            }
        )
        
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """通过调用阿里云API返回流式输出。"""
        messages_dict = [_convert_message_to_dict(message) for message in messages]
        
        client = OpenAI(
            api_key=self.api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages_dict,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
            timeout=self.timeout,
            stream=True,
            **kwargs
        )

        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                content = delta.content
                chunk_message = AIMessageChunk(content=content)
                if run_manager:
                    run_manager.on_llm_new_token(content, chunk=ChatGenerationChunk(message=chunk_message))
                yield ChatGenerationChunk(message=chunk_message)

    @property
    def _llm_type(self) -> str:
        """获取此聊天模型使用的语言模型类型。"""
        return self.model_name

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回一个标识参数的字典。"""
        return {
            "model_name": self.model_name,
        }

def _convert_message_to_dict(message: BaseMessage) -> dict:
    """把LangChain的消息格式转为OpenAI支持的格式"""
    message_dict: Dict[str, Any] = {"content": message.content}
    if (name := message.name or message.additional_kwargs.get("name")) is not None:
        message_dict["name"] = name

    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict

if __name__ == "__main__":
    # Test
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv())
    
    llm = AliyunLLM(model_name="qwen3-max")
    print(llm.invoke("你好"))
