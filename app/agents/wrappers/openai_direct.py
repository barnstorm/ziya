import json
import asyncio
import re
from typing import List, Dict, Optional, AsyncIterator, Any, Tuple, Union
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from app.utils.logging_utils import logger
from app.mcp.manager import get_mcp_manager
from openai import AsyncOpenAI


class DirectOpenAIModel:
    """
    Direct OpenAI model wrapper that uses the native openai SDK to support
    proper conversation history and native tool calling.

    Supports both GPT models (gpt-4o, gpt-4-turbo) and reasoning models (o1, o3-mini).
    """

    def __init__(
        self,
        model_name: str,
        temperature: Optional[float] = 1.0,
        max_output_tokens: int = 4096,
        top_p: Optional[float] = None,
        reasoning_effort: Optional[str] = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort  # "low", "medium", "high" for o-series
        self.mcp_manager = get_mcp_manager()

        # Determine if this is a reasoning model (o1, o3, etc.)
        self.is_reasoning_model = self._is_reasoning_model(model_name)

        logger.info(f"DirectOpenAIModel initialized: model={model_name}, temp={temperature}, max_output_tokens={max_output_tokens}")
        if self.is_reasoning_model:
            logger.info(f"Reasoning model detected, reasoning_effort={reasoning_effort}")

        # Create async client with optional base_url override
        import os
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        if base_url:
            self.client = AsyncOpenAI(base_url=base_url)
            logger.info(f"Created OpenAI async client with base_url={base_url}")
        else:
            self.client = AsyncOpenAI()
            logger.info("Created OpenAI async client")

    def _is_reasoning_model(self, model_name: str) -> bool:
        """Check if this is a reasoning model (o-series or GPT-5)."""
        # o-series reasoning models
        if model_name.startswith("o1") or model_name.startswith("o3") or model_name.startswith("o4"):
            return True
        # GPT-5 models are also reasoning models that don't support temperature
        if model_name.startswith("gpt-5"):
            return True
        return False

    def _extract_text_from_mcp_result(self, result: Any) -> str:
        """Extracts the text content from a structured MCP tool result."""
        if not isinstance(result, dict) or 'content' not in result:
            return str(result)

        content = result['content']
        if not isinstance(content, list) or not content:
            return str(result)

        first_item = content[0]
        if isinstance(first_item, dict) and 'text' in first_item:
            return first_item['text']

        return str(result)

    def _convert_langchain_tools_to_openai(self, tools: List[BaseTool]) -> List[Dict]:
        """Converts LangChain tools to OpenAI function calling format."""
        if not tools:
            return []

        openai_tools = []
        for tool in tools:
            try:
                schema = tool.args_schema.schema() if tool.args_schema else {}

                # Remove title from schema if present (OpenAI doesn't need it)
                if "title" in schema:
                    del schema["title"]

                # Ensure properties and required fields exist
                if "properties" not in schema:
                    schema["properties"] = {}
                if "required" not in schema:
                    schema["required"] = []

                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": schema
                    }
                }
                openai_tools.append(openai_tool)
            except Exception as e:
                logger.warning(f"Could not convert tool '{tool.name}' to OpenAI format: {e}")

        return openai_tools

    def _convert_messages_to_openai_format(self, messages: List[BaseMessage]) -> List[Dict]:
        """
        Converts LangChain messages to the format required by the OpenAI API.
        """
        openai_messages = []

        for message in messages:
            if isinstance(message, SystemMessage):
                # For reasoning models, system messages should be converted to user messages
                # as o-series models have limited system message support
                if self.is_reasoning_model:
                    openai_messages.append({
                        "role": "user",
                        "content": f"[System Instructions]\n{message.content}"
                    })
                else:
                    openai_messages.append({
                        "role": "system",
                        "content": message.content
                    })
            elif isinstance(message, HumanMessage):
                content = message.content

                # Handle multimodal content (images)
                if isinstance(content, list):
                    openai_content = self._format_multimodal_content(content)
                    openai_messages.append({
                        "role": "user",
                        "content": openai_content
                    })
                else:
                    openai_messages.append({
                        "role": "user",
                        "content": content
                    })
            elif isinstance(message, AIMessage):
                content = message.content

                # Clean out tool blocks from historical AI messages
                if isinstance(content, str):
                    content = re.sub(r'<TOOL_SENTINEL>.*?</TOOL_SENTINEL>', '', content, flags=re.DOTALL)
                    content = re.sub(r'```tool:.*?```', '', content, flags=re.DOTALL).strip()

                # Handle tool calls in AI message
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    tool_calls = []
                    for tc in message.tool_calls:
                        tool_calls.append({
                            "id": tc.get("id", f"call_{len(tool_calls)}"),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": json.dumps(tc.get("args", {}))
                            }
                        })
                    openai_messages.append({
                        "role": "assistant",
                        "content": content if content else None,
                        "tool_calls": tool_calls
                    })
                else:
                    if content:  # Skip empty messages
                        openai_messages.append({
                            "role": "assistant",
                            "content": content
                        })
            elif isinstance(message, ToolMessage):
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": message.tool_call_id if hasattr(message, 'tool_call_id') else message.name,
                    "content": message.content
                })

        return openai_messages

    def _format_multimodal_content(self, content: List) -> List[Dict]:
        """Format multimodal content (text + images) for OpenAI API."""
        openai_content = []

        for part in content:
            if isinstance(part, str):
                openai_content.append({"type": "text", "text": part})
            elif isinstance(part, dict):
                part_type = part.get('type')

                if part_type == 'text':
                    openai_content.append({"type": "text", "text": part.get('text', '')})
                elif part_type == 'image':
                    # Claude/Bedrock format: {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
                    source = part.get('source', {})
                    if source.get('type') == 'base64':
                        media_type = source.get('media_type', 'image/jpeg')
                        data = source.get('data', '')
                        openai_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{data}"
                            }
                        })
                elif part_type == 'image_url':
                    # Already in OpenAI format
                    openai_content.append(part)

        return openai_content if openai_content else [{"type": "text", "text": ""}]

    async def astream(self, messages: List[BaseMessage], **kwargs) -> AsyncIterator[Dict]:
        """
        Streams responses from the OpenAI model, handling native tool calls.
        """
        tools = kwargs.get("tools", [])
        openai_tools = self._convert_langchain_tools_to_openai(tools)
        history = self._convert_messages_to_openai_format(messages)

        # Main loop for handling multi-turn tool calls
        while True:
            logger.info(f"Calling OpenAI model {self.model_name} with history...")
            try:
                # Build request parameters
                request_params = {
                    "model": self.model_name,
                    "messages": history,
                    "max_completion_tokens": self.max_output_tokens,
                    "stream": True
                }

                # Add tools if available and model supports them
                if openai_tools:
                    request_params["tools"] = openai_tools
                    request_params["tool_choice"] = "auto"

                # Add parameters based on model type
                if self.is_reasoning_model:
                    # Reasoning models don't support temperature
                    if self.reasoning_effort:
                        request_params["reasoning_effort"] = self.reasoning_effort
                else:
                    # Standard GPT models support temperature and top_p
                    if self.temperature is not None:
                        request_params["temperature"] = self.temperature
                    if self.top_p is not None:
                        request_params["top_p"] = self.top_p

                response = await self.client.chat.completions.create(**request_params)

            except Exception as e:
                error_message = f"OpenAI API Error ({type(e).__name__}): {str(e)}"
                logger.error(error_message, exc_info=True)
                yield {"type": "error", "content": error_message}
                return

            tool_calls = {}  # Accumulate tool calls by index
            content_buffer = ""
            finish_reason = None

            async for chunk in response:
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta

                    # Capture finish reason
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                        logger.info(f"OpenAI model finish_reason: {finish_reason}")

                    # Handle content streaming
                    if delta.content:
                        content_buffer += delta.content
                        yield {"type": "text", "content": delta.content}

                    # Handle tool calls streaming
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls:
                                tool_calls[idx] = {
                                    "id": tc.id or f"call_{idx}",
                                    "name": "",
                                    "arguments": ""
                                }
                            if tc.id:
                                tool_calls[idx]["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    tool_calls[idx]["name"] = tc.function.name
                                if tc.function.arguments:
                                    tool_calls[idx]["arguments"] += tc.function.arguments

            logger.info(f"Stream ended. Tool calls: {len(tool_calls)}, Finish reason: {finish_reason}")

            if not tool_calls:
                logger.info("No tool calls from model. Ending loop.")
                break

            logger.info(f"Model returned {len(tool_calls)} tool call(s).")

            # Build assistant message with tool calls
            assistant_message = {
                "role": "assistant",
                "content": content_buffer if content_buffer else None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"]
                        }
                    }
                    for tc in tool_calls.values()
                ]
            }
            history.append(assistant_message)

            # Execute tool calls
            for tc in tool_calls.values():
                tool_name = tc["name"]
                try:
                    tool_args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    tool_args = {}

                yield {"type": "tool_start", "tool_name": tool_name, "input": tool_args}

                try:
                    tool_result_obj = await self.mcp_manager.call_tool(tool_name, tool_args)
                    tool_result_str = self._extract_text_from_mcp_result(tool_result_obj)

                    yield {"type": "tool_display", "tool_name": tool_name, "result": tool_result_str}

                    # Add tool result to history
                    history.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result_str
                    })
                except Exception as e:
                    error_message = f"Error executing tool {tool_name}: {e}"
                    logger.error(error_message)
                    yield {"type": "error", "content": error_message}
                    history.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": f"Error: {error_message}"
                    })

    def bind(self, **kwargs):
        """Compatibility method - return self for chaining."""
        return self
