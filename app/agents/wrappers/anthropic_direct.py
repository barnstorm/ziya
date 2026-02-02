import json
import asyncio
import re
from typing import List, Dict, Optional, AsyncIterator, Any, Tuple, Union
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from app.utils.logging_utils import logger
from app.mcp.manager import get_mcp_manager
from anthropic import AsyncAnthropic


class DirectAnthropicModel:
    """
    Direct Anthropic model wrapper that uses the native anthropic SDK to support
    proper conversation history and native tool calling.

    Supports Claude models (claude-sonnet-4, claude-opus-4, etc.) with full
    tool use and extended thinking capabilities.
    """

    def __init__(
        self,
        model_name: str,
        temperature: Optional[float] = 1.0,
        max_output_tokens: int = 4096,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.mcp_manager = get_mcp_manager()

        logger.info(f"DirectAnthropicModel initialized: model={model_name}, temp={temperature}, max_output_tokens={max_output_tokens}")

        # Create async client
        self.client = AsyncAnthropic()
        logger.info("Created Anthropic async client")

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

    def _convert_langchain_tools_to_anthropic(self, tools: List[BaseTool]) -> List[Dict]:
        """Converts LangChain tools to Anthropic tool format."""
        if not tools:
            return []

        anthropic_tools = []
        for tool in tools:
            try:
                schema = tool.args_schema.schema() if tool.args_schema else {}

                # Remove title from schema if present
                if "title" in schema:
                    del schema["title"]

                # Ensure properties exists
                if "properties" not in schema:
                    schema["properties"] = {}

                anthropic_tool = {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": schema
                }
                anthropic_tools.append(anthropic_tool)
            except Exception as e:
                logger.warning(f"Could not convert tool '{tool.name}' to Anthropic format: {e}")

        return anthropic_tools

    def _convert_messages_to_anthropic_format(self, messages: List[BaseMessage]) -> Tuple[List[Dict], Optional[str]]:
        """
        Converts LangChain messages to the format required by the Anthropic API.
        Returns (messages, system_prompt) tuple since Anthropic handles system separately.
        """
        anthropic_messages = []
        system_prompt = None

        for message in messages:
            if isinstance(message, SystemMessage):
                # Anthropic handles system message separately
                if system_prompt is None:
                    system_prompt = message.content
                else:
                    system_prompt += f"\n\n{message.content}"
            elif isinstance(message, HumanMessage):
                content = message.content

                # Handle multimodal content (images)
                if isinstance(content, list):
                    anthropic_content = self._format_multimodal_content(content)
                    anthropic_messages.append({
                        "role": "user",
                        "content": anthropic_content
                    })
                else:
                    anthropic_messages.append({
                        "role": "user",
                        "content": content
                    })
            elif isinstance(message, AIMessage):
                content = message.content

                # Clean out tool blocks from historical AI messages
                if isinstance(content, str):
                    content = re.sub(r'<TOOL_SENTINEL>.*?</TOOL_SENTINEL>', '', content, flags=re.DOTALL)
                    content = re.sub(r'```tool:.*?```', '', content, flags=re.DOTALL).strip()

                # Handle tool use in AI message
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    msg_content = []
                    if content:
                        msg_content.append({"type": "text", "text": content})
                    for tc in message.tool_calls:
                        msg_content.append({
                            "type": "tool_use",
                            "id": tc.get("id", f"toolu_{len(msg_content)}"),
                            "name": tc.get("name", ""),
                            "input": tc.get("args", {})
                        })
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": msg_content
                    })
                else:
                    if content:  # Skip empty messages
                        anthropic_messages.append({
                            "role": "assistant",
                            "content": content
                        })
            elif isinstance(message, ToolMessage):
                # Anthropic expects tool results in a specific format
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id if hasattr(message, 'tool_call_id') else message.name,
                    "content": message.content
                }
                # Tool results should be part of a user message
                if anthropic_messages and anthropic_messages[-1]["role"] == "user":
                    # Append to existing user message
                    if isinstance(anthropic_messages[-1]["content"], list):
                        anthropic_messages[-1]["content"].append(tool_result)
                    else:
                        anthropic_messages[-1]["content"] = [
                            {"type": "text", "text": anthropic_messages[-1]["content"]},
                            tool_result
                        ]
                else:
                    anthropic_messages.append({
                        "role": "user",
                        "content": [tool_result]
                    })

        return anthropic_messages, system_prompt

    def _format_multimodal_content(self, content: List) -> List[Dict]:
        """Format multimodal content (text + images) for Anthropic API."""
        anthropic_content = []

        for part in content:
            if isinstance(part, str):
                anthropic_content.append({"type": "text", "text": part})
            elif isinstance(part, dict):
                part_type = part.get('type')

                if part_type == 'text':
                    anthropic_content.append({"type": "text", "text": part.get('text', '')})
                elif part_type == 'image':
                    # Claude/Bedrock format: {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
                    source = part.get('source', {})
                    if source.get('type') == 'base64':
                        anthropic_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": source.get('media_type', 'image/jpeg'),
                                "data": source.get('data', '')
                            }
                        })
                elif part_type == 'image_url':
                    # OpenAI format - convert to Anthropic format if possible
                    image_url = part.get('image_url', {})
                    url = image_url.get('url', '')
                    if url.startswith('data:'):
                        # Parse data URL
                        match = re.match(r'data:([^;]+);base64,(.+)', url)
                        if match:
                            anthropic_content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": match.group(1),
                                    "data": match.group(2)
                                }
                            })

        return anthropic_content if anthropic_content else [{"type": "text", "text": ""}]

    async def astream(self, messages: List[BaseMessage], **kwargs) -> AsyncIterator[Dict]:
        """
        Streams responses from the Anthropic model, handling native tool calls.
        """
        tools = kwargs.get("tools", [])
        anthropic_tools = self._convert_langchain_tools_to_anthropic(tools)
        history, system_prompt = self._convert_messages_to_anthropic_format(messages)

        # Main loop for handling multi-turn tool calls
        while True:
            logger.info(f"Calling Anthropic model {self.model_name} with history...")
            try:
                # Get model limits and clamp max_tokens
                from app.agents.models import ModelManager
                model_config = None
                try:
                    # Find model config by model_id
                    for name, cfg in ModelManager.MODEL_CONFIGS.get("anthropic", {}).items():
                        if cfg.get("model_id") == self.model_name:
                            model_config = cfg
                            break
                except Exception:
                    pass

                max_tokens = self.max_output_tokens
                if model_config:
                    model_max = model_config.get("max_output_tokens", 8192)
                    if max_tokens > model_max:
                        logger.warning(f"Clamping max_tokens from {max_tokens} to model max {model_max}")
                        max_tokens = model_max

                # Build request parameters
                request_params = {
                    "model": self.model_name,
                    "messages": history,
                    "max_tokens": max_tokens,
                }
                logger.info(f"ANTHROPIC_REQUEST: model={self.model_name}, max_tokens={max_tokens}")

                # Add system prompt if present
                if system_prompt:
                    request_params["system"] = system_prompt

                # Add tools if available
                if anthropic_tools:
                    request_params["tools"] = anthropic_tools

                # Add optional parameters (ensure proper types)
                if self.temperature is not None:
                    request_params["temperature"] = float(self.temperature)
                if self.top_p is not None:
                    request_params["top_p"] = float(self.top_p)
                if self.top_k is not None:
                    top_k_val = int(self.top_k)
                    if top_k_val > 0:
                        request_params["top_k"] = top_k_val

                # Use streaming
                async with self.client.messages.stream(**request_params) as response:
                    tool_uses = []
                    content_buffer = ""
                    current_tool_use = None

                    async for event in response:
                        if event.type == "content_block_start":
                            if hasattr(event, 'content_block'):
                                block = event.content_block
                                if block.type == "tool_use":
                                    current_tool_use = {
                                        "id": block.id,
                                        "name": block.name,
                                        "input": ""
                                    }
                        elif event.type == "content_block_delta":
                            if hasattr(event, 'delta'):
                                delta = event.delta
                                if delta.type == "text_delta":
                                    content_buffer += delta.text
                                    yield {"type": "text", "content": delta.text}
                                elif delta.type == "input_json_delta":
                                    if current_tool_use is not None:
                                        current_tool_use["input"] += delta.partial_json
                        elif event.type == "content_block_stop":
                            if current_tool_use is not None:
                                tool_uses.append(current_tool_use)
                                current_tool_use = None
                        elif event.type == "message_stop":
                            pass

                    # Get final message for stop reason
                    final_message = await response.get_final_message()
                    stop_reason = final_message.stop_reason
                    logger.info(f"Anthropic model stop_reason: {stop_reason}")
                    logger.info(f"Anthropic ACTUAL model used: {final_message.model}")

            except Exception as e:
                error_message = f"Anthropic API Error ({type(e).__name__}): {str(e)}"
                logger.error(error_message, exc_info=True)
                yield {"type": "error", "content": error_message}
                return

            logger.info(f"Stream ended. Tool uses: {len(tool_uses)}, Stop reason: {stop_reason}")

            if not tool_uses:
                logger.info("No tool calls from model. Ending loop.")
                break

            logger.info(f"Model returned {len(tool_uses)} tool call(s).")

            # Build assistant message with tool uses
            assistant_content = []
            if content_buffer:
                assistant_content.append({"type": "text", "text": content_buffer})
            for tu in tool_uses:
                try:
                    tool_input = json.loads(tu["input"]) if tu["input"] else {}
                except json.JSONDecodeError:
                    tool_input = {}
                assistant_content.append({
                    "type": "tool_use",
                    "id": tu["id"],
                    "name": tu["name"],
                    "input": tool_input
                })

            history.append({
                "role": "assistant",
                "content": assistant_content
            })

            # Execute tool calls and collect results
            tool_results = []
            for tu in tool_uses:
                tool_name = tu["name"]
                try:
                    tool_args = json.loads(tu["input"]) if tu["input"] else {}
                except json.JSONDecodeError:
                    tool_args = {}

                yield {"type": "tool_start", "tool_name": tool_name, "input": tool_args}

                try:
                    tool_result_obj = await self.mcp_manager.call_tool(tool_name, tool_args)
                    tool_result_str = self._extract_text_from_mcp_result(tool_result_obj)

                    yield {"type": "tool_display", "tool_name": tool_name, "result": tool_result_str}

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "content": tool_result_str
                    })
                except Exception as e:
                    error_message = f"Error executing tool {tool_name}: {e}"
                    logger.error(error_message)
                    yield {"type": "error", "content": error_message}
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "content": f"Error: {error_message}",
                        "is_error": True
                    })

            # Add tool results as a user message
            history.append({
                "role": "user",
                "content": tool_results
            })

    def bind(self, **kwargs):
        """Compatibility method - return self for chaining."""
        return self
