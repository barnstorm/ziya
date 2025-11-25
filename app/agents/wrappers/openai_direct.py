"""
Direct OpenAI API wrapper for Ziya.

This module provides a direct integration with the OpenAI API,
supporting streaming responses and native function calling.
"""

import json
import asyncio
import re
from typing import List, Dict, Optional, AsyncIterator, Any, Tuple, Union
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from app.utils.logging_utils import logger
from app.mcp.manager import get_mcp_manager
from langchain_core.tools import BaseTool

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai package not installed. Install with: pip install openai")


class DirectOpenAIModel:
    """
    Direct OpenAI model wrapper that uses the native OpenAI SDK
    to support proper conversation history and native function calling.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0
    ):
        """
        Initialize the OpenAI model wrapper.

        Args:
            model_name: The OpenAI model ID (e.g., "gpt-4o", "gpt-4-turbo")
            temperature: Sampling temperature (0.0-2.0)
            max_output_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            presence_penalty: Presence penalty (-2.0 to 2.0)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required for DirectOpenAIModel. "
                "Install with: pip install openai"
            )

        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.mcp_manager = get_mcp_manager()

        # Initialize the OpenAI client (uses OPENAI_API_KEY from environment)
        self.client = openai.AsyncOpenAI()

        logger.info(
            f"DirectOpenAIModel initialized: model={model_name}, "
            f"temp={temperature}, max_output_tokens={max_output_tokens}"
        )

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
        """
        Converts LangChain tools to OpenAI function calling format.

        Args:
            tools: List of LangChain tools

        Returns:
            List of tools in OpenAI format
        """
        if not tools:
            return []

        openai_tools = []
        for tool in tools:
            try:
                schema = tool.args_schema.schema() if tool.args_schema else {}

                # Convert to OpenAI tool format
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": {
                            "type": "object",
                            "properties": schema.get("properties", {}),
                            "required": schema.get("required", [])
                        }
                    }
                }
                openai_tools.append(openai_tool)
            except Exception as e:
                logger.warning(f"Could not convert tool '{tool.name}' to OpenAI format: {e}")

        return openai_tools

    def _convert_messages_to_openai_format(
        self,
        messages: List[BaseMessage]
    ) -> List[Dict]:
        """
        Converts LangChain messages to the format required by the OpenAI API.

        Args:
            messages: List of LangChain messages

        Returns:
            List of messages in OpenAI format
        """
        openai_messages = []

        for message in messages:
            if isinstance(message, SystemMessage):
                openai_messages.append({
                    "role": "system",
                    "content": message.content
                })
            elif isinstance(message, HumanMessage):
                openai_messages.append({
                    "role": "user",
                    "content": message.content
                })
            elif isinstance(message, AIMessage):
                content = message.content

                # Clean out tool blocks from historical AI messages
                content = re.sub(r'<TOOL_SENTINEL>.*?</TOOL_SENTINEL>', '', content, flags=re.DOTALL)
                content = re.sub(r'\`\`\`tool:.*?\`\`\`', '', content, flags=re.DOTALL).strip()

                # Skip empty messages
                if content:
                    openai_messages.append({
                        "role": "assistant",
                        "content": content
                    })

        # Merge consecutive messages of the same role (OpenAI allows this but it's cleaner)
        merged_messages = []
        for msg in openai_messages:
            if merged_messages and merged_messages[-1]["role"] == msg["role"]:
                merged_messages[-1]["content"] += f"\n\n{msg['content']}"
            else:
                merged_messages.append(msg)

        return merged_messages

    async def astream(self, messages: List[BaseMessage], **kwargs) -> AsyncIterator[Dict]:
        """
        Streams responses from the OpenAI model, handling native function calls.

        Args:
            messages: List of LangChain messages
            **kwargs: Additional arguments (tools, etc.)

        Yields:
            Dict with type and content keys
        """
        tools = kwargs.get("tools", [])
        openai_tools = self._convert_langchain_tools_to_openai(tools)
        history = self._convert_messages_to_openai_format(messages)

        # Main loop for handling multi-turn tool calls
        while True:
            logger.info("Calling OpenAI model with history...")

            try:
                # Build request parameters
                request_params = {
                    "model": self.model_name,
                    "messages": history,
                    "max_completion_tokens": self.max_output_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "frequency_penalty": self.frequency_penalty,
                    "presence_penalty": self.presence_penalty,
                    "stream": True,
                }

                # Add tools if available
                if openai_tools:
                    request_params["tools"] = openai_tools
                    request_params["tool_choice"] = "auto"

                # Create streaming request
                response_stream = await self.client.chat.completions.create(**request_params)

            except openai.AuthenticationError as e:
                error_message = f"OpenAI Authentication Error: {str(e)}. Check your OPENAI_API_KEY."
                logger.error(error_message)
                yield {"type": "error", "content": error_message}
                return
            except openai.RateLimitError as e:
                error_message = f"OpenAI Rate Limit Error: {str(e)}"
                logger.error(error_message)
                yield {"type": "error", "content": error_message}
                return
            except openai.BadRequestError as e:
                error_message = f"OpenAI Bad Request Error: {str(e)}"
                logger.error(error_message)
                yield {"type": "error", "content": error_message}
                return
            except Exception as e:
                error_message = f"OpenAI API Error ({type(e).__name__}): {str(e)}"
                logger.error(error_message, exc_info=True)
                yield {"type": "error", "content": error_message}
                return

            tool_calls = {}  # Dict to accumulate tool calls by index
            text_content = ""
            finish_reason = None

            try:
                async for chunk in response_stream:
                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]
                    delta = choice.delta
                    finish_reason = choice.finish_reason

                    # Handle text content
                    if delta.content:
                        text_content += delta.content
                        yield {"type": "text", "content": delta.content}

                    # Handle tool calls
                    if delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            idx = tool_call_delta.index

                            if idx not in tool_calls:
                                tool_calls[idx] = {
                                    "id": "",
                                    "name": "",
                                    "arguments": ""
                                }

                            if tool_call_delta.id:
                                tool_calls[idx]["id"] = tool_call_delta.id

                            if tool_call_delta.function:
                                if tool_call_delta.function.name:
                                    tool_calls[idx]["name"] = tool_call_delta.function.name
                                if tool_call_delta.function.arguments:
                                    tool_calls[idx]["arguments"] += tool_call_delta.function.arguments

                logger.info(f"Stream ended. Finish reason: {finish_reason}")

            except Exception as e:
                error_message = f"Error during OpenAI stream: {str(e)}"
                logger.error(error_message, exc_info=True)
                yield {"type": "error", "content": error_message}
                return

            # Process tool calls if any
            if not tool_calls or finish_reason != "tool_calls":
                logger.info("No tool calls from model. Ending loop.")
                break

            logger.info(f"Model returned {len(tool_calls)} tool call(s).")

            # Build assistant message with tool calls
            assistant_message = {
                "role": "assistant",
                "content": text_content if text_content else None,
                "tool_calls": []
            }

            for idx in sorted(tool_calls.keys()):
                tc = tool_calls[idx]
                assistant_message["tool_calls"].append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"]
                    }
                })

            history.append(assistant_message)

            # Execute tools and collect results
            for idx in sorted(tool_calls.keys()):
                tc = tool_calls[idx]
                tool_name = tc["name"]
                tool_call_id = tc["id"]

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
                        "tool_call_id": tool_call_id,
                        "content": tool_result_str
                    })
                except Exception as e:
                    error_message = f"Error executing tool {tool_name}: {e}"
                    logger.error(error_message)
                    yield {"type": "error", "content": error_message}

                    # Add error result to history
                    history.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": f"Error: {error_message}"
                    })

    def bind(self, **kwargs):
        """Compatibility method for LangChain interface."""
        return self
