"""
Direct Anthropic API wrapper for Ziya.

This module provides a direct integration with the Anthropic API,
supporting streaming responses, native tool calling, and extended thinking.
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
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic package not installed. Install with: pip install anthropic")


class DirectAnthropicModel:
    """
    Direct Anthropic model wrapper that uses the native Anthropic SDK
    to support proper conversation history, native tool calling, and extended thinking.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.3,
        max_output_tokens: int = 8192,
        thinking_enabled: bool = False,
        thinking_budget: int = 10000
    ):
        """
        Initialize the Anthropic model wrapper.

        Args:
            model_name: The Anthropic model ID (e.g., "claude-sonnet-4-20250514")
            temperature: Sampling temperature (0.0-1.0)
            max_output_tokens: Maximum tokens to generate
            thinking_enabled: Whether to enable extended thinking
            thinking_budget: Token budget for thinking (when enabled)
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package is required for DirectAnthropicModel. "
                "Install with: pip install anthropic"
            )

        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.thinking_enabled = thinking_enabled
        self.thinking_budget = thinking_budget
        self.mcp_manager = get_mcp_manager()

        # Initialize the Anthropic client (uses ANTHROPIC_API_KEY from environment)
        self.client = anthropic.AsyncAnthropic()

        logger.info(
            f"DirectAnthropicModel initialized: model={model_name}, "
            f"temp={temperature}, max_output_tokens={max_output_tokens}, "
            f"thinking_enabled={thinking_enabled}"
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

    def _convert_langchain_tools_to_anthropic(self, tools: List[BaseTool]) -> List[Dict]:
        """
        Converts LangChain tools to Anthropic tool format.

        Args:
            tools: List of LangChain tools

        Returns:
            List of tools in Anthropic format
        """
        if not tools:
            return []

        anthropic_tools = []
        for tool in tools:
            try:
                schema = tool.args_schema.schema() if tool.args_schema else {}

                # Convert to Anthropic tool format
                anthropic_tool = {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": {
                        "type": "object",
                        "properties": schema.get("properties", {}),
                        "required": schema.get("required", [])
                    }
                }
                anthropic_tools.append(anthropic_tool)
            except Exception as e:
                logger.warning(f"Could not convert tool '{tool.name}' to Anthropic format: {e}")

        return anthropic_tools

    def _convert_messages_to_anthropic_format(
        self,
        messages: List[BaseMessage]
    ) -> Tuple[List[Dict], Optional[str]]:
        """
        Converts LangChain messages to the format required by the Anthropic API.

        Args:
            messages: List of LangChain messages

        Returns:
            Tuple of (messages list, system instruction)
        """
        anthropic_messages = []
        system_instruction = None

        for message in messages:
            if isinstance(message, SystemMessage):
                # Anthropic handles system as a separate parameter
                if system_instruction is None:
                    system_instruction = message.content
                else:
                    # Append additional system messages
                    system_instruction += f"\n\n{message.content}"
                continue

            role = ""
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                # Skip unknown message types
                continue

            content = message.content

            # Clean out tool blocks from historical AI messages
            if role == "assistant":
                content = re.sub(r'<TOOL_SENTINEL>.*?</TOOL_SENTINEL>', '', content, flags=re.DOTALL)
                content = re.sub(r'\`\`\`tool:.*?\`\`\`', '', content, flags=re.DOTALL).strip()

            # Skip empty messages
            if not content or not content.strip():
                continue

            # Merge consecutive messages of the same role
            if anthropic_messages and anthropic_messages[-1]['role'] == role:
                # Append to existing message
                existing_content = anthropic_messages[-1]['content']
                if isinstance(existing_content, str):
                    anthropic_messages[-1]['content'] = f"{existing_content}\n\n{content}"
                elif isinstance(existing_content, list):
                    existing_content.append({"type": "text", "text": content})
            else:
                anthropic_messages.append({
                    'role': role,
                    'content': content
                })

        return anthropic_messages, system_instruction

    async def astream(self, messages: List[BaseMessage], **kwargs) -> AsyncIterator[Dict]:
        """
        Streams responses from the Anthropic model, handling native tool calls.

        Args:
            messages: List of LangChain messages
            **kwargs: Additional arguments (tools, etc.)

        Yields:
            Dict with type and content keys
        """
        tools = kwargs.get("tools", [])
        anthropic_tools = self._convert_langchain_tools_to_anthropic(tools)
        history, system_instruction = self._convert_messages_to_anthropic_format(messages)

        # Main loop for handling multi-turn tool calls
        while True:
            logger.info("Calling Anthropic model with history...")

            try:
                # Build request parameters
                request_params = {
                    "model": self.model_name,
                    "max_tokens": self.max_output_tokens,
                    "messages": history,
                }

                # Add system instruction if present
                if system_instruction:
                    request_params["system"] = system_instruction

                # Add tools if available
                if anthropic_tools:
                    request_params["tools"] = anthropic_tools

                # Handle thinking mode vs regular mode
                if self.thinking_enabled:
                    # Extended thinking requires temperature=1 and special budget
                    request_params["temperature"] = 1.0
                    request_params["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": self.thinking_budget
                    }
                else:
                    request_params["temperature"] = self.temperature

                # Create streaming request
                response_stream = self.client.messages.stream(**request_params)

            except anthropic.AuthenticationError as e:
                error_message = f"Anthropic Authentication Error: {str(e)}. Check your ANTHROPIC_API_KEY."
                logger.error(error_message)
                yield {"type": "error", "content": error_message}
                return
            except anthropic.RateLimitError as e:
                error_message = f"Anthropic Rate Limit Error: {str(e)}"
                logger.error(error_message)
                yield {"type": "error", "content": error_message}
                return
            except Exception as e:
                error_message = f"Anthropic API Error ({type(e).__name__}): {str(e)}"
                logger.error(error_message, exc_info=True)
                yield {"type": "error", "content": error_message}
                return

            tool_calls = []
            text_content = ""
            thinking_content = ""

            try:
                async with response_stream as stream:
                    async for event in stream:
                        # Handle different event types
                        if event.type == "content_block_start":
                            block = event.content_block
                            if hasattr(block, 'type'):
                                if block.type == "thinking":
                                    # Start of thinking block
                                    pass
                                elif block.type == "tool_use":
                                    # Start of tool use block
                                    tool_calls.append({
                                        "id": block.id,
                                        "name": block.name,
                                        "input": ""
                                    })

                        elif event.type == "content_block_delta":
                            delta = event.delta
                            if hasattr(delta, 'type'):
                                if delta.type == "text_delta":
                                    text = delta.text
                                    text_content += text
                                    yield {"type": "text", "content": text}
                                elif delta.type == "thinking_delta":
                                    thinking_content += delta.thinking
                                    # Optionally yield thinking content
                                    # yield {"type": "thinking", "content": delta.thinking}
                                elif delta.type == "input_json_delta":
                                    # Accumulate tool input JSON
                                    if tool_calls:
                                        tool_calls[-1]["input"] += delta.partial_json

                        elif event.type == "message_stop":
                            logger.info("Message stream completed")

                    # Get the final message for stop reason
                    final_message = await stream.get_final_message()
                    stop_reason = final_message.stop_reason
                    logger.info(f"Stream ended. Stop reason: {stop_reason}")

            except Exception as e:
                error_message = f"Error during Anthropic stream: {str(e)}"
                logger.error(error_message, exc_info=True)
                yield {"type": "error", "content": error_message}
                return

            # Process tool calls if any
            if not tool_calls or stop_reason != "tool_use":
                logger.info("No tool calls from model. Ending loop.")
                break

            logger.info(f"Model returned {len(tool_calls)} tool call(s).")

            # Build assistant message with tool use
            assistant_content = []
            if text_content:
                assistant_content.append({"type": "text", "text": text_content})

            for tool_call in tool_calls:
                # Parse the accumulated JSON input
                try:
                    tool_input = json.loads(tool_call["input"]) if tool_call["input"] else {}
                except json.JSONDecodeError:
                    tool_input = {}

                assistant_content.append({
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    "input": tool_input
                })

            history.append({"role": "assistant", "content": assistant_content})

            # Execute tools and collect results
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                try:
                    tool_args = json.loads(tool_call["input"]) if tool_call["input"] else {}
                except json.JSONDecodeError:
                    tool_args = {}

                yield {"type": "tool_start", "tool_name": tool_name, "input": tool_args}

                try:
                    tool_result_obj = await self.mcp_manager.call_tool(tool_name, tool_args)
                    tool_result_str = self._extract_text_from_mcp_result(tool_result_obj)

                    yield {"type": "tool_display", "tool_name": tool_name, "result": tool_result_str}

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call["id"],
                        "content": tool_result_str
                    })
                except Exception as e:
                    error_message = f"Error executing tool {tool_name}: {e}"
                    logger.error(error_message)
                    yield {"type": "error", "content": error_message}
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call["id"],
                        "content": f"Error: {error_message}",
                        "is_error": True
                    })

            # Add tool results to history
            if tool_results:
                history.append({"role": "user", "content": tool_results})

    def bind(self, **kwargs):
        """Compatibility method for LangChain interface."""
        return self
