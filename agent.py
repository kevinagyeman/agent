from typing import List, Dict, Tuple, Optional, Any
import anthropic
from pathlib import Path
import os
import json
from config import SYSTEM_PROMPT, TOOLS_SCHEMA
from tools.file_ops import read_file

class CodingAgent:
    def __init__(self, api_key: str, working_directory: str = ".", history_file: str = "agent_history.json"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.working_directory = Path(working_directory).resolve()
        self.history_file = history_file
        self.messages: List[Dict] = []
        self.load_history()

    async def _call_claude(self, messages: List[Dict]) -> Tuple[Any, Optional[str]]:
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                system=SYSTEM_PROMPT,
                tools=TOOLS_SCHEMA,
                messages=messages,
                temperature=0.7
            )
            return response.content, None
        except anthropic.APIError as e:
            return None, f"API Error: {str(e)}"
        except Exception as e:
            return None, f"Unexpected error calling Claude API: {str(e)}"

    async def _read_file(self, path: str) -> dict:
        return await read_file(self.working_directory, path)

    async def _execute_tool_calls(self, tool_uses: List[Any]) -> List[Dict]:
        tool_results = []
        for tool_use in tool_uses:
            print(f"   Executing: {tool_use.name}")
            try:
                if tool_use.name == "read_file":
                    result = await self._read_file(tool_use.input.get("path", ""))
                # Implement other tool names as needed
                else:
                    result = {"error": f"Unknown tool: {tool_use.name}"}
            except Exception as e:
                result = {"error": f"Tool execution failed: {str(e)}"}

            if "success" in result and result["success"]:
                print(f"Tool executed successfully")
            elif "error" in result:
                print(f"Error: {result['error']}")
            tool_results.append({
                "tool_use_id": tool_use.id,
                "content": json.dumps(result)
            })
        return tool_results

    def save_history(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.messages, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save history: {e}")

    def load_history(self):
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.messages = json.load(f)
        except Exception:
            self.messages = []

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self.save_history()

    def build_messages_list(self, user_input: Optional[str] = None,
                            tool_results: Optional[List[Dict]] = None,
                            assistant_content: Optional[Any] = None,
                            max_history: int = 20) -> List[Dict]:
        messages = []
        start_idx = max(0, len(self.messages) - max_history)
        for msg in self.messages[start_idx:]:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})
        if user_input:
            messages.append({"role": "user", "content": user_input})
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})
        if tool_results:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tr["tool_use_id"], "content": tr["content"]}
                    for tr in tool_results
                ]
            })
        return messages

    async def react_loop(self, user_input: str) -> str:
        self.add_message("user", user_input)
        messages = self.build_messages_list(user_input=user_input)
        last_complete_response = None
        safety_limit = 20
        iterations = 0
        while iterations < safety_limit:
            iterations += 1
            content_blocks, error = await self._call_claude(messages)
            if error:
                error_msg = f"Error: {error}"
                self.add_message("assistant", error_msg)
                return error_msg
            text_responses, tool_uses = self._parse_claude_response(content_blocks)
            if text_responses:
                last_complete_response = "\n".join(text_responses)
            if not tool_uses:
                break
            tool_results = await self._execute_tool_calls(tool_uses)
            messages = self.build_messages_list(
                assistant_content=content_blocks,
                tool_results=tool_results
            )
        if not last_complete_response:
            final_response = "I couldn't generate a response."
        elif iterations >= safety_limit:
            final_response = f"{last_complete_response}\n\n(Note: I reached my processing limit. You may want to break this down into smaller steps.)"
        else:
            final_response = last_complete_response
        self.add_message("assistant", final_response)
        return final_response

    async def process_message(self, user_input: str) -> str:
        try:
            response = await self.react_loop(user_input)
            return response
        except Exception as e:
            error_msg = f"Unexpected error processing message: {str(e)}"
            self.add_message("assistant", error_msg)
            return error_msg

    def _parse_claude_response(self, content_blocks: Any) -> Tuple[List[str], List[Any]]:
        text_responses = []
        tool_uses = []
        for block in content_blocks:
            if block.type == "text":
                text_responses.append(block.text)
                print(f" {block.text}")
            elif block.type == "tool_use":
                tool_uses.append(block)
                print(f" Tool call: {block.name}")
        return text_responses, tool_uses
