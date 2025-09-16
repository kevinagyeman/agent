from typing import List, Dict, Tuple, Optional, Any
import asyncio
import os
import anthropic
from pathlib import Path
import json
from dotenv import load_dotenv

load_dotenv()


SYSTEM_PROMPT = """You are a helpful coding agent that assists with programming tasks and file operations.

When responding to requests:
1. Analyze what the user needs
2. Use the minimum number of tools necessary to accomplish the task
3. After using tools, provide a concise summary of what was done

IMPORTANT: Once you've completed the requested task, STOP and provide your final response. Do not continue creating additional files or performing extra actions unless specifically asked.

Examples of good behavior:
- User: "Create a file that adds numbers" → Create ONE file, then summarize
- User: "Create files for add and subtract" → Create ONLY those two files, then summarize
- User: "Create math operation files" → Ask for clarification on which operations, or create a reasonable set and stop

After receiving tool results:
- If the task is complete, provide a final summary
- Only continue with more tools if the original request is not yet fulfilled
- Do not interpret successful tool execution as a request to do more

Be concise and efficient. Complete the requested task and stop."""


TOOLS_SCHEMA = [
    {
        "name": "read_file",
        "description": "Read the contents of a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The path to the file to read"}
            },
            "required": ["path"]
        }
    },
    # Other tool definitions...
]


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

    async def _read_file(self, path: str) -> Dict[str, Any]:
        try:
            file_path = (self.working_directory / path).resolve()
            if not str(file_path).startswith(str(self.working_directory)):
                return {"error": "Access denied: path outside working directory"}
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {"success": True, "content": content, "path": str(file_path)}
        except Exception as e:
            return {"error": f"Could not read file: {str(e)}"}

    async def _execute_tool_calls(self, tool_uses: List[Any]) -> List[Dict]:
        tool_results = []
        for tool_use in tool_uses:
            print(f"   Executing: {tool_use.name}")
            try:
                if tool_use.name == "read_file":
                    result = await self._read_file(tool_use.input.get("path", ""))
                elif tool_use.name == "write_file":
                    result = await self._write_file(tool_use.input.get("path", ""), 
                                                   tool_use.input.get("content", ""))
                elif tool_use.name == "list_files":
                    result = await self._list_files(tool_use.input.get("path", "."))
                elif tool_use.name == "search_files":
                    result = await self._search_files(tool_use.input.get("pattern", ""), 
                                                     tool_use.input.get("path", "."))
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


async def main():
    print("Welcome to Baby Claude Code!!")
    print("Type 'exit' or 'quit' to quit, 'clear' to clear history, 'history' to show recent messages")
    print("-" * 50)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        api_key = input("Enter your Anthropic API key: ").strip()
    agent = CodingAgent(api_key)
    while True:
        try:
            user_input = input("\n You: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            elif user_input.lower() == 'clear':
                agent.messages = []
                agent.save_history()
                print("History cleared!")
                continue
            elif user_input.lower() == 'history':
                print("\nRecent conversation history:")
                for msg in agent.messages[-10:]:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if len(content) > 100:
                        content = content[:100] + "..."
                    timestamp = msg.get("timestamp", "")
                    print(f"  [{role}] {content}")
                continue
            elif not user_input:
                continue
            print("\n Agent processing...")
            response = await agent.process_message(user_input)
            print(f"\n Agent: {response}")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
