import os
from agent import CodingAgent
from dotenv import load_dotenv
import asyncio

load_dotenv()

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
