from src.myagent.agent import run_agent

MODEL = "gpt-6-luna"

tools = ["search_web", "read_webpage", "inspect_path", "read_file"]

conversation_input = input("Conversation ID: ").strip()
conversation_id = conversation_input if conversation_input else None

client_id = "juju"

while (prompt := input("You: ").strip()).lower() != "exit":
    if not prompt:
        continue

    response, conversation_id, _ = run_agent(
        prompt=prompt,
        model=MODEL,
        agent_tools=tools,
        authenticated_client_id=client_id,
        conversation_id=conversation_id,
    )
    print(response)

