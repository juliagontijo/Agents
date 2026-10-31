from src.myagent.agent import MEMORY_PATH, run_agent
from src.resources.conversation.repository import ConversationRepository

MODEL = "gpt-6-luna"
COMPACTION_TOKEN_LIMIT = 2_000

tools = ["search_web", "read_webpage", "inspect_path", "read_file"]

conversation_input = input("Conversation ID: ").strip()
conversation_id = conversation_input if conversation_input else None

client_id = "luca"
repository = ConversationRepository(MEMORY_PATH)
conversation = repository.open(client_id=client_id, conversation_id=conversation_id)
print(f"Conversation ID: {conversation.conversation_id}")

while (prompt := input("You: ").strip()).lower() != "exit":
    if not prompt:
        continue

    response, conversation, _ = run_agent(
        prompt=prompt,
        model=MODEL,
        agent_tools=tools,
        conversation=conversation,
        repository=repository,
        compaction_token_limit=COMPACTION_TOKEN_LIMIT,
    )
    print(response)

