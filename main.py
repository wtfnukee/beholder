from pyrogram import Client, filters
from queue import Queue
from collections import deque
from yandex_cloud_ml_sdk import YCloudML
from langchain_core.messages import SystemMessage, HumanMessage
import json
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

# A queue to hold incoming messages and a deque for last N messages
message_queue = Queue()
chat_messages = {}  # Dict to store deque for each chat
MESSAGES_PER_CHAT = 2
# Store events by chat_id
events_by_chat = {}  # Dict to store events for each chat

sdk = YCloudML(
    folder_id=os.getenv("YANDEX_FOLDER_ID"), auth=os.getenv("YANDEX_AUTH_TOKEN")
)
model = sdk.models.completions("yandexgpt").langchain()


app = Client(
    "name",
    api_id=os.getenv("TELEGRAM_API_ID"),
    api_hash=os.getenv("TELEGRAM_API_HASH"),
)


def format_events_context(chat_id):
    """Format existing events for the chat as context"""
    if chat_id not in events_by_chat or not events_by_chat[chat_id]:
        return "No previous events recorded."

    events_str = "Previously recorded events:\n"
    for idx, event in enumerate(events_by_chat[chat_id], 1):
        events_str += f"{idx}. Event: {event['event']}, Type: {event['type']}, Time: {event['time']}\n"
    return events_str


SYSTEM_PROMPT_TEMPLATE = """You are an event analyzer and manager that processes conversation messages and manages a calendar of events.
You will receive the current conversation messages and a list of previously recorded events.

Your task is to:
1. Analyze new messages for events
2. Check for conflicts or updates with existing events
3. Return a JSON object with the following fields:
   - action: "add" (new event), "update" (modify existing), "delete" (remove event), or "none" (no action needed)
   - event: brief description of the event
   - type: categorization (e.g., "meeting", "task", "reminder", "social", "question")
   - time: specific time/date or "none". Convert relative times (e.g., "tomorrow") to absolute dates
   - conflicts: array of indices of conflicting events from the previous events list (empty if none)
   - update_index: if updating/deleting, index of event to modify (from previous events list)

Current time is: {current_time}

{events_context}

Example response:
{{
    "action": "add",
    "event": "team meeting discussion",
    "type": "meeting",
    "time": "18:00 22.01.2025",
    "conflicts": [2],
    "update_index": null
}}
"""


def process_messages_through_llm(chat_id):
    if chat_id not in chat_messages or len(chat_messages[chat_id]) == 0:
        return None

    current_time = datetime.now().strftime("%d.%m.%Y %H:%M")
    events_context = format_events_context(chat_id)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        current_time=current_time, events_context=events_context
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content="\n".join([msg.text for msg in chat_messages[chat_id] if msg.text])
        ),
    ]

    try:
        response = model.invoke(messages)
        event_data = json.loads(response.content[3:-3])

        # Initialize events list for this chat if it doesn't exist
        if chat_id not in events_by_chat:
            events_by_chat[chat_id] = []

        if event_data["action"] == "add":
            events_by_chat[chat_id].append(
                {
                    "event": event_data["event"],
                    "type": event_data["type"],
                    "time": event_data["time"],
                    "processed_at": datetime.now().isoformat(),
                }
            )
            action_msg = "New event added!"
        elif (
            event_data["action"] == "update" and event_data["update_index"] is not None
        ):
            idx = event_data["update_index"] - 1  # Convert to 0-based index
            if 0 <= idx < len(events_by_chat[chat_id]):
                events_by_chat[chat_id][idx].update(
                    {
                        "event": event_data["event"],
                        "type": event_data["type"],
                        "time": event_data["time"],
                        "updated_at": datetime.now().isoformat(),
                    }
                )
                action_msg = "Event updated!"
        elif (
            event_data["action"] == "delete" and event_data["update_index"] is not None
        ):
            idx = event_data["update_index"] - 1
            if 0 <= idx < len(events_by_chat[chat_id]):
                del events_by_chat[chat_id][idx]
                action_msg = "Event deleted!"
        else:
            action_msg = "No action needed."

        # Prepare conflict warnings
        conflict_msg = ""
        if event_data["conflicts"]:
            conflict_indices = [str(i) for i in event_data["conflicts"]]
            conflict_msg = f"\n⚠️ Conflicts with events: {', '.join(conflict_indices)}"

        # Send confirmation message back to the chat
        confirmation_msg = (
            f"{action_msg}\n"
            f"Event: {event_data['event']}\n"
            f"Type: {event_data['type']}\n"
            f"Time: {event_data['time']}"
            f"{conflict_msg}"
        )
        print(confirmation_msg)
        app.send_message(chat_id, confirmation_msg)

        return event_data
    except Exception as e:
        print(f"Error processing messages: {e}")
        return None


@app.on_message(filters.incoming)
def handle_new_message(client, message):
    if not message.text:
        return

    chat_id = message.chat.id

    # Initialize deque for new chats
    if chat_id not in chat_messages:
        chat_messages[chat_id] = deque(maxlen=MESSAGES_PER_CHAT)

    # Add the message to both queue and chat-specific deque
    message_queue.put(message)
    chat_messages[chat_id].append(message)
    print(f"New message from {chat_id}: {message.text}")

    # Process messages when we have enough from this chat
    if len(chat_messages[chat_id]) == MESSAGES_PER_CHAT:
        process_messages_through_llm(chat_id)


if __name__ == "__main__":
    print("Starting Pyrogram app...")
    app.run()
