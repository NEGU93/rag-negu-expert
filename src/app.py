import os
import gradio as gr
from functools import cache
from dotenv import load_dotenv
from src.wiki_query import INITIAL_MESSAGE, create_wiki_chain

load_dotenv(override=True)
if key := os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = key


@cache
def get_wiki_chat():
    """Initialize once and cache the wiki chat chain."""
    return create_wiki_chain()


def chat(question, history):
    try:
        chain = get_wiki_chat()
    except Exception as e:
        return f"⚠️ Initialization error: {e}"
    result = chain.invoke({"question": question})
    return result["answer"]


initial_history = [{"role": "assistant", "content": INITIAL_MESSAGE}]

chat_interface = gr.ChatInterface(
    chat,
    type="messages",
    chatbot=gr.Chatbot(value=initial_history, type="messages", height="70vh"),
    title="AI Expert on Jose Agustin BARRACHINA (LLM Wiki)",
    fill_height=False,
)
if __name__ == "__main__":
    try:
        chat_interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            debug=True,
        )
    except KeyboardInterrupt:
        print("\nServer stopped.")
