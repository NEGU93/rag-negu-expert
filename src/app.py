import os
import gradio as gr
from functools import cache
from dotenv import load_dotenv
from src.rag_llm import longchain_magic, INITIAL_MESSAGE
from src.chunking import init_db


@cache
def get_conversation_chain():
    """Initialize once and cache the result"""
    vectorstore = init_db()
    return longchain_magic(vectorstore)


def chat(question, history):
    conversation_chain = get_conversation_chain()
    result = conversation_chain.invoke({"question": question})
    return result["answer"]


initial_history = [{"role": "assistant", "content": INITIAL_MESSAGE}]

load_dotenv(override=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

chat_interface = gr.ChatInterface(
    chat,
    type="messages",
    chatbot=gr.Chatbot(value=initial_history, type="messages", height="80vh"),
    title="🤖 AI Expert on Jose Agustin BARRACHINA Assistant powered by RAG",
    fill_height=True,
)
chat_interface.launch()
