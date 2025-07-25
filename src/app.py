import os
import gradio as gr
from dotenv import load_dotenv
from src.rag_llm import longchain_magic
from src.chunking import init_db


def chat(question, history):
    result = conversation_chain.invoke({"question": question})
    return result["answer"]


load_dotenv(override=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
vectorstore = init_db()
conversation_chain = longchain_magic(vectorstore)
view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)
