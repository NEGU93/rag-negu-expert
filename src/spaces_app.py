import os
import gradio as gr
from dotenv import load_dotenv
from src.rag_llm import longchain_magic
from src.chunking import init_db


def chat(question, history, api_key):
    if not api_key or not api_key.strip():
        return "Please enter your OpenAI API key to use this service."

    if not api_key.startswith("sk-"):
        return "Invalid API key format. OpenAI API keys start with 'sk-'."

    # Set the API key for this session
    os.environ["OPENAI_API_KEY"] = api_key.strip()

    try:
        # Initialize the conversation chain with the user's API key
        vectorstore = init_db()
        conversation_chain = longchain_magic(vectorstore)
        result = conversation_chain.invoke({"question": question})
        return result["answer"]
    except Exception as e:
        if (
            "invalid_api_key" in str(e).lower()
            or "unauthorized" in str(e).lower()
        ):
            return "Invalid API key. Please check your OpenAI API key and try again."
        else:
            return f"An error occurred: {str(e)}"


# Load environment variables (but don't require OPENAI_API_KEY)
load_dotenv(override=True)

# Create the interface
with gr.Blocks(title="RAG Chat Assistant") as demo:
    gr.Markdown("# RAG Chat Assistant")
    gr.Markdown(
        "Enter your OpenAI API key below to start chatting. Your key is not stored and is only used for your session."
    )

    with gr.Row():
        api_key_input = gr.Textbox(
            label="OpenAI API Key",
            placeholder="sk-...",
            type="password",
            info="Your API key is not stored and only used during your session",
        )

    chatbot = gr.ChatInterface(
        fn=chat,
        additional_inputs=[api_key_input],
        type="messages",
        title="Chat with your documents",
    )

    gr.Markdown("""
    ### How to get your OpenAI API Key:
    1. Go to [OpenAI's website](https://platform.openai.com/api-keys)
    2. Sign in to your account
    3. Navigate to API Keys section
    4. Create a new secret key
    5. Copy and paste it above
    
    **Note: Your API key is only used during your session and is not stored anywhere.**
    """)

if __name__ == "__main__":
    demo.launch()
