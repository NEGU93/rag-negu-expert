import gradio as gr
from dotenv import load_dotenv
from src.rag_llm import longchain_magic
from src.chunking import init_db
import os

load_dotenv(override=True)
if os.getenv("OPENAI_API_KEY") is not None:
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def init_chain():
    vectorstore = init_db()
    return longchain_magic(vectorstore)


with gr.Blocks(title="RAG Chatbot") as demo:
    gr.Markdown("# 🔑 RAG Chatbot with Your OpenAI Key")

    if os.getenv("OPENAI_API_KEY") is not None:
        confirmation_msg = gr.Text(
            value="✅ OpenAI API Key already set",
            visible=True,
        )

    chatbot = gr.Chatbot(height="60vh")
    msg = gr.Textbox(
        label="Type your message",
        placeholder="Ask me anything about Agustin...",
        interactive=True,
        lines=1,
    )
    submit_btn = gr.Button("Send")
    history = gr.State([])  # Saves chat history
    chain_state = gr.State(None)  # Saves the chain instance

    def user_submit(user_message, chat_history, chain):
        if chain is None:
            try:
                chain = init_chain()
            except Exception as e:
                return (
                    chat_history
                    + [
                        [
                            user_message,
                            f"❌ Error initializing chain: {str(e)}",
                        ]
                    ],
                    "",
                    None,
                )

        result = chain.invoke({"question": user_message})
        chat_history.append([user_message, result["answer"]])

        return gr.update(value=chat_history), "", chain

    submit_btn.click(
        fn=user_submit,
        inputs=[msg, history, chain_state],
        outputs=[chatbot, msg, chain_state],
    )
    msg.submit(
        fn=user_submit,
        inputs=[msg, history, chain_state],
        outputs=[chatbot, msg, chain_state],
    )

demo.launch()
