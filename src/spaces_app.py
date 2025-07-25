import gradio as gr
from dotenv import load_dotenv
from src.rag_llm import longchain_magic
from src.chunking import init_db
import os

load_dotenv(override=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def init_chain():
    vectorstore = init_db()
    return longchain_magic(vectorstore)


with gr.Blocks() as demo:
    gr.Markdown("# 🔑 RAG Chatbot with Your OpenAI Key")

    def store_api_key(api_key):
        os.environ["OPENAI_API_KEY"] = api_key
        return gr.Textbox.update(visible=False), gr.Text.update(
            value="✅ OpenAI KEY provided", visible=True
        )

    if os.getenv("OPENAI_API_KEY") is None:
        api_key_input = gr.Textbox(
            label="Enter your OpenAI API Key",
            type="password",
            interactive=True,
        )
        confirmation_msg = gr.Text(value="", visible=False)
        api_key_input.submit(
            fn=store_api_key,
            inputs=[api_key_input],
            outputs=[api_key_input, confirmation_msg],
        )
    else:
        confirmation_msg = gr.Text(
            value="✅ OpenAI API Key already set",
            visible=True,
        )

    chatbot = gr.Chatbot()
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
        if os.getenv("OPENAI_API_KEY") is None:
            return (
                chat_history
                + [[user_message, "❌ Please enter your API key"]],
                "",
                chain,
            )

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
