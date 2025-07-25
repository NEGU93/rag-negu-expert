import gradio as gr
from src.rag_llm import longchain_magic
from src.chunking import init_db
import os


def init_chain(api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    vectorstore = init_db()  # <- Delay this until after key is set
    return longchain_magic(vectorstore)


with gr.Blocks() as demo:
    gr.Markdown("# 🔑 RAG Chatbot with Your OpenAI Key")

    api_key_input = gr.Textbox(
        label="Enter your OpenAI API Key", type="password"
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

    def user_submit(user_message, chat_history, chain, api_key):
        if not api_key:
            return (
                chat_history
                + [[user_message, "❌ Please enter your API key"]],
                "",
                chain,
            )

        if chain is None:
            try:
                chain = init_chain(api_key)
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
        inputs=[msg, history, chain_state, api_key_input],
        outputs=[chatbot, msg, chain_state],
    )
    msg.submit(
        fn=user_submit,
        inputs=[msg, history, chain_state, api_key_input],
        outputs=[chatbot, msg, chain_state],
    )

demo.launch()
