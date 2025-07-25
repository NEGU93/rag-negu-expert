import gradio as gr
from dotenv import load_dotenv
from src.rag_llm import longchain_magic
from src.chunking import init_db
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(override=True)
if os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def init_chain():
    """Initialize the RAG chain with error handling."""
    try:
        vectorstore = init_db()
        return longchain_magic(vectorstore)
    except Exception as e:
        logger.error(f"Failed to initialize chain: {e}")
        raise


def validate_api_key(api_key):
    """Validate OpenAI API key format."""
    if not api_key:
        return False, "API key cannot be empty"
    if not api_key.startswith(("sk-", "sk-proj-")):
        return False, "Invalid API key format"
    if len(api_key) < 20:
        return False, "API key appears too short"
    return True, "Valid"


with gr.Blocks(title="RAG Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔑 RAG Chatbot with Your OpenAI Key")
    gr.Markdown("Ask questions about Agustin using RAG-powered AI responses.")

    def store_api_key(api_key):
        """Store and validate the API key."""
        is_valid, message = validate_api_key(api_key)
        if not is_valid:
            return gr.update(), gr.update(value=f"❌ {message}", visible=True)

        os.environ["OPENAI_API_KEY"] = api_key.strip()
        logger.info("API key successfully stored")
        return gr.update(visible=False), gr.update(
            value="✅ OpenAI API Key provided successfully", visible=True
        )

    # API Key section
    api_key_provided = bool(os.getenv("OPENAI_API_KEY"))

    with gr.Row():
        if not api_key_provided:
            api_key_input = gr.Textbox(
                label="Enter your OpenAI API Key",
                type="password",
                placeholder="sk-proj-... or sk-...",
                interactive=True,
                info="Your API key is only stored locally and not transmitted elsewhere",
            )
            confirmation_msg = gr.Text(value="", visible=False)
            api_key_input.submit(
                fn=store_api_key,
                inputs=[api_key_input],
                outputs=[api_key_input, confirmation_msg],
            )
        else:
            confirmation_msg = gr.Text(
                value="✅ OpenAI API Key already configured",
                visible=True,
            )

    # Chat interface
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                height=500,
                show_copy_button=True,
                bubble_full_width=False,
                show_share_button=False,
            )

        with gr.Column(scale=1):
            gr.Markdown("### 💡 Tips")
            gr.Markdown("""
            - Ask specific questions about Agustin
            - Be clear and concise
            - Wait for responses to complete
            """)

    with gr.Row():
        msg = gr.Textbox(
            label="Your message",
            placeholder="Ask me anything about Agustin...",
            interactive=True,
            lines=2,
            max_lines=10,
            scale=4,
        )
        submit_btn = gr.Button("Send 📤", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary")

    # State management
    history = gr.State([])
    chain_state = gr.State(None)

    def user_submit(user_message, chat_history, chain):
        """Handle user message submission with comprehensive error handling."""
        if not user_message.strip():
            return chat_history, "", chain

        if not os.getenv("OPENAI_API_KEY"):
            error_response = "❌ Please enter your OpenAI API key first"
            chat_history.append([user_message, error_response])
            return chat_history, "", chain

        # Initialize chain if needed
        if chain is None:
            try:
                with gr.Info("Initializing RAG system..."):
                    chain = init_chain()
                logger.info("Chain initialized successfully")
            except Exception as e:
                error_msg = f"❌ Failed to initialize RAG system: {str(e)}"
                logger.error(error_msg)
                chat_history.append([user_message, error_msg])
                return chat_history, "", None

        # Process the question
        try:
            with gr.Info("Processing your question..."):
                result = chain.invoke({"question": user_message.strip()})
                response = result.get("answer", "No answer provided")
                chat_history.append([user_message, response])
                logger.info(
                    f"Successfully processed question: {user_message[:50]}..."
                )

        except Exception as e:
            error_msg = f"❌ Error processing question: {str(e)}"
            logger.error(error_msg)
            chat_history.append([user_message, error_msg])

        return chat_history, "", chain

    def clear_chat():
        """Clear the chat history."""
        logger.info("Chat cleared by user")
        return [], []

    # Event handlers
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

    clear_btn.click(fn=clear_chat, outputs=[chatbot, history])

    # Update chatbot display when history changes
    history.change(lambda h: h, inputs=[history], outputs=[chatbot])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False,  # Set to True if you want a public link
    )
