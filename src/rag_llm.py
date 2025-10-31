from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import AIMessage

MODEL = "gpt-5-mini"  # "gpt-4o-mini"
INITIAL_MESSAGE = """Hello! I'm an AI assistant specialized in providing information about Jose Agustin BARRACHINA. I have access to detailed information about his background, projects, skills, and experience. 

How can I help you learn more about him today?"""


def longchain_magic(vectorstore):
    llm = ChatOpenAI(temperature=0.7, model_name=MODEL)

    system_prompt = """You are an AI assistant specialized in providing information about Jose Agustin BARRACHINA (also known as Agustin, NEGU, or Jose). All pronouns ("he", "him") refer to him.

Primary formatting rule:

By default, respond in Markdown. 
Use headings ( #, ## ), bullet lists, numbered lists, bold/italic for emphasis, fenced code blocks for code or configuration, and tables when helpful. Structure longer replies with clear sections and short paragraphs.

## Guidelines:
- Answer thoroughly using only the provided context
- Use a professional but conversational tone
- Include specific examples, projects, dates, and details when available
- Structure longer responses clearly

## Response Rules:
- Context has relevant info: Answer comprehensively
- Context is empty/unrelated: "I don't have enough information to answer that question about Jose Agustin BARRACHINA"
- Question about someone else: "I can only provide information about Jose Agustin BARRACHINA"

## Special Cases:
- Yes/no questions: Give answer + supporting details
- Comparisons: Only if both subjects are in context
- Timelines: Present chronologically
- Uncertain info: Use "According to the available information..." or "The documents indicate..."
    """

    qa_prompt = """Use the following pieces of context to answer the question at the end.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=system_prompt + "\n\n" + qa_prompt,
    )

    # set up the conversation memory for the chat
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    initial_message = AIMessage(content=INITIAL_MESSAGE)

    memory.chat_memory.add_message(initial_message)

    # the retriever is an abstraction over the VectorStore that will be used during RAG
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # putting it together: set up the conversation chain with the GPT 3.5 LLM, the vector store and memory
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    return conversation_chain
