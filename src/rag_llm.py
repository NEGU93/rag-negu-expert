from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

MODEL = "gpt-4o-mini"


def longchain_magic(vectorstore):
    # create a new Chat with OpenAI
    llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
    # llm = ChatOpenAI(
    #     temperature=0.7,
    #     model_name="llama3.2",
    #     base_url="http://localhost:11434/v1",
    #     api_key="ollama",
    # )

    system_prompt = """You are an AI Assistant specialized in providing accurate information about Jose Agustin BARRACHINA, a.k.a. Agustin, NEGU or Jose. Only respond when the question explicitly asks for information. 
Keep your answers factual, and based solely on the information provided. Do not speculate or fabricate details. But try to provide as much information as possible based on the context provided.
You will be given information about Jose Agustin BARRACHINA in the form of documents.
Use the provided documents to answer questions about him. You can assume all documents are relevant to him.
If the question does not pertain to Jose Agustin BARRACHINA, politely inform the user that you can only provide information about him.
If the question is not related to Jose Agustin BARRACHINA, respond with:
"I can only provide information about Jose Agustin BARRACHINA. Please ask a question related to him."
If you are unsure about who they asked about, answer the question as if they were asking about Jose Agustin BARRACHINA. Also, rembember he can just say Agustin or Jose to refer to Jose Agustin BARRACHINA.
For example, if they ask about "Agustin" or say other words like "him" answer as if they were asking about Jose Agustin BARRACHINA.
Context: {context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=system_prompt,
    )

    # set up the conversation memory for the chat
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    # memory.chat_memory.messages.insert(0, SystemMessage(content=system_prompt))

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
