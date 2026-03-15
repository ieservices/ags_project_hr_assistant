"""
Nestlé HR Assistant Chatbot
============================
A conversational chatbot that answers HR policy questions using RAG
(Retrieval Augmented Generation) over the Nestlé HR Policy PDF.

Stack:
- OpenAI GPT-4o for answer generation
- OpenAI text-embedding-3-small for document embeddings
- FAISS for vector similarity search
- LangChain (LCEL) for the RAG pipeline
- Gradio for the chat UI
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import gradio as gr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. "
        "Create a .env file based on .env.example and add your key."
    )

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
pdf_files = list(DATASET_DIR.glob("*.pdf"))
if not pdf_files:
    raise FileNotFoundError(f"No PDF found in {DATASET_DIR}")
PDF_PATH = str(pdf_files[0])

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 5


# ---------------------------------------------------------------------------
# Step 1 – Load and split the PDF
# ---------------------------------------------------------------------------

def load_and_split_pdf(pdf_path: str):
    """Load PDF pages and split into overlapping text chunks."""
    print(f"[1/3] Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"      Loaded {len(pages)} pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    print(f"      Created {len(chunks)} text chunks.")
    return chunks


# ---------------------------------------------------------------------------
# Step 2 – Build the FAISS vector store
# ---------------------------------------------------------------------------

def build_vector_store(chunks):
    """Embed each chunk and store in an in-memory FAISS index."""
    print("[2/3] Building FAISS vector store with OpenAI embeddings...")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("      Vector store ready.")
    return vector_store


# ---------------------------------------------------------------------------
# Step 3 – Build the RAG chain (LCEL)
# ---------------------------------------------------------------------------

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and a follow-up question, rephrase the follow-up "
     "question into a self-contained standalone question. Return ONLY the "
     "rephrased question, nothing else."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are a professional HR assistant for Nestlé.
Answer the employee's question using ONLY the context from the HR policy document below.
If the answer is not found in the document, say so clearly rather than guessing.
Be concise, empathetic, and professional. Mention the relevant policy section when helpful.

HR Policy Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vector_store):
    """Create a conversational RAG chain using LCEL."""
    print("[3/3] Initialising RAG chain...")
    llm = ChatOpenAI(
        model_name=CHAT_MODEL,
        temperature=0.2,
        openai_api_key=OPENAI_API_KEY,
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K},
    )

    # Sub-chain: condense follow-up question → standalone question
    condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()

    def get_standalone_question(inputs: dict) -> str:
        """Return a standalone question, condensing if there is history."""
        if inputs.get("chat_history"):
            return condense_chain.invoke(inputs)
        return inputs["question"]

    # Full RAG chain
    rag_chain = (
        RunnablePassthrough.assign(
            standalone_question=get_standalone_question
        )
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(
                retriever.invoke(x["standalone_question"])
            ),
            source_docs=lambda x: retriever.invoke(x["standalone_question"]),
        )
        | {
            "answer": QA_PROMPT | llm | StrOutputParser(),
            "source_docs": lambda x: x["source_docs"],
        }
    )

    print("      Chain ready.\n")
    return rag_chain


# ---------------------------------------------------------------------------
# Initialise everything at startup
# ---------------------------------------------------------------------------

print("=" * 60)
print("  Nestlé HR Assistant – starting up")
print("=" * 60)
chunks = load_and_split_pdf(PDF_PATH)
vector_store = build_vector_store(chunks)
rag_chain = build_rag_chain(vector_store)
print("  Ready to answer HR questions!")
print("=" * 60)

# In-memory conversation history (list of LangChain message objects)
chat_history: list = []


# ---------------------------------------------------------------------------
# Gradio chat interface
# ---------------------------------------------------------------------------

def chat(user_message: str, history: list):
    """
    Gradio chat callback.

    Parameters
    ----------
    user_message : str  – latest message from the user.
    history : list[tuple[str, str]]  – Gradio conversation history.

    Returns
    -------
    tuple[str, list]  – clears input box, returns updated history.
    """
    global chat_history

    if not user_message.strip():
        return "", history

    result = rag_chain.invoke({
        "question": user_message,
        "chat_history": chat_history,
    })

    answer = result["answer"]

    # Append source page citations
    source_docs = result.get("source_docs", [])
    if source_docs:
        pages = sorted({
            doc.metadata.get("page", None)
            for doc in source_docs
            if isinstance(doc.metadata.get("page"), int)
        })
        if pages:
            page_str = ", ".join(str(p + 1) for p in pages)
            answer += f"\n\n*Source: HR Policy document, page(s) {page_str}*"

    # Update LangChain message history
    chat_history.append(HumanMessage(content=user_message))
    chat_history.append(AIMessage(content=answer))

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer})
    return "", history


def clear_chat():
    """Reset conversation memory and clear the UI."""
    global chat_history
    chat_history = []
    return []


# ---------------------------------------------------------------------------
# Build the Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Nestlé HR Assistant") as demo:

    gr.Markdown(
        """
        # Nestlé HR Assistant
        Ask any question about Nestlé's HR policies. The assistant searches
        the official HR Policy document and provides accurate, referenced answers.

        > **Note:** Answers are based solely on *The Nestlé HR Policy (2012)*.
        """
    )

    chatbot = gr.Chatbot(
        label="Conversation",
        height=480,
    )

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Ask an HR question, e.g. 'What is Nestlé's policy on harassment?'",
            label="Your Question",
            scale=9,
            autofocus=True,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    clear_btn = gr.Button("Clear Conversation", variant="secondary")

    gr.Examples(
        examples=[
            "What is Nestlé's policy on equal opportunity and non-discrimination?",
            "How does Nestlé handle harassment in the workplace?",
            "What are the components of total rewards at Nestlé?",
            "How does Nestlé support employee development and training?",
            "What are the working conditions and work-life balance policies?",
            "How does Nestlé define its recruitment and hiring process?",
        ],
        inputs=msg_box,
        label="Example Questions",
    )

    # Wire up events
    send_btn.click(fn=chat, inputs=[msg_box, chatbot], outputs=[msg_box, chatbot])
    msg_box.submit(fn=chat, inputs=[msg_box, chatbot], outputs=[msg_box, chatbot])
    clear_btn.click(fn=clear_chat, outputs=[chatbot])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="blue"),
    )
