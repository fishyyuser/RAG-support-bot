from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from logger import setup_logger

logger = setup_logger()


class RAGPipeline:
    def __init__(self, vectorstore_path="vectorstore"):
        try:
            logger.info("Initializing RAG pipeline...")

            # Embeddings
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

            # Load FAISS index
            logger.info("Loading FAISS vector store...")
            self.db = FAISS.load_local(
                vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            # LLM (stable + cheap)
            logger.info("Initializing LLM...")
            self.llm = ChatOpenAI(
                model="gpt-5.4-nano",
                temperature=0
            )

            logger.info("RAG pipeline initialized successfully.")

        except Exception as e:
            logger.exception(f"Initialization failed: {str(e)}")
            raise

    def query(self, question: str, k: int = 3):
        try:
            logger.info(f"Received query: {question}")

            # Step 1: Retrieve relevant docs
            docs = self.db.similarity_search(question, k=k)

            if not docs:
                logger.warning("No relevant documents found.")
                return "I couldn't find relevant information."

            logger.info(f"Retrieved {len(docs)} relevant chunks")

            # Debug: log chunk previews
            for i, doc in enumerate(docs):
                preview = doc.page_content[:200].replace("\n", " ")
                logger.info(f"Chunk {i+1}: {preview}...")

            # Step 2: Build context
            context = "\n\n".join([doc.page_content for doc in docs])

            # Step 3: Prompt
            prompt = f"""
You are a helpful assistant. Answer ONLY using the provided context.

Rules:
- Do not use outside knowledge
- If the answer is not in the context, say "I don't know"
- Be concise and accurate

Context:
{context}

Question:
{question}

Answer:
"""

            # Step 4: Generate response
            response = self.llm.invoke(prompt)

            logger.info("Response generated successfully.")

            return response.content

        except Exception as e:
            logger.exception(f"Query failed: {str(e)}")
            return "Error processing request."


# CLI testing
if __name__ == "__main__":
    rag = RAGPipeline()

    while True:
        q = input("\nAsk a question (type 'exit' to quit): ")
        if q.lower() == "exit":
            break

        answer = rag.query(q)
        print("\nAnswer:", answer)