import os
import requests
import re
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate

# Initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

def truncate_input(input_text):
    """
    Truncate input text to max 512 tokens for T5 compatibility.
    """
    tokens = tokenizer.encode(input_text, return_tensors="pt")
    if len(tokens[0]) > 512:
        tokens = tokens[0][:512]
        input_text = tokenizer.decode(tokens, skip_special_tokens=True)
    return input_text

# Load the PDF as knowledge base
pdf_path = "D:/Class/assignment_cyfuture/Assignment_AI_Cyf.pdf"
pdf_loader = PyPDFLoader(pdf_path)
documents = pdf_loader.load()

# Create embeddings and FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_db = FAISS.from_documents(documents, embeddings)

# Load T5 model and tokenizer for response generation
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
hf_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=hf_pipe)

# Define a custom prompt to guide the model better
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Use the following context to answer the user's question. If you don't know the answer, just say "I don't know".

    Context:
    {context}

    Question:
    {question}
    """
)

# Setup RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=faiss_db.as_retriever(),
    chain_type_kwargs={"prompt": custom_prompt}
)

# Chatbot logic
def ask_chatbot(user_input):
    # Handle complaint ID
    print(user_input)
    if user_input.lower():
        match = re.search(r'\b([A-Z0-9]{8})\b', user_input)
        print(match)
        if match:
            complaint_id = match.group(1)
            # (your API call and logic here)
            print(complaint_id)
            response = requests.get(f"http://127.0.0.1:8000/complaints/{complaint_id}")
            if response.status_code == 200:
                data = response.json()
                return (
                    f"Complaint ID: {data['complaint_id']}\n"
                    f"Name: {data['name']}\n"
                    f"Phone: {data['phone_number']}\n"
                    f"Email: {data['email']}\n"
                    f"Details: {data['complaint_details']}\n"
                    f"Created At: {data['created_at']}"
                )
            else:
                return "Sorry, we couldn't find the complaint ID. Please check again."

    # For general queries
    user_input = truncate_input(user_input)
    print(user_input)
    response = rag_chain.invoke({"query": user_input})
    return response["result"]

# Chat loop
if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            print("Bot: Goodbye!")
            break
        bot_response = ask_chatbot(user_input)
        print(f"Bot: {bot_response}")
