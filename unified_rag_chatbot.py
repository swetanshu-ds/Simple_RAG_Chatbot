import os
import re
import requests
from datetime import datetime
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate

# Load PDF knowledge base
pdf_path = "D:/Class/assignment_cyfuture/Assignment_AI_Cyf.pdf"
pdf_loader = PyPDFLoader(pdf_path)
documents = pdf_loader.load()

# Create embeddings and FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_db = FAISS.from_documents(documents, embeddings)

# Load T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
hf_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=hf_pipe)

# Prompt for RAG
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

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=faiss_db.as_retriever(),
    chain_type_kwargs={"prompt": custom_prompt}
)

# Complaint details collector with memory
class ComplaintSession:
    def __init__(self):
        self.fields = ["name", "phone_number", "email", "complaint_details"]
        self.data = {}
        self.current_field_index = 0

    def is_complete(self):
        return all(field in self.data for field in self.fields)

    def next_prompt(self):
        field = self.fields[self.current_field_index]
        prompts = {
            "name": "Please provide your name:",
            "phone_number": "What is your phone number?",
            "email": "Please provide your email address:",
            "complaint_details": "Can you describe your complaint in detail?"
        }
        return prompts[field]

    def receive_input(self, user_input):
        field = self.fields[self.current_field_index]
        self.data[field] = user_input
        self.current_field_index += 1

    def submit_complaint(self):
        response = requests.post("http://127.0.0.1:8000/complaints", json=self.data)
        if response.status_code == 200:
            complaint_id = response.json().get("complaint_id")
            return f"Your complaint has been registered with ID: {complaint_id}"
        else:
            return "Failed to create complaint. Please try again."

# Chatbot session
if __name__ == "__main__":
    print("Bot: Hello! How can I help you today?")
    complaint_session = None

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            print("Bot: Goodbye!")
            break

        # Check for complaint ID in user query
        match = re.search(r'\b([A-Z0-9]{8})\b', user_input)
        print(user_input)
        print(match)
        if match:
            complaint_id = match.group(1)
            print(complaint_id)
            response = requests.get(f"http://127.0.0.1:8000/complaints/{complaint_id}")
            if response.status_code == 200:
                data = response.json()
                print(
                    f"Bot: Complaint ID: {data['complaint_id']}\n"
                    f"Name: {data['name']}\n"
                    f"Phone: {data['phone_number']}\n"
                    f"Email: {data['email']}\n"
                    f"Details: {data['complaint_details']}\n"
                    f"Created At: {data['created_at']}"
                )
            else:
                print("Bot: Sorry, we couldn't find the complaint ID. Please check again.")
            continue

        # Start complaint session if needed
        if any(x in user_input.lower() for x in ["complaint", "report issue", "file issue","complain","problem"]):
            complaint_session = ComplaintSession()
            print(f"Bot: I'm sorry to hear that. {complaint_session.next_prompt()}")
            continue

        # Handle ongoing complaint session
        if complaint_session:
            complaint_session.receive_input(user_input)
            if complaint_session.is_complete():
                result = complaint_session.submit_complaint()
                print(f"Bot: {result}")
                complaint_session = None
            else:
                print(f"Bot: {complaint_session.next_prompt()}")
            continue

        # Else default to RAG-based response
        tokens = tokenizer.encode(user_input, return_tensors="pt")
        if len(tokens[0]) > 512:
            tokens = tokens[0][:512]
            user_input = tokenizer.decode(tokens, skip_special_tokens=True)

        response = rag_chain.invoke({"query": user_input})
        print(f"Bot: {response['result']}")
