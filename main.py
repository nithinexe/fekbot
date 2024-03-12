import openai
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
# import google.generativeai as genai Change it to openai embeddings 

from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI Change to openai 
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


import streamlit as st 
import openai
st.set_page_config(page_title="Chadukune Bot", layout="wide")
# st.write("stream lit check")

st.markdown("""## Chaduko bro bagupadthav....""")

# pdf_docs = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text 
    
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.environ.get("OPENAI_API_KEY"))
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

from openai import Client, Completion, OpenAI
import os 



def get_conversational_chain(context, user_question):
    client = Client(api_key=os.environ.get("OPENAI_API_KEY"))
    prompt_template = "Please answer the question with reference to the document provided."

    # Incorporate the document context and the user question into the prompt
    prompt = f"Document: {context}\nQuestion: {user_question}\n{prompt_template}"

    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",  # Use an appropriate model
        prompt=prompt,
        temperature=0.5,  # Adjust temperature as needed
        max_tokens=550  # Adjust max_tokens as needed
    )

    return response.choices[0].text


# ignore this shitty functions
# def get_conversational_chain():
#     client = Client(api_key=os.environ.get("OPENAI_API_KEY"))
#     prompt_template = "answer with reference to the doc"
    
#     # Example context and question
#     context = "This is an example context."
#     question = "What is the example question?"
#     prompt = f"Context: {context}\nQuestion: {question}\n{prompt_template}"
    
#     # Correctly use the completions.create method on the client instance
#     response = client.completions.create(
#         model="gpt-3.5-turbo-instruct",  # Specify the model you want to use
#         prompt=prompt,
#         temperature=0.7,
#         max_tokens=150
#     )
    
#     # Assuming you want to process the response further or return it
#     return response.choices[0].text  # Adjusted to access the text of the first choice

# Assuming get_conversational_chain is defined as before and returns a string response from the OpenAI API

def user_input(user_question, document_text):
    # This part remains unchanged
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",api_key=os.environ.get("OPENAI_API_KEY"))
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    # Get the response from the OpenAI API
    response_text = get_conversational_chain(document_text, user_question)
    
    # Now, instead of trying to call `response_text` as a function,
    # you should process it according to your application's needs.
    # For example, you might log it, display it, or use it to inform further processing.
    st.write("Reply: ", response_text) 




# def user_input(user_question):
#     embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",api_key=os.environ.get("OPENAI_API_KEY"))
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     st.write("Reply: ", response["output_text"]) 

# st.write("stream lit check")
api_key = os.environ.get("OPENAI_API_KEY")

def main():

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")
    document_text=""

    if user_question and api_key:  # Ensure API key and user question are provided
        user_input(user_question, document_text)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Done")

if __name__ == "__main__":
    main()
