import os
import nltk
import faiss
import numpy as np
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain import PromptTemplate
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
load_dotenv()
nltk.download('punkt')

model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("MY PDF ANSWERING MACHINE")
st.text("(Upload your PDF -- Ask a question)")
target_dir = "pdfs"

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

def generate_embeddings(text):
    # Use NLTK's sentence tokenizer
    sentences = nltk.tokenize.sent_tokenize(text)
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return embeddings, sentences

def create_faiss_index(embeddings):
    d = embeddings.shape[1]  # dimension of vectors
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.cpu().detach().numpy())  # add vectors to the index
    return index

def search_in_index(index, query, sentences, top_k=5):
    query_embedding = model.encode([query], convert_to_tensor=True)
    query_embedding_np = query_embedding.cpu().detach().numpy()
    distances, indices = index.search(query_embedding_np, top_k)

    results = [(sentences[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results

if uploaded_file:
    try:
        pdf_reader = PdfReader(uploaded_file)
        extracted_text = ""
        for page in pdf_reader.pages:
            extracted_text += page.extract_text()

        # text_splitter = RecursiveCharacterTextSplitter(
        #             # Set a really small chunk size, just to show.
        #             chunk_size = 1024,
        #             chunk_overlap  = 128,            
        #         )

        # chunks = text_splitter.create_documents([extracted_text])

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.1)

        # prompt_template = """
        # Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        # provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        # Context:\n {context}?\n
        # Question: \n{question}\n

        # Answer:
        # """
        # prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])

        # def create_chain(prompt):
        #     model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
        #     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        #     return chain

        # chain = create_chain(prompt)


        template = """
        As an expert in extracting relevant and meaningful information, your task is to process the input provided. The input contains a user's query followed by a text. They are separated by the phrase 'End of Query'. Carefully understand the query to extract the relevant information from the text.
        "Note that the text that you may have to read to extract query may be disorganized. You need to use your intelligence to understand the same and properly extract information".

        Input Provided:
        {input_str}

        Based on the user's query, provide precise information related to it from the text. Present the information in bullet points, if possible.

        \n\n"""

        prompt_template = PromptTemplate(input_variables=["input_str"], template=template)
        question_chain = LLMChain(llm=llm, prompt=prompt_template)

        overall_chain = SimpleSequentialChain(
            chains=[question_chain]
        )

        embeddings, sentences = generate_embeddings(extracted_text)

        index = create_faiss_index(embeddings)

        # text = st.text_input('Enter your question here ')
        # if text:
        #     st.write("Response : ")
        #     with st.spinner("Searching for answers ..... "):
        #         response = chain(
        #             {"input_documents":chunks, "question": text},
        #             return_only_outputs=True)
        #         st.write(response['output_text'])
        #     st.write("")


        text = st.text_input('Enter your question here ')
        if text:
            st.write("Response : ")
            with st.spinner("Searching for answers ..... "):
                results = search_in_index(index, text, sentences)
                txt = ''
                for sentence, score in results:
                    text+=sentence
                input_str = f"query: {text}\nEnd of Query\ntext: {txt}"
                result = overall_chain.run(input_str)
                st.write(result)
            st.write("")
        

    except Exception as e:
        st.write(e)
    



