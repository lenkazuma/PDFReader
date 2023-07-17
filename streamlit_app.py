from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.title("Ask your PDF ✨")

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text
    if pdf is not None:
        try:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # split into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            # create embeddings
            embeddings = OpenAIEmbeddings(disallowed_special=())
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            # brief summary
            st.header("Here's a brief summary of your PDF:")
            pdf_summary = "Give me a brief summary of the pdf"

            docs = knowledge_base.similarity_search(pdf_summary)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=pdf_summary)
                print(cb)
            st.write(response)

            # show user input
            user_question = st.text_input("Ask a question about your PDF:")
            if user_question:
                docs = knowledge_base.similarity_search(user_question)

                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=user_question)
                    print(cb)
                
                st.write(response)

        except IndexError:
            st.caption("Well, Seems like your PDF doesn't contain any text, try another one.🆖")
            st.error("Please upload another PDF. This PDF does not contain any text.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
