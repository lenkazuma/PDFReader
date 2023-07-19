from dotenv import load_dotenv
import streamlit as st
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from docx import Document
from docx.table import _Cell

def extract_text_from_table(table):
    text = ""
    for row in table.rows:
        for cell in row.cells:
            if isinstance(cell, _Cell):
                text += cell.text + "\n"
    return text.strip()


def main():
    # brief summary
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")

    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.title("EEC PDFReader ✨")
    
    # upload file
    uploaded_file  = st.file_uploader("Upload your PDF", type=["pdf", "docx"])
    
    # Initialize session state
    if 'pdf_name' not in st.session_state:
        st.session_state.pdf_name = None
    
    # extract the text
    if uploaded_file  is not None :
        file_type = uploaded_file.type

        # Clear summary if a new file is uploaded
        if 'summary' in st.session_state and st.session_state.file_name != uploaded_file.name:
            st.session_state.summary = None

        st.session_state.file_name = uploaded_file.name

        
        try:
            if file_type == "application/pdf":
                # Handle PDF files
                
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                # Handle Word documents
                doc = Document(uploaded_file)
                paragraphs = [p.text for p in doc.paragraphs]
                text = "\n".join(paragraphs)

                # Extract text from tables
                for table in doc.tables:
                    table_text = extract_text_from_table(table)
                    if table_text:
                        text += "\n" + table_text
                        
            else:
                st.error("Unsupported file format. Please upload a PDF or DOCX file.")
                return

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

            
            st.header("Here's a brief summary of your file:")
            pdf_summary = "Give me a brief summary,use the language that the file is in"
 
            docs = knowledge_base.similarity_search(pdf_summary)
            
            
            if 'summary' not in st.session_state or st.session_state.summary is None:
              with st.spinner('Wait for it...'):
                st.session_state.summary = chain.run(input_documents=docs, question=pdf_summary)
            st.write(st.session_state.summary)


            # show user input
            user_question = st.text_input("Ask a question about your file :")
            if user_question:
                docs = knowledge_base.similarity_search(user_question)
                with st.spinner('Wait for it...'):
                  with get_openai_callback() as cb:
                     response = chain.run(input_documents=docs, question=user_question)
                     print(cb)
                     # show/hide section using st.beta_expander
                     with st.expander("Used Tokens", expanded=False):
                       st.write(cb)
                st.write(response)
                
        #except IndexError:
            #st.caption("Well, Seems like your PDF doesn't contain any text, try another one.🆖")
            #st.error("Please upload another PDF. This PDF does not contain any text.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")



if __name__ == '__main__':
    main()
