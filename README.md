# PDF & Word Reader

LangChain Streamlit PDF & Word Reader

## Acknowledge

This project is built by replicating the functionality of [Alejandro AO's langchain-ask-pdf](https://github.com/alejandro-ao/langchain-ask-pdf) (also check out [his tutorial on YouTube](https://www.youtube.com/watch?v=wUAUdEw5oxM)).

I added some custom sections that I needed, error handling, summary prompt, callback token used count, and most importantly, worked around the max token limit. Now it uses map-reduce as well as RecursiveCharacterTextSplitter.

## Installation

To install the necessary requirements, run the following command:

- Install requirements : `pip install -r requirements.txt`


## Deployment

I deployed it on Streamlit Cloud. You can check it out [PDFReader-LangChain](https://pdfreader-langchain.streamlit.app/). Please note that "langchain" was misspelled in the URL.

If you want to deploy your own instance, sign up for Streamlit and follow their deployment instructions. Remember to add your OpenAI API key in the "APP Settings -> Secrets" section.

