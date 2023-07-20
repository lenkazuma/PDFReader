# PDF & Word Reader
LangChain Streamlit PDF & Word Reader

## Ackowledge
This project is build by replicating the functionality of [Alejandro AO's langchain-ask-pdf](https://github.com/alejandro-ao/langchain-ask-pdf) (also check out [his tutorial on YT](https://www.youtube.com/watch?v=wUAUdEw5oxM)). 

I added some custom sections that I need, error handling, summary prompt, callback token used count, and most of all, worked around the max token limit. So now it uses map-reduce as well as RecursiveCharacterTextSplitter. 

## Installation

- Install requirements : `pip install -r requirements.txt`

## Deployment
I deployed it on streamlit cloud, checkout [PDFReader-LongChain](https://pdfreader-longchain.streamlit.app/), misspelt langchain. 
You can simply signup streamlit and deploy your owns. Remember to add your openai API key in APP Settings -> Secret 
