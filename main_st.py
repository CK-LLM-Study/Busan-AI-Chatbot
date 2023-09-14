# LangChain RetrievalQA, ChromaDB, Streamlt

import streamlit as st
import os
from streamlit_chat import message

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from collections import Counter
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = "OpenAI_API_KEY"

loader = DirectoryLoader('./data', glob="*.txt", loader_cls=TextLoader)
documents = loader.load()
# print('문서의 개수 :', len(documents))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
# print('분할된 텍스트의 개수 :', len(texts))

# source_lst = []
# for i in range(0, len(texts)):
#   source_lst.append(texts[i].metadata['source'])

# element_counts = Counter(source_lst)
# filtered_counts = {key: value for key, value in element_counts.items() if value >= 2}
# print('2개 이상으로 분할된 문서 :', filtered_counts)
# print('분할된 텍스트의 개수 :', len(documents) + len(filtered_counts))

embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding)

retriever = vectordb.as_retriever()

# docs = retriever.get_relevant_documents("신혼 부부를 위한 정책이 있어?")
# print('유사 문서 개수 :', len(docs))
# print('--' * 20)
# print('첫번째 유사 문서 :', docs[0])
# print('--' * 20)
# print('각 유사 문서의 문서 출처 :')
# for doc in docs:
#     print(doc.metadata["source"])

retriever = vectordb.as_retriever(search_kwargs={"k": 2})
# for doc in docs:
#     print(doc)
#     print(doc.metadata["source"])
    
prompt_template = """You are a personal ChatBot assistant for answering any question about documents.
You are given a question and a set of documents.
If the user's question requires you provide specific information from documents, give your answer based on the examples provided below.
DON'T generate an answer that is NOT written in the provided examplse.
If you don't find the answer to the question with the examples provided to you below, answer that you didn't find the answer in the documentation
and propose him to rephrase his query qith more details.
Use bullet poings if you have to make a list, only if necessary. You shoult answer by Korean langage.

QUESTION: {question}

DOCEMENTS:
==========
{context}
==========
Finish by proposing your hlep for anything else.
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# input_text = "대출과 관련된 정책이 궁금합니다"
# chatbot_response = qa_chain(input_text)
# print(chatbot_response['result'])

### Streamlit
st.image('images/ask_me_chatbot.png')

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('정책을 물어보세요!', '', key='input')
    submitted = st.form_submit_button('Send')

if submitted and user_input:
    # 프롬프트 생성 후 프롬프트를 기반으로 챗봇의 답변을 반환
    chatbot_response = qa_chain(user_input)
    st.session_state['past'].append(user_input)
    st.session_state["generated"].append(chatbot_response['result'])

if st.session_state['generated']:
    for i in reversed(range(len(st.session_state['generated']))):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))