import os
os.environ["OPENAI_API_KEY"] = "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"

import re
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
# from langchain import LLMChain, PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
# from langchain.chains import ConversationalRetrievalQAChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain.memory import ConversationSummaryBufferMemory 
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationKGMemory
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import SeleniumURLLoader
from langchain.document_loaders import PlaywrightURLLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import ImageCaptionLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from utils import *


# preparation
llm = OpenAI(temperature=0)
chat = ChatOpenAI(temperature=0)
memory = ConversationSummaryBufferMemory(memory_key="chat_history", llm=llm, max_token_limit=10, return_messages=True)
# conversation = ConversationChain(memory=memory, prompt=chat_prompt, llm=chat)
embeddings = OpenAIEmbeddings()


urls = [
    "https://stanford-cs324.github.io/winter2022/lectures/introduction/",
    # "https://stanford-cs324.github.io/winter2022/lectures/harms-1/",
    # "https://stanford-cs324.github.io/winter2022/lectures/harms-2/",
    # "https://stanford-cs324.github.io/winter2022/lectures/capabilities/",
]
table = ["https://github.com/Hannibal046/Awesome-LLM#milestone-papers"]
table_loader = get_table_metadata(table)[0]


loader = UnstructuredURLLoader(urls=urls)
# loader = SeleniumURLLoader(urls=urls)
# loader = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"])
documents = loader.load()
documents = preprocess_data(documents)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
index = VectorstoreIndexCreator().from_loaders([loader])
chat_history = []
# query = "Which model did Google release in 2018"
query = "list all models released after 2017"
# docsearch = Chroma.from_documents(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))])
vectorstore = Chroma.from_documents(documents, embeddings)

chain = load_qa_with_sources_chain(llm, chain_type="stuff")
qa = ConversationalRetrievalChain.from_llm(llm=chat, 
                                           retriever=vectorstore.as_retriever(), 
                                           memory=memory, 
                                        #    combine_docs_chain=chain,
                                           return_source_documents=True
                                           )
chat_history = []
response = qa({"question": query, "chat_history": chat_history})
print(response['answer'])

# Front end web app
import gradio as gr
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    chat_history = []
    
    def user(user_message, history):
        # Get response from QA chain
        response = qa({"question": user_message, "chat_history": history})
        # Append user message and response to chat history
        history.append((user_message, response["answer"]))
        return gr.update(value=""), history
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(debug=True)


