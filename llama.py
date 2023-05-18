import os
os.environ["OPENAI_API_KEY"] = "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"

from llama_index import download_loader, GPTVectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage
from pathlib import Path
from utils import *
from langchain.agents import Tool, initialize_agent

from llama_index import GPTListIndex, LLMPredictor, ServiceContext, load_graph_from_storage
from langchain import OpenAI
from llama_index.indices.composability import ComposableGraph

from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent

from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from llama_index.query_engine.transform_query_engine import TransformQueryEngine

urls = [
    "https://stanford-cs324.github.io/winter2022/lectures/introduction/",
    "https://stanford-cs324.github.io/winter2022/lectures/harms-1/",
    "https://stanford-cs324.github.io/winter2022/lectures/harms-2/",
    "https://stanford-cs324.github.io/winter2022/lectures/capabilities/",
]
table_urls = ["https://github.com/Hannibal046/Awesome-LLM#milestone-papers"]

BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")
loader = BeautifulSoupWebReader()
documents = loader.load_data(urls=urls)

# image_metadata = get_img_metadata(urls)
# all_images = []
# for image_m in image_metadata:
#     image_doc = Document(text=image_m)
#     all_images.append(image_doc)
# documents.extend(all_images)

table_loader = get_table_metadata(table_urls)
table_docs = [Document(text=t.to_string()) for t in table_loader]

doc_set = {x : [y] for x, y in enumerate(documents)}

# initialize simple vector indices + global vector index
service_context = ServiceContext.from_defaults(chunk_size_limit=512)

index_set = {}
for i in range(len(documents)):
    storage_context = StorageContext.from_defaults()
    cur_index = GPTVectorStoreIndex.from_documents(
        doc_set[i], 
        service_context=service_context,
        storage_context=storage_context,
    )
    index_set[i] = cur_index
    storage_context.persist(persist_dir=f'./storage/ind{i}')

# Load indices from disk
index_set = {}
for i in range(len(documents)):
    storage_context = StorageContext.from_defaults(persist_dir=f'./storage/ind{i}')
    cur_index = load_index_from_storage(storage_context=storage_context)
    index_set[i] = cur_index

# describe each index to help traversal of composed graph
index_summaries = [f"Lecture {i}" for i in range(len(documents))]

# define an LLMPredictor set number of output tokens
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
storage_context = StorageContext.from_defaults()

# define a list index over the vector indices
# allows us to synthesize information across each index
graph = ComposableGraph.from_indices(
    GPTListIndex,
    [index_set[y] for y in range(len(documents))], 
    index_summaries=index_summaries,
    service_context=service_context,
    storage_context=storage_context,
)
root_id = graph.root_id

# [optional] save to disk
storage_context.persist(persist_dir=f'./storage/root')

# [optional] load from disk, so you don't need to build graph from scratch
graph = load_graph_from_storage(
    root_id=root_id, 
    service_context=service_context,
    storage_context=storage_context,
)

# define a decompose transform
decompose_transform = DecomposeQueryTransform(
    llm_predictor, verbose=True
)

# define custom retrievers

custom_query_engines = {}
for index in index_set.values():
    query_engine = index.as_query_engine()
    # retriever = index.as_retriever()
    query_engine = TransformQueryEngine(
        query_engine,
        query_transform=decompose_transform,
        transform_extra_info={'index_summary': index.index_struct.summary},
    )
    custom_query_engines[index.index_id] = query_engine
custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(
    response_mode='tree_summarize',
    verbose=True,
)

# tool config
graph_config = IndexToolConfig(
    query_engine=query_engine,
    name=f"Graph Index",
    description="useful for when you want to answer queries that require analyzing multiple documents.",
    tool_kwargs={"return_direct": True}
)

# define toolkit
index_configs = []
for y in range(len(documents)):
    query_engine = index_set[y].as_query_engine(
        similarity_top_k=3,
    )
    tool_config = IndexToolConfig(
        query_engine=query_engine, 
        name=f"Vector Index {y}",
        description=f"useful for when you want to answer queries about the {y} document",
        tool_kwargs={"return_direct": True}
    )
    index_configs.append(tool_config)

toolkit = LlamaToolkit(
    index_configs=index_configs + [graph_config],
)

memory = ConversationBufferMemory(memory_key="chat_history")
llm = OpenAI(temperature=0)
agent_chain = create_llama_chat_agent(
    toolkit,
    llm,
    memory=memory,
    verbose=True,
)
# ans from agent should be a Response object for sources

# from_tool_config

chat_history = []
# query1 = "hi, i am bob"
# query2 = "What is my name"
# query3 = "If I was born in 2000, how old am I"
# response = agent_chain.run(input=query1)
# chat_history.append((query1, response))
# print(response)
# # print(response.source_nodes)
# response = agent_chain.run(input=query2)
# chat_history.append((query2, response))
# response = agent_chain.run(input=query3)
# chat_history.append((query3, response))
# response = summarize_chat(chat_history, agent_chain)
# print(response)
# while True:
#     text_input = input("User: ")
#     response = agent_chain.run(input=text_input)
#     print(f'Agent: {response}')

# Front end web app
import gradio as gr
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    chat_history = []
    
    def user(user_message, history):
        # Get response from QA chain
        response = agent_chain.run(input=user_message)
        # Append user message and response to chat history
        history.append((user_message, response)) 
        return gr.update(value=""), history
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(debug=True)

