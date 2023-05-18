import re
import os
import os
os.environ["OPENAI_API_KEY"] = "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"

# import streamlit as st 
import urllib
import base64
import os
from llama_index import GPTListIndex #GPTSimpleVectorIndex, SimpleDirectoryReader, 
from dotenv import load_dotenv
from llama_index import Document

#summarization 
def summarize_chat(chat_history, agent_chain):
    response = agent_chain.run(input="Summarize our conversation until now")
    return response


def summarize(chat_history):
    documents = Document(text=chat_history)
    index = GPTListIndex.from_documents([documents])
    response = index.query("Summarize the document", response_mode="tree_summarize")
    return response


def preprocess_data(documents):
    from nltk.corpus import stopwords
    import nltk
    
    stop_words = set(stopwords.words('english'))
    not_stopwords = ['no', 'not', 'nor', 'only', 'too', 'very', 'against', \
                    'very', 'only', 'until', 'while', 'further', 'most', 'other', 'some',\
                        'before', 'after']
    stop_words = [word for word in stop_words if word not in not_stopwords]
    for i in range(len(documents)):
        data_temp = documents[i].page_content
        data_temp = data_temp.replace("\n", " ")    
        data_temp = re.sub(r'[^\x00-\x7F]+',' ', data_temp)
        # re.sub(r'\W+', '', data_temp)
        words = data_temp.split(" ")
        words = [word for word in words if word not in stop_words]
        documents[i].page_content = ' '.join(words)
 
    return documents


def get_section_subsec_names(tag: 'img', soup):
    import re
    tags = soup.find_all([re.compile('^h[1-6]$'), tag])
    tag_names = [tag.name for tag in tags]
    tag_values = []
    for i in range(len(tags)):
        if tag_names[i][0] == 'h' and len(tag_names[i]) == 2:
            tag_values.append(tags[i].text.strip())
        else:
            tag_values.append('')
    level = '1'
    sec = ['0']*6
    sec_name = ['']*6
    sec_str = ''
    # image_sec = []
    image_section_no = []
    image_section_name = []
    image_section_no = '1.1.1'
    image_section_name = 'no_name'
    return image_section_no, image_section_name


def get_text_between_html_tags(tag: str, urls):
    from bs4 import BeautifulSoup
    import urllib
    # get attributes such as location (doc_id, doc_name, section_no, section_name) , to identify where to insert metadata
    tag_values = {}
    for j, url in enumerate(urls):
        page = urllib.request.urlopen(url)
        soup = BeautifulSoup(page)
        links = soup.find_all('a')
        tags = [link for link in links if tag in link.get('href', [])] 
        tag_vals = []
        for i in range(len(tags)):
            section_no, section_name = get_section_subsec_names(tag, soup)
            tag_vals.append(tags[i].text.strip())
        tag_values[j] = tag_vals
    return tag_values


def get_img_metadata(urls):
    import json
    from bs4 import BeautifulSoup
    import urllib
    from langchain.document_loaders import ImageCaptionLoader
    # ideally want to use ImageVisionLLMReader = download_loader("ImageVisionLLMReader") # multimodal visionLLM for captioning
    count = 0
    image_metadata = []
    image_attributes = {
        'doc_no': '',
        'section_no': '',
        'section_name': '',
        'before_text': '',
        'after_text': '',
        'alt_text': '',
        'caption': '',
        'hyperlink': '',
    }
    for i, url in enumerate(urls):
        page = urllib.request.urlopen(url)
        soup = BeautifulSoup(page)
        images = soup.findAll('img')
        image_attributes['doc_no'] = str(i)
        for image in images:
            # add section, sub_section identification for image
            img_section_no, img_section_name = get_section_subsec_names('img', soup)
            image_attributes['section_no'] = img_section_no[count]
            image_attributes['section_name'] = img_section_name[count]
            image_attributes['hyperlink'] = '/'.join(url.split('/')[:-2]) + '/' + image['src'][3:]
            image_attributes['alt_text'] = image['alt']

            # # load image captioning model
            
            # list_image_urls = [image_attributes['hyperlink']]
            # img_loader = ImageCaptionLoader(path_images=list_image_urls)
            # list_docs = img_loader.load()
            # image_attributes['caption'] = list_docs[0].page_content
            image_metadata.append(json.dumps(image_attributes))
            count += 1
            
    return image_metadata


def get_url_metadata(urls):
    import json
    from bs4 import BeautifulSoup
    import urllib
    count = 0
    url_metadata = []
    url_attributes = {
        'doc_no': '',
        'section_no': '',
        'section_name': '',
        'before_text': '',
        'after_text': '',
        'text': '',
        'summary': '',
        'hyperlink': '',
    }
    for j, url in enumerate(urls):
        page = urllib.request.urlopen(url)
        soup = BeautifulSoup(page)
        links = soup.find_all('a')
        urls = [link.get('href', []) for link in links if len(link.attrs) == 1]
        url_attributes['doc_no'] = str(j)
        pdfs = [link.get('href', []) for link in links] 
        for i, url in enumerate(urls):
            # add section, sub_section identification for url
            section_no, section_name = get_section_subsec_names('http', soup)
            url_attributes['section_no'] = section_no
            url_attributes['section_name'] = section_name
            url_attributes['hyperlink'] = url 
            url_attributes['text'] = ''
            url_metadata.append(json.dumps(url_attributes))
            count += 1
    return url_metadata


def get_pdf_metadata(urls):
    import json
    from bs4 import BeautifulSoup
    import urllib
    count = 0
    pdf_metadata = []
    pdf_attributes = {
        'doc_no': '',
        'section_no': '',
        'section_name': '',
        'before_text': '',
        'after_text': '',
        'text': '',
        'summary': '',
        'hyperlink': '',
    }
    tag = 'pdf'
    list_texts = get_text_between_html_tags(tag, urls)
    for j, url in enumerate(urls):
        page = urllib.request.urlopen(url)
        soup = BeautifulSoup(page)
        links = soup.find_all('a')
        pdfs = [link for link in links if tag in link.get('href', [])] 
        pdf_attributes['doc_no'] = str(j)
        
        for i, pdf in enumerate(pdfs):
            # add section, sub_section identification for image
            section_no, section_name = get_section_subsec_names(tag, soup)
            pdf_attributes['section_no'] = section_no
            pdf_attributes['section_name'] = section_name
            pdf_attributes['hyperlink'] = url # '/'.join(url.split('/')[:-2]) + '/' + image['src'][3:]
            pdf_attributes['text'] = list_texts[j][i]
            # pdf_attributes['summary'] = some PDF summary generator
            pdf_metadata.append(json.dumps(pdf_attributes))
            count += 1
    return pdf_metadata


def get_pdf_summary(urls):
    from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
    from langchain.chains.summarize import load_summarize_chain
    from langchain.llms import OpenAI
    from bs4 import BeautifulSoup
    import urllib
    import requests
    all_pdf = []
    for url in urls:
        page = urllib.request.urlopen(url)
        soup = BeautifulSoup(page)
        # Find all hyperlinks present on webpage
        links = soup.find_all('a')
        links = [link.get('href') for link in links if '.pdf' in link.get('href', [])] 
        summary = ''
        for link in links:
            if not 'arxiv' in link:
                continue
            response = requests.get(link)
            # loader = UnstructuredPDFLoader(link)
            # pages = loader.load()   
            docs = Document(page_content = response.content)
            llm = OpenAI(temperature=0)
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            summary += chain.run(docs) 
            break
        all_pdf.append(summary)
    return all_pdf


def get_table_metadata(urls):
    import pandas as pd
    all_tables = []
    for url in urls:
        tables = pd.read_html(url)
        all_tables.extend(tables)
    return all_tables

