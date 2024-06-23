"""
This script is used to create a vectorstore from a directory of pdf files. The vectorstore is later used in main.py to \
search for similar documents.
"""


import os
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import docx 

os.environ["OPENAI_API_KEY"] = st.secrets["api_key"]


def get_text_from_dir(dir_path):
    files = os.listdir(dir_path)
    # files = ['Part 1_ Software and Mobile App Introduction.pdf']

    files_txt = []
    for file in files:
        pdf_reader = PdfReader(os.path.join(dir_path, file))

        raw_text = ''
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                raw_text += '\n' + text

        files_txt.append(raw_text)

    for i, raw_text in enumerate(files_txt):
        with open(f'docs_txt/{i}.txt', 'w', encoding='utf-8') as f:
            f.write(raw_text)

    raw_text = '\n'.join(files_txt)
    return raw_text


def do_formating_with_gpt(raw_text):
    print('1')
    client = OpenAI()
    final_text=f"""\
You are given an raw text from a pdf file. You are asked to format the text into a readable format. \
The raw text may include tables and nested lists. The text is as follows:
```{raw_text}```"""

    response = client.chat.completions.create(
        messages=[
            {'role': 'system', 'content': final_text},
        ],
        model="gpt-3.5-turbo",
        temperature=0,
    )
    txt = response.choices[0].message.content
    return txt


def text_splitter(raw_text):
    # split into 3250 characters chunks
    # docs = NLTKTextSplitter(chunk_size=1500, chunk_overlap=400).split_text(raw_text)
    docs = raw_text.split("-------------------------------------------------------------------")
    return docs


def create_embedding(docs):
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(docs, embeddings)
    return docsearch


def get_doc_string(doc):
    doc = docx.Document(doc)


    # Iterate through paragraphs and append text to the entire content
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)

    # Print the entire content
    return '\n'.join(fullText)


if __name__ == '__main__':

    with open("docs_txt/0.txt","r", encoding="utf-8") as f:
        pdf_content = f.read()

    docs = text_splitter(pdf_content)
    docs = [do_formating_with_gpt(doc) for doc in docs]
    docsearch = create_embedding(docs)
    docsearch.save_local('vectorstore')
