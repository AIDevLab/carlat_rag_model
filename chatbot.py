"""
This script contains the ChatBot class, which is used to define the chatbot's behavior and interaction with the user.
"""
from copy import deepcopy
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.faiss import DistanceStrategy
import os
from trulens_eval.feedback.provider import OpenAI
from trulens_eval.tru_custom_app import instrument


os.environ["OPENAI_API_KEY"] = st.secrets["api_key"]

class ChatBot:
    def __init__(self, verbose=False):
        #self.trulens = TruChain()  # Initialize TruLens
        self.prompt_template = """\
Your task is being an friendly assistant that specializes in answering the user's questions only from the provided context. \
You are asked to answer the Query based on a Context. \
If the query is greeting, you should answer with a greeting. \
If the query is not related to the ADHD medication you should answer with a message saying that \
"I'm here to provide valuable information related to ADHD medication. Your question falls out of \
my expertise. Please ask a related question." \
Don't make up answers by your own. If you don't know the answer just say, \
"I don't have enough information about this. Can I help you with something else?" \
Also keep the answers to the point according to Query.\
Always format your answer in enumerated bullet points whenever there are steps to follow in your answer.
Give all details that appears in the context related to the user's query.\
Make sure to use a proper list/sublist bullet points where necessary.\
Use below  Context to answer the Query.\

Context: ```{context}```
Query: ```{query}```"""

        self.llm = ChatOpenAI(model='gpt-3.5-turbo-0125', temperature=0.1)
        self.prompt = PromptTemplate(input_variables=['query', 'context'], template=self.prompt_template)

        self.qa_chain = load_qa_chain(self.llm,
                                      chain_type='stuff',
                                      prompt=self.prompt,)
                                      #verbose=verbose)

        self.docs = self.load_vectorstore()

    def load_vectorstore(self):
        embeddings = OpenAIEmbeddings()
        # always remove allow_dangerous_deserialization in prod 
        docsearch = FAISS.load_local('vectorstore', embeddings, allow_dangerous_deserialization= True)
        # Initialize the FAISS object with a different distance strategy
        docsearch = FAISS(
            embedding_function=embeddings,
            index= docsearch.index,
            docstore= docsearch.docstore,
            index_to_docstore_id = docsearch.index_to_docstore_id,
            distance_strategy=DistanceStrategy.COSINE)
        return docsearch

    @instrument
    def retrieve(self, query, docsearch, k=5):
        results = docsearch.similarity_search(query, k=k)
        return results

    @instrument
    def query(self, query, chat_history):

        chat_history = deepcopy(chat_history)
        mem_str = self.memory_to_string(chat_history)


        # Perform TruLens evaluation

        sim_docs = self.retrieve(query, self.docs)

        res = self.qa_chain({'input_documents': sim_docs, 'query': query,'chat_history': mem_str}, return_only_outputs=True)
        output = res['output_text'].strip()
        self.add_query_and_response_to_memory(query, output, chat_history)

        return output, chat_history

    def add_query_and_response_to_memory(self, query, response, memory):
        memory.append({'role': 'user', 'content': query})    
        memory.append({'role': 'assistant', 'content': response})

    def memory_to_string(self, memory):
        mem_str = ''
        for msg in memory:
            if msg['role'] == 'user':
                mem_str += f'Human: {msg["content"]}\n'
            else:
                mem_str += f'AI: {msg["content"]}\n'

        return mem_str


if __name__ == '__main__':
    chatbot = ChatBot(verbose=True)
    chat_history = []

    while True:
        query = input('Query: ')
        if query == 'exit':
            break

        output, chat_history = chatbot.query(query, chat_history)
        # chat_history.append({'role': 'user', 'content': query})
        # chat_history.append({'role': 'AI', 'content': output})
        #print('Answer: ', output, end='\n\n')

