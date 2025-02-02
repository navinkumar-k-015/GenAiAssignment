from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from utils.creds import GITHUB_TOKEN

endpoint = 'https://models.inference.ai.azure.com' 
llm = ChatOpenAI( openai_api_base=endpoint, openai_api_key=GITHUB_TOKEN, model="gpt-4o-mini")


template = """
You are a Ubuntu Expert. This Human will ask you questions about Ubuntu.
Use the following piece of context to answer the question.
History:
{history}

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["history", "context", "question"]
)

output_parser = StrOutputParser()
chain = LLMChain(prompt=prompt, llm=llm, output_parser=output_parser)

def get_llm_response(history_text, context, question):
    
    answer = chain.invoke({"history": history_text, "context": context, "question": question})
    return answer