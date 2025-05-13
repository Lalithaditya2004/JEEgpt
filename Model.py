from langchain_core.runnables import RunnableLambda,RunnablePassthrough
from base64 import b64decode
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

def parse_docs(docs):
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    return {'images':b64,'texts':text}

def build_prompt(kwargs):
    docs_by_type = kwargs['context']
    user_question = kwargs['question']

    content_text=""

    if len(docs_by_type['texts']) > 0:
        for text_ele in docs_by_type["texts"]:
            content_text += text_ele.text

    prompt_template=f"""
    Answer the question based on the following context, which can include text, tables, and the below image.
    Answer the question in one line only unless it is mentioned to explain.
    Context : {content_text}
    Question : {user_question}
    """

    prompt_content = [{"type":"text","text":prompt_template}]

    if len(docs_by_type['images']) > 0:
        for image in docs_by_type['images']:
            prompt_content.append(
                {
                    "type":"image_url",
                    "image_url":{"url":f"data:image/jpeg;base64,{image}"},
                }
            ) 

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content)
        ]
    )


def model(retriever,query):
    chain = (
        {
            "context":retriever | RunnableLambda(parse_docs),
            "question":RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | ChatOpenAI(temperature=0.5,model="meta-llama/llama-4-maverick:free",base_url="https://openrouter.ai/api/v1",api_key=os.getenv('LLAMA_KEY'))
        | StrOutputParser()
    )

    return chain.invoke(query)
