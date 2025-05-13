from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from Extractor import partition

import os

def tt_summary(tables,texts):
    prompt_text=""" you are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additional comments.
    Do not start your message by saying "Here is a summary" or anything like that
    Just give the summary as it is.

    Table or text chunk : {element}
    """

    prompt = ChatPromptTemplate.from_template(prompt_text)

    model = ChatGroq(temperature=0.2,model='llama-3.1-8b-instant')
    summarize_chain = prompt | model | StrOutputParser()
    text_summaries = summarize_chain.batch(texts,{"max_concurrency":3})
    tables_html = [table.metadata.text_as_html for table in tables]
    table_summaries = summarize_chain.batch(tables_html,{"max_concurrency":3})
    
    return table_summaries,text_summaries

def img_summary(images):
    prompt_text=""" you are an assistant tasked with summarizing Images.
    Give a concise summary of the Image.
    If the Image contains labelled diagrams make sure to be specific about position and other structural details.
    If the Image contains graph or plots be specific about them.
    Respond only with the summary, no additional comments.
    Do not start your message by saying "Here is a summary" or anything like that
    Just give the summary as it is.
    """

    messages=[
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt_text
            },
            {
                "type": "image_url",
                "image_url": {"url":"data:image/jpeg;base64,{image}"}  
            }
        ]
    }
    ]   

    prompt = ChatPromptTemplate.from_messages(messages)

    model = ChatOpenAI(temperature=0.2,model="opengvlab/internvl3-14b:free",base_url="https://openrouter.ai/api/v1",api_key=os.getenv('OPENAI_API_KEY'))
    
    summarize_chain = prompt | model | StrOutputParser()
    img_summaries = summarize_chain.batch(images)
    return img_summaries
    

def main():
    tables,texts,images = partition(DIR_PATH=r'D:\Projects\JEEgpt\Files')
    # table_summaries,text_summaries = tt_summary(tables=tables,texts=texts)
    image_summaries = img_summary(images)
    print(image_summaries[0])


if __name__=='__main__':
    main()