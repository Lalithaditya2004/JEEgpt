from langchain_chroma import Chroma
# from langchain.storage import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain.storage import InMemoryStore
import pickle
import uuid
from Extractor import partition
from Summarize import tt_summary,img_summary

CHROMA_DIR = 'chroma_store'
DOC_DIR = 'docstore.db' 
RETRIEVER_PATH = 'retriever.pkl'
DIR_PATH=r'D:\Projects\JEEgpt\Files'

def init():
    embedder = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base",
        encode_kwargs={"normalize_embeddings": True}  
    )

    vectorstore = Chroma(collection_name='mmr',embedding_function=embedder)

    store = InMemoryStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key
    )
    return retriever


def loader(texts,text_summaries,tables,table_summaries):
    retriever = init()
    id_key = "doc_id"
    if len(texts)!=0:
        doc_ids =  [str(uuid.uuid4()) for t in texts]
        summary_texts = [
            Document(page_content=summary, metadata={id_key:doc_ids[i]}) for i,summary in enumerate(text_summaries)
        ]
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids,texts)))
    
    if len(tables)!=0:
        table_ids =  [str(uuid.uuid4()) for t in tables]
        summary_tables = [
            Document(page_content=summary, metadata={id_key:table_ids[i]}) for i,summary in enumerate(table_summaries)
        ]
        retriever.vectorstore.add_documents(summary_tables)
        retriever.docstore.mset(list(zip(table_ids,tables)))

    # if len(imgs)!=0:
    #     img_ids =  [str(uuid.uuid4()) for t in imgs]
    #     summary_imgs = [
    #         Document(page_content=summary, metadata={id_key:img_ids[i]}) for i,summary in enumerate(img_summaries)
    #     ]
    #     retriever.vectorstore.add_documents(summary_imgs)
    #     retriever.docstore.mset(list(zip(img_ids,imgs)))


    return retriever

def invoker(msg,retriever):
    chunks = retriever.invoke(msg)
    return chunks

def main():
    tables,texts,images = partition(DIR_PATH)
    table_summaries,text_summaries = tt_summary(tables=tables,texts=texts)
    image_summaries = img_summary(images)
    retriever = loader(texts,text_summaries,tables,table_summaries,images,image_summaries)

    with open(RETRIEVER_PATH, "wb") as f:
        pickle.dump(retriever, f)


if __name__ == '__main__':
    main()