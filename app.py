from dotenv import load_dotenv
from Extractor import partition
# from Summarize import tt_summary,img_summary
from VectorStore import loader
from Model import model
import pickle

load_dotenv()

DIR_PATH=r'D:\Projects\JEEgpt\Files'
def main():
    # tables,texts,images = partition(DIR_PATH)
    with open(r"D:\Projects\JEEgpt\App_Files\table_summaries.pkl",'rb') as f:
        table_summaries = pickle.load(f)
    with open(r"D:\Projects\JEEgpt\App_Files\text_summaries.pkl",'rb') as f:
        text_summaries = pickle.load(f)
    with open(r"D:\Projects\JEEgpt\App_Files\tables.pkl",'rb') as f:
        tables = pickle.load(f)
    with open(r"D:\Projects\JEEgpt\App_Files\texts.pkl",'rb') as f:
        texts = pickle.load(f)
    # image_summaries = img_summary(images)
    retriever = loader(texts,text_summaries,tables,table_summaries)
    while True:
        query = str(input("How can i Help you today?...."))
        response = model(retriever,query)
        print(response)
        
    

if __name__=='__main__':
    main()