from unstructured.partition.pdf import partition_pdf
import os
from tqdm import tqdm

def get_img(chunks):
    imgs=[]
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_ele = chunk.metadata.orig_elements
            for el in chunk_ele:
                if "Image" in str(type(el)):
                    imgs.append(el.metadata.image_base64)
    return imgs

def partition(DIR_PATH):
    tables = []
    texts = []
    images = []
    for file in os.listdir(DIR_PATH):
        chunks = partition_pdf(
            filename=os.path.join(DIR_PATH,file),
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            chunking_strategy='by_title'
        )
        for chunk in chunks:
            if "Table" in str(type(chunk)):
                tables.append(chunk)
            if "CompositeElement" in str(type(chunk)):
                texts.append(chunk)
        
        imgs = get_img(chunks)
        for img in imgs:
            images.append(img)
    
    return tables,texts,images

def main():
    tables,texts,images = partition(DIR_PATH=r'D:\Projects\JEEgpt\Files')
    print('hi')
    print(type(images))
    # print(tables)
    # print(texts)
    


if __name__=='__main__':
    main()