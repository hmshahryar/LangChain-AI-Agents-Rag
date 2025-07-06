# Text splitters usually are to break elngth in to manageble number of pages or length 
# context length is diferen in any model how much can we enter input 


# text then vectors  then  using embedding 

# text then chunk then embedding so captre more sementic meaning

# embed quality is more if little chunck give more reasonable search


# sementic search after spliting is more accurate 

# so use smal chunk for better resource managemnt 

# length basae  text spliting define split size 100 character 
# text structure 
# doc structure 
# smentinc meanig split

from langchain_text_splitters import  CharacterTextSplitter

text = """Artificial Intelligence (AI) has rapidly evolved from a futuristic concept to a transformative force deeply integrated into modern life. Initially conceived in the mid-20th century as a theoretical field, AI was aimed at enabling machines to mimic human intelligence. Early AI systems were rule-based and limited in capability, often confined to narrow tasks such as playing chess or solving equations. However, with advances in computing power, access to large data sets, and sophisticated algorithms, AI has experienced exponential growth and widespread adoption."""


splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0 ,
    separator = ''
)

resutl = splitter.split_documents(text)
print(resutl)


# --------------------------------- now for docs loader ------------------

from langchain_community.document_loaders import PyPDFLoader

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0 , # whct overlap betwenn two chunk context treading is good in context overlap
    separator = ''
)


loader = PyPDFLoader(file_path="Pictures_Notes\\the-art-of-seduction-robert-greene.pdf")

res = loader.lazy_load()

resutl = splitter.split_documents(res)
print(resutl[0])