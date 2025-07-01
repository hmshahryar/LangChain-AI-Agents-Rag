from langchain_community.document_loaders import WebBaseLoader , SeleniumURLLoader


path = "https://www.tripadvisor.com/Airline_Review-d8729039-Reviews-British-Airways"
loader = WebBaseLoader(path)

docs = loader.load()

