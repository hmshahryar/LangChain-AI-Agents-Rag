from langchain_core.prompts import ChatPromptTemplate



template = ChatPromptTemplate([
    ('system' , 'you are a professional {profession} guidance expert '),('human','explain about the {topic}')
])

prompt = template.invoke({'profession': 'expert datascientist','topic' : "explain best way to fill na in data_set "})
print(prompt)

# we also sees that the chatpromptTemplate.message  some time is uses but still both refers the same thing 
