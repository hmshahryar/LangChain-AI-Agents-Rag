from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen3-0.6B",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature = 0.5,
        max_new_tokens = 100
    )
)
model = ChatHuggingFace(llm = llm)

result = model.invoke("capital of pakistan")
print(result.content)

# import os   os.environ['Hf_HOME'] = "D:/huggingface_cache"