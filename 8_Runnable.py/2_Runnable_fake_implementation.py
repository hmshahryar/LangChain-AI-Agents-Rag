from abc import ABC, abstractmethod
import random
class Runnable(ABC):
    @abstractmethod
    def invoke(input_data):
        pass

    import random

class Runnable_connector(Runnable):
    def __init__(self , runable_list:list):
        self.runnable = runable_list
        
    def invoke(self , input_data):
        for runble in self.runnable:
            input_data = runble.invoke(input_data)
        return input_data



class FakeLLM(Runnable):

    def __init__(self):
        print('FakeLLM initialized.')

    def invoke(self , prompt ):
        response_list = [
            'Islamabad is the capital of Pakistan.',
            'Karachi is the capital of Sindh.'
        ]
        return {'response': random.choice(response_list)}

        

    def predict(self, prompt: str):
        response_list = [
            'Islamabad is the capital of Pakistan.',
            'Karachi is the capital of Sindh.'
        ]
        return {'response': random.choice(response_list)}


class PromptTemplate(Runnable):
    def __init__(self, template: str, input_variables: list):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict: dict):
        return self.template.format(**input_dict)

    

    def format(self, input_dict: dict):
        return self.template.format(**input_dict)


class LLMChain:
    def __init__(self, llm: FakeLLM, prompt: PromptTemplate):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict: dict):
        final_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(final_prompt)
        return result['response']


poem_prompt = PromptTemplate(
    template='Write a {style} poem about {topic}.',
    input_variables=['topic', 'style']
)

fake_llm = FakeLLM()
llm_chain = LLMChain(llm=fake_llm, prompt=poem_prompt)


output = llm_chain.run({'topic': 'Pakistan', 'style': 'short'})
print("Generated response:", output)

# _________________________________________________________________
from abc import ABC, abstractmethod
import random

# --------------------------
# 1. Abstract Runnable Interface
# --------------------------
class Runnable(ABC):
    @abstractmethod
    def invoke(self, input_data):
        pass

# --------------------------
# 2. PromptTemplate – Formats input into a prompt string
# --------------------------
class PromptTemplate(Runnable):
    def __init__(self, template: str, input_variables: list):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict: dict) -> str:
        return self.template.format(**input_dict)

# --------------------------
# 3. FakeLLM – Mocks a language model that takes prompt string
# --------------------------
class FakeLLM(Runnable):
    def __init__(self):
        print('FakeLLM initialized.')

    def invoke(self, prompt: str) -> dict:
        response_list = [
            'Islamabad is the capital of Pakistan.',
            'Karachi is the capital of Sindh.'
        ]
        return {'response': random.choice(response_list)}

# --------------------------
# 4. LLMChain – Connects PromptTemplate and FakeLLM using invoke
# --------------------------
class LLMChain(Runnable):
    def __init__(self, prompt_template: Runnable, llm: Runnable):
        self.prompt_template = prompt_template
        self.llm = llm

    def invoke(self, input_data: dict) -> str:
        # Step 1: Format the prompt using the template
        formatted_prompt = self.prompt_template.invoke(input_data)

        # Step 2: Pass the prompt to the LLM
        result = self.llm.invoke(formatted_prompt)

        # Step 3: Return just the response text
        return result['response']

# --------------------------
# 5. RunnableConnector – Optional: Chain multiple components
# --------------------------
class RunnableConnector(Runnable):
    def __init__(self, runnables: list):
        self.runnables = runnables

    def invoke(self, input_data):
        for runnable in self.runnables:
            input_data = runnable.invoke(input_data)
        return input_data

# --------------------------
# 6. Example Usage
# --------------------------
if __name__ == "__main__":
    # Create the prompt template
    poem_prompt = PromptTemplate(
        template="Write a {style} poem about {topic}.",
        input_variables=["topic", "style"]
    )

    # Create the fake LLM
    fake_llm = FakeLLM()

    # Combine them in a chain
    poem_chain = LLMChain(prompt_template=poem_prompt, llm=fake_llm)

    # Input data
    input_data = {"topic": "Pakistan", "style": "short"}

    # Invoke the chain
    result = poem_chain.invoke(input_data)
    print("Generated response:", result)
