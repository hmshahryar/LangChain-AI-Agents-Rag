# runable process sigle process input process outpu
# All runable follows some set of method 
# comon interface  so can conet and can perform comple work fow 

# r1 r2 and r3 are part of one runable r4 and r5 of one both 
# so both can combine as one runable like lego box 
import random


class FakeLLM:
    def __init__(self):
        print('FakeLLM initialized.')

    def predict(self, prompt: str):
        response_list = [
            'Islamabad is the capital of Pakistan.',
            'Karachi is the capital of Sindh.'
        ]
        return {'response': random.choice(response_list)}


class PromptTemplate:
    def __init__(self, template: str, input_variables: list):
        self.template = template
        self.input_variables = input_variables

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
