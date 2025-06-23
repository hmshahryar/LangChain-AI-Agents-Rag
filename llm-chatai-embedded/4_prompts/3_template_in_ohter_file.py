from langchain_core.prompts import PromptTemplate


templatee = PromptTemplate(
    template="""
            You are a creative poet. Write a {style} poem about {topic} in {language}.
            The tone should be {tone}, and it should include references to {reference}.
        """,
    input_variables=["style", "topic", "language", "reference", "tone"],validate_template=True
)

templatee.save('template.json')