import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

## Function To get response from LLAma 2 model

def getLLamaresponse(subject):

    ### LLama2 model
    llm=CTransformers(model='./model/llama-2-7b-chat.ggmlv3.q5_1.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})
    
    ## Prompt Template

    template="""
    You are an expert MCQ maker.
    Create a quiz of {number} multiple choice questions for {subject} students in {tone} tone. 
    Make sure the questions are not repeated.
    Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
    Ensure to make {number} MCQs
    ### RESPONSE_JSON
    {response_json}
    """

    response_json = {
        "1": {
            "mcq": "multiple choice question",
            "options": {
                "a": "choice here",
                "b": "choice here",
                "c": "choice here",
                "d": "choice here",
            },
            "correct": "correct answer",
        },
        "2": {
            "mcq": "multiple choice question",
            "options": {
                "a": "choice here",
                "b": "choice here",
                "c": "choice here",
                "d": "choice here",
            },
            "correct": "correct answer",
        },
        "3": {
            "mcq": "multiple choice question",
            "options": {
                "a": "choice here",
                "b": "choice here",
                "c": "choice here",
                "d": "choice here",
            },
            "correct": "correct answer",
        },
    }

    tone = "simple"

    number = "ten"
        
    prompt=PromptTemplate(input_variables=["text", "number", "subject", "tone", "response_json"],
                        template=template)
    
    ## Generate the ressponse from the LLama 2 model
    prompt_value = prompt.invoke({"number": number, "subject": subject, "tone": tone, "response_json": response_json})
    response = llm.invoke(prompt_value)
    return str(response)

st.set_page_config(page_title="Generate Multiple Choice Questions",
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate Multiple Choice Questions")

subject = st.text_input("Enter the subject")

submit=st.button("Generate")

## Final response
if submit:
    st.write(getLLamaresponse(subject))