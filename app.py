from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
import requests
import os
import streamlit as st

load_dotenv(find_dotenv())
HUGGINFACE_API_TOKEN = os.getenv("HUGGINFACE_API_TOKEN")

# img2text
def img2text(url):
    img_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    
    text = img_to_text(url)[0]["generated_text"]
    
    print(text)
    return text

# llm
def generate_story(scenario):
    template = """"
    you are a story teller;
    you can generate a short story based on a simple narrative, the story should be no more than 20 words;

    CONTEXT = {scenario}
    STORY:
    """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=1), prompt=prompt, verbose=True)

    story = story_llm.predict(scenario=scenario)
    return story


# text2speech
def text2speech(message):
    headers = {"Authorization": f"Bearer {HUGGINFACE_API_TOKEN}"}
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    payloads = {
        "inputs": message
    }
    response = requests.post(API_URL, headers=headers, json=payloads)
    # print(response.content)
    # return response
    with open('audio.mp3', 'wb') as file:
        file.write(response.content)


scenario = img2text('R1.jpg')
story = generate_story(scenario)
text2speech(story)

def main():
   st.set_page_config(page_title="image to Audio Story", page_icon="🔉") 
   
   st.header("Turn img into audio story")
   uploaded_file = st.file_uploader("Choose an image...", type="jpg")
   
   if uploaded_file:
       print(uploaded_file)
       bytes_data = uploaded_file.getvalue()
       with open(uploaded_file.name, "wb") as f:
            f.write(bytes_data)
       st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
       
       scenario = img2text(uploaded_file.name)
       story = generate_story(scenario)
       text2speech(story)
       
       with st.expander("scenario"):
           st.write(scenario)
       with st.expander("story"):
           st.write(story)
       
       st.audio("audio.mp3")


if __name__ == '__main__':
    main()