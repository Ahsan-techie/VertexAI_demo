import vertexai
import streamlit as st
import os
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
from dotenv import load_dotenv


load_dotenv()

project_id = os.getenv("project_id")
project_region = os.getenv("region")

vertexai.init(project=project_id, location=project_region)

model = GenerativeModel("gemini-1.5-pro")


def user_interfaces():
    st.set_page_config("VertexAI Demo")
    st.header("Vertex AI Local demo")

    user_question = st.text_input("Ask a question...")

    if user_question:
        response = model.generate_content(
            user_question,
            generation_config=GenerationConfig(temperature=0.2, max_output_tokens=5000),
            stream=True,
        )

        for res in response:
            st.write(res.text, end="")


if __name__ == "__main__":
    user_interfaces()
