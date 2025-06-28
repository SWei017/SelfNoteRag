import streamlit as st
from main import SelfNoteRAGPipeline
from config import CONFIG

rag_pipeline = SelfNoteRAGPipeline(CONFIG)

st.title("💬 Text Input & Output Demo")

left, middle, right = st.columns([10,15,0.5])

# button to retrain
if left.button("Embed"):
    rag_pipeline.embed(CONFIG["document_folder"])

# Input from user
user_input = middle.text_area("Enter your question or text:", height=150)

# Action button
if middle.button("Submit"):
    # Simulated processing (replace this with your logic)
    content, response = rag_pipeline.ask(user_input)
    # Output display
    middle.markdown("### 📤 Output")
    middle.write(response)

    middle.markdown("### 📤 Content")
    middle.write(content)