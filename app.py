import streamlit as st
from ask import main
from utils import embed_with_markdown_splitter
from config import CONFIG

st.title("💬 Text Input & Output Demo")

left, middle, right = st.columns([10,15,0.5])

# button to retrain
if left.button("Embed"):
    embed_with_markdown_splitter(CONFIG["document_folder"])

# Input from user
user_input = middle.text_area("Enter your question or text:", height=150)

# Action button
if middle.button("Submit"):
    # Simulated processing (replace this with your logic)
    content, response = main(user_input)
    # Output display
    middle.markdown("### 📤 Output")
    middle.write(response)

    middle.markdown("### 📤 Content")
    middle.write(content)