import streamlit as st
from ask import main

st.title("💬 Text Input & Output Demo")

# Input from user
user_input = st.text_area("Enter your question or text:", height=150)

# Action button
if st.button("Submit"):
    # Simulated processing (replace this with your logic)
    content, response = main(user_input)
    # Output display
    st.markdown("### 📤 Output")
    st.write(response)

    st.markdown("### 📤 Content")
    st.write(content)