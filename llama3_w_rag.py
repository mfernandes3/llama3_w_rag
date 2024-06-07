# %%
# Import necessary libraries
import os
import tempfile
import streamlit as st
from embedchain import App


# %%
# Define a function to create an instance of the embedchain App
def embedchain_bot(db_path):
    return App.from_config(
        config={
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "llama3:instruct",
                    "max_tokens": 250,
                    "temperature": 0.01,
                    "stream": True,
                    "base_url": "http://localhost:11434",
                    "prompt": "Use the following pieces of context to answer the query at the end.\nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n$context: \n\nQuery: $query\find in the information provided the right answer\n Answer:",
                    "system_prompt": "Act as a factual chatbot",
                    "number_documents": 50,
                },
            },
            "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": "llama3:instruct",
                    "base_url": "http://localhost:11434",
                },
            },
        }
    )


st.title("Local Llama3 Chatbot with Ollama and Embedchain")
st.caption(
    "This app allows you to chat with a PDF using Llama3 running locally with Ollama!"
)

# Create a temporary directory to store the PDF file
db_path = "/Users/miguel.fernandes/Documents/rag_pdf_files/"
print(f"Created temporary directory: {db_path}")
# Create an instance of the embedchain App
app = embedchain_bot(db_path)
# %%

# Reset the app function
def reset_app():
    # Reset the app
    app.reset()
    st.success("App reset successfully!")
# Reset button
if st.button("Reset Data and Context"):
    reset_app()


# %%
# Upload a PDF file
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

# If a PDF file is uploaded, add it to the knowledge base
if pdf_file:
    pdf_context= st.text_area("Add context as text to the PDF file")
    if st.button("Add PDF"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(pdf_file.getvalue())
            
            app.add(f.name, data_type="pdf_file", metadata={"context": pdf_context})
        os.remove(f.name)
        st.success(f"Added {pdf_file.name} to knowledge base!")

# Add website URL
website_url = st.text_input("Add a website URL")
if website_url:
    website_context = st.text_area("Add context as text to the website")
    if st.button("Add Website"):
        
        app.add(website_url, data_type='web_page', context=website_context)
        st.success("Website added successfully!")

# Add context as text
context_text = st.text_area("Add text as context")
if context_text:
    context_text = st.text_area("Add additional ontext as text")
    if st.button("Add Context"):
        
        app.add(context_text, data_type="text", context= context_text)
        st.success("Context added successfully!")
 

# Ask a question about the content provided
prompt = st.text_input("Ask a question about the content provided")
# Display the answer
if prompt:
    answer = app.search(prompt)
    st.write(answer)
    st.write("data sources ", app.get_data_sources())
