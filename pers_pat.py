# Import necessary libraries
import os
import tempfile
import streamlit as st
from embedchain import App

# Define a function to create an instance of the embedchain App
def embedchain_bot(db_path, model_name):
    return App.from_config(
        config={
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "llama3:instruct",
                    "max_tokens": 250,
                    "temperature": 0.1,
                    "stream": True,
                    "base_url": "http://localhost:11434",
                    "prompt": r'''Write in German a detailed doctor's recommendation coming from BetterDoc for a patient in formulated sentences about the following doctor and his key data. 
                    Do not write as a first person, but as a third person coming from the company BetterDoc.
                    Do not write information you don't know like the patient's name, case ID, clinics names, and medics names or make up information like the specialty of the doctor.
                    Follow the Example:
                    Sehr geehrte/r Frau/Herr {patient_name},
                    Hiermit empfiehlt BetterDoc Dr. med. XXX als hervorragenden Facharzt im Bereich der ...
                    Im Folgenden stellen wir Ihnen einen Spezialisten für {Fachgebiet} vor. Alle empfohlenen Ärzte sind für eine Beratung und Behandlung in Ihrem Fall qualifiziert.

                    Bitte klären Sie vor einem stationären Aufenthalt oder einer Operation immer die Kostenübernahme mit Ihrer Versicherung ab, um zusätzliche Kosten zu vermeiden. Sollte nach Ihrer ambulanten Konsultation ein stationärer Aufenthalt notwendig sein, wenden Sie sich bitte erneut an BetterDoc, damit wir Sie bei der Suche nach dem für Sie am besten geeigneten Spezialisten unterstützen können.

                    BetterDoc empfiehlt nur den für die jeweilige Situation am besten geeigneten Experten. Wir verzichten bewusst auf jegliche vertragliche Bindung mit den empfohlenen Ärzten, um weiterhin unabhängig agieren zu können. Einige der von uns empfohlenen Ärzte kennen unseren Service noch nicht, was aber für Sie bei der Beratung kein Nachteil sein wird.
                                        

                    Use the following pieces of context at the answer to reply with
                    patient_name: {patient_name}
                    patient_info: {patient_info}
                    case_id: {case_id}
                    clinics_names: {clinics_names}
                    medics_names: {medics_names}
                    Refer to what the patient information is about, like the patient's history of migraines and the medication he is taking.
                    :\nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n$context: \n\nQuery: $query\Use provided documents to improve responses\n Answer:''',
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

st.title("Personalized Patient Reply with Llama3")
st.caption("This app allows you to create personalized results with a Llama3 model and apply RAG running locally with Ollama!")



# Input for model name
model_name = st.text_input("Enter the model name", "personal_reply_patient")

patient_name = st.text_input("Enter the patient name", "Miguel Fernandes")
patient_info = st.text_area("Enter the patient information", "Patient has a history of migraines and is currently taking medication for it.")
case_id = st.text_input("Enter the case ID", "A6GX-9A19-XW0W")

# Add a list of clinics
clinics_name = st.text_input("Enter the clinic names", "Neurobaden − Praxis für Neurologie")

# Add a list of medics
medics_name = st.text_input("Enter the medic names", "Dr. med. Andreas Horst")

specialty = st.text_input("Enter the specialty", "Neurologie")

# Create a temporary directory to store the files and the database in the local folder
if not os.path.exists("pers_pat_db"):
    os.mkdir("pers_pat_db")

# Update db_path to include model name
db_path = os.path.join("pers_pat_db", model_name.replace(":", "_"))

# Create an instance of the embedchain App
app = embedchain_bot(db_path, model_name)
# Reset the app function
def reset_app():
    # Reset the app
    app.reset()
    st.success("App reset successfully!")

with st.sidebar:
# Reset button
    if st.button("Reset Data and Context"):
        reset_app()

    # Upload a PDF file
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

    # If a PDF file is uploaded, add it to the knowledge base
    if pdf_file:
        pdf_context = st.text_area("Add context as text to the PDF file")
        if st.button("Add PDF"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(pdf_file.getvalue())    
                app.add(f.name, data_type="pdf_file", metadata={"metadata": pdf_context})
            os.remove(f.name)
            st.success(f"Added {pdf_file.name} to knowledge base!")

    # Add website URL
    website_url = st.text_input("Add a website URL")
    if website_url:
        website_context = st.text_area("Add context as text to the website")
        if st.button("Add Website"):
            st.write("Metadata to be added:", website_context)
            app.add(website_url, data_type='web_page', metadata= {"metadata": website_context})
            st.success("Website added successfully!")

    # Add context as text for medic
    medical_context_for_prompt = None
    medical_context = st.text_area("Add text for improving answer for recommendation of medic")
    if medical_context:
        if st.button("Add Context"):
            medical_context_for_prompt = medical_context
            st.success("Context added successfully!")
            st.write(medical_context_for_prompt)


if patient_info is None:
    patient_info = specialty

# Ask a question about the content provided
# Display the answer
if st.button("Generate Answer"):
    prompt = f"\nPatient Name: {patient_name} \Patient information: {patient_info} \nCase ID: {case_id} \nClinics: {clinics_name} \Medics names:{medics_name} \nSpecialty: {specialty} \n{medical_context_for_prompt}"
    answer = app.query(prompt)
    st.write(answer)



