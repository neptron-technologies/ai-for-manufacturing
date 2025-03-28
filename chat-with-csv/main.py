# # from langchain_experimental.agents import create_csv_agent
# # from langchain_experimental.agents import create_csv_agent
# # from langchain.llms import OpenAI
# # from dotenv import load_dotenv
# # import os
# # import streamlit as st


# # def main():
# #     load_dotenv()

# #     # Load the OpenAI API key from the environment variable
# #     if os.getenv("GOOGLE_API_KEY") is None or os.getenv("GOOGLE_API_KEY") == "":
# #         print("GOOGLE_API_KEY is not set")
# #         exit(1)
# #     else:
# #         print("GOOGLE_API_KEY is set")

# #     st.set_page_config(page_title="Ask your CSV")
# #     st.header("Ask your CSV 📈")

# #     csv_file = st.file_uploader("Upload a CSV file", type="csv")
# #     if csv_file is not None:

# #         agent = create_csv_agent(
# #             OpenAI(temperature=0), csv_file, verbose=True)

# #         user_question = st.text_input("Ask a question about your CSV: ")

# #         if user_question is not None and user_question != "":
# #             with st.spinner(text="In progress..."):
# #                 st.write(agent.run(user_question))


# # if __name__ == "__main__":
# #     main()
# from langchain_experimental.agents import create_csv_agent
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# import os
# import streamlit as st

# # Load environment variables
# load_dotenv()

# # Set up Google Gemini API Key
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     st.error("GOOGLE_API_KEY is not set in environment variables.")
#     st.stop()

# # Initialize Google Gemini (Gemini Pro)
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY, temperature=0,api_version="v1" )

# # Streamlit UI
# st.set_page_config(page_title="Ask your CSV")
# st.header("Ask your CSV 📈")

# # Upload CSV file
# csv_file = st.file_uploader("Upload a CSV file", type="csv")

# if csv_file is not None:
#     # Create an agent for the uploaded CSV with security opt-in
#     agent = create_csv_agent(llm, csv_file, verbose=True, allow_dangerous_code=True,)

#     # Input box for user question
#     user_question = st.text_input("Ask a question about your CSV:")

#     if user_question:
#         with st.spinner(text="Processing..."):
#             try:
#                 response = agent.run(user_question)
#                 st.write(response)
#             except Exception as e:
#                 st.error(f"Error: {e}")



from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd

# Load environment variables
load_dotenv()

# Set up Google Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is not set in environment variables.")
    st.stop()

# Initialize Google Gemini (Gemini Pro)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY, temperature=0, api_version="v1")

# Streamlit UI
st.set_page_config(page_title="Ask your PDF")
st.header("Ask your PDF 📄")

# Upload PDF file
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

if pdf_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getbuffer())
    
    # Load PDF and extract text
    loader = PyMuPDFLoader("temp.pdf")
    docs = loader.load()
    text = "\n".join([doc.page_content for doc in docs])
    
    # Convert text to a DataFrame for processing
    df = pd.DataFrame({"content": text.split("\n")})
    
    # Create an agent for querying the PDF text
    agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
    
    # Input box for user question
    user_question = st.text_input("Ask a question about your PDF:")

    if user_question:
        with st.spinner(text="Processing..."):
            try:
                response = agent.run(user_question)
                st.write(response)
            except Exception as e:
                st.error(f"Error: {e}")
