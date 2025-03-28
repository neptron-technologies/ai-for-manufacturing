from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import pyodbc

# Load environment variables
load_dotenv()

# Initialize database connection
def init_database(host: str, database: str) -> SQLDatabase:
    db_uri = f"mssql+pyodbc://@{host}/{database}?trusted_connection=yes&driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
    return SQLDatabase.from_uri(db_uri)

# Save chat message to database
def save_to_memory(chat_id, role, message):
    conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={st.session_state['Host']};"
        f"DATABASE={st.session_state['Database']};"
        f"Trusted_Connection=yes;"
        f"TrustServerCertificate=yes;"
    )
    cursor = conn.cursor()
    sql = "INSERT INTO chat_memory (chat_id, role, message) VALUES (?, ?, ?)"
    cursor.execute(sql, (chat_id, role, message))
    conn.commit()
    cursor.close()
    conn.close()

# Load chat history from database
def load_chat_history(chat_id):
    conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={st.session_state['Host']};"
        f"DATABASE={st.session_state['Database']};"
        f"Trusted_Connection=yes;"
        f"TrustServerCertificate=yes;"
    )
    cursor = conn.cursor()
    sql = "SELECT role, message FROM chat_memory WHERE chat_id = ? ORDER BY timestamp ASC"
    cursor.execute(sql, (chat_id,))
    messages = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return [AIMessage(content=m[1]) if m[0] == "AI" else HumanMessage(content=m[1]) for m in messages]

# Generate SQL query from user input
def get_sql_chain(db):
    template = """
    You are an expert data analyst & SQL Expert. Based on the table schema below, generate a SQL query for the user's question.
    
    <SCHEMA>{schema}</SCHEMA>
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    Question: {question}
    SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", client=genai, temperature=0)
    
    return RunnablePassthrough.assign(schema=lambda _: db.get_table_info()) | prompt | llm | StrOutputParser()

# Process user query and generate response
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
    You are a SQL assistant & SQL Expert. Generate a natural language response based on the table schema, user query, SQL query, and response.
    
    <SCHEMA>{schema}</SCHEMA>
    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        ) | prompt | llm | StrOutputParser()
    )
    
    return chain.invoke({"question": user_query, "chat_history": chat_history})

# Streamlit UI Setup
st.set_page_config(page_title="Chat with SQL Server", page_icon=":speech_balloon:")
st.title("Chat with SQL Server Database")

with st.sidebar:
    st.subheader("Database Settings")
    st.session_state["Host"] = st.text_input("Host", value="localhost")
    st.session_state["Database"] = st.text_input("Database", value="Chinook")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            st.session_state["db"] = init_database(
                st.session_state["Host"],
                st.session_state["Database"]
            )
            st.session_state["chat_history"] = []
            st.success("Connected to database!")

# Initialize chat history
chat_id = "default_session"
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = load_chat_history(chat_id)

# Display chat messages
for message in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        st.markdown(message.content)

# Handle user input
user_query = st.chat_input("Ask a question about your database...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    save_to_memory(chat_id, "Human", user_query)
    
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))
    save_to_memory(chat_id, "AI", response)
