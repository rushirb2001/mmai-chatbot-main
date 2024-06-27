from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter             
from langchain_groq import ChatGroq
from streamlit_extras import grid, row
from streamlit_extras.bottom_container import bottom
import pandas as pd
import numpy as np
import gdown
import tempfile, os
import streamlit as st

LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]

@st.cache_data
def init_database():
    db_uri = f"sqlite:///supplier-database.db"
    db = SQLDatabase.from_uri(db_uri)
    return db

if "db" not in st.session_state:
    db = init_database()
    st.session_state.db = db


def get_schema(_):
    return db.get_table_info()

def get_sql_chain(db):
    print("Inside get_sql_chain")
    template = """
        You are a Procurement Match-Making Analyst at a company. You are interacting with a user who is asking for companies matching the services he is interested from company's database.
        Based on the table schema below, write a SQL query (sqlite3) that would answer the user's question. Take the conversation history into account.
        
        <SCHEMA>{schema}</SCHEMA>
        
        Conversation History: {chat_history}
        
        Write only the SQL query (sqlite3) and nothing else. Do not wrap the SQL query in any other text, not even backticks. Limit the records to 10.
        Formulate the SQL query (sqlite3) to match only the first FOUR DIGITS of the NAICS code. Use OR instead of AND to match the conditions of search.
        Use all words of the services requested, and the NAICS code to match the services provided by the companies.
        
        For example:
        Question: List the companies that provide creative or production services.
        SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE services LIKE '%/creative%production%' OR servicetype LIKE '%production%' OR servicetype LIKE '%/creative%' AND naics LIKE '5414%' OR naics LIKE '7225%';

        Question: List companies providing IT services.
        SQL Query:
        SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE UPPER(services) LIKE UPPER('%IT%') AND naics LIKE '5415%' LIMIT 10;

        Question: List the next 10 companies providing IT services.
        SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE services LIKE '%IT%' AND naics LIKE '5415%' LIMIT 10 OFFSET 10;

        Question: List companies with ISO certification.
        SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE "ISO Standard" IS NOT NULL;

        Question: List companies in California.
        SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE state = 'CA' LIMIT 10;

        Question: Give me 10 more companies in California.
        SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE state = 'CA' LIMIT 10 OFFSET 10;

        Question: Give me 10 more companies in California.
        SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE state = 'CA' LIMIT 10 OFFSET 20;
        
        Your turn:
        
        Question: {question}. Give me Company Names, Address [Address, City, State, Zip] and services Offered.
        SQL Query:
        """
    print("Template: ")

    print("Prompt")
    prompt = ChatPromptTemplate.from_template(template)
    
    print("LLM")
    llm = ChatGroq(model="llama3-70b-8192", temperature=0, api_key=st.secrets["GROQ_API_KEY"])
    
    print("Chain")
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )
    
def get_response(user_query: str, db: SQLDatabase, chat_history: list):

    print("Inside get_response")
    sql_chain = get_sql_chain(db)
    
    print("SQL Chain: ")
    template = """
            You are a match-maker-ai at a company. You are interacting with a user who is asking for companies matching the services they are interested in from company's database.
            Based on the table schema below, question, sql query, and sql response, Format the the sql response in a numbered list format with Heading for each company and attributes [Format Address City State and Zip in One Line].
            Do not mention the instance of SQL query or any context of it. Contextually format the opening and closing of the response to match the tone and context of the conversation.
            If the SQL response is empty, mention "No Matching Companies Found".
            <SCHEMA>{schema}</SCHEMA>

            Conversation History: {chat_history}
            SQL Query: <SQL>{query}</SQL>
            User question: {question}
            SQL Response: {response}
            """
    
    print("Template: ")
    prompt = ChatPromptTemplate.from_template(template)
    
    print("LLM: ")
    llm = ChatGroq(model="llama3-70b-8192", temperature=0, api_key=st.secrets["GROQ_API_KEY"])
    
    print("Chain: ")
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
        schema=lambda _: db.get_table_info(),
        response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    print("Result: ")
    result = chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })
    print(result)
    return result

def save_uploaded_file(uploaded_file):
    try:
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
        return temp_file_path
    except Exception as e:
        return None
    
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_pdf_nlp_query(pdf_file: list):

    pdf_fx = save_uploaded_file(pdf_file)
    loader = PyPDFLoader(pdf_fx)
    pdf_f = loader.load_and_split()
    faiss_index = FAISS.from_documents(pdf_f, OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536, api_key=st.secrets["OPENAI_API_KEY"]))

    question = "From the RFP document, extract the services sought and their respective NAICS codes. Then, generate a SQL query to retrieve a list of contractors offering these services. Limit the search results to 10."

    template = """
    You are an Service Identifier AI at a match-making company. You are working with a RFP pdf that a User has uploaded and you are tasked 
    to contextually extract the sought services mentioned in the Request for Proposal (RFP) / Request for Information (RFI) document context provided under statement of procurement purpose.
    Following the extraction of the specified services and NAICS codes [Atmost 2 to 3 Codes] extracted from the RFP, create a Natural Language query to find companies offering these services.

    <SCHEMA> {schema} </SCHEMA>

    <CONTEXT>{document_context}</CONTEXT>

    Write only the Natural Language query and nothing else. Do not wrap the query in any other text, not even backticks.

    Question: {question}

    Example:
    Natural Language Query: Find companies offering web development and graphic design services or with possible NAICS codes: [541511] or [541430].

    Your turn:
    Natural Language Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=st.secrets["GROQ_API_KEY"])
    
    # faiss_index.as_retriever() | format_docs
    chain = (
        {"document_context": faiss_index.as_retriever() | format_docs, "schema": get_schema, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = chain.invoke(question)
    print(result)
    return result



if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
    AIMessage(content="Hello! I'm a Match-Making assistant. Write a Query to retrieve the matching Businesses."),
    ]

load_dotenv()

st.set_page_config(page_title="Match-Maker-AI", page_icon=":speech_balloon:")

st.title("Chat with Match-Maker-AI")
    
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            with st.spinner("Generating Response..."):
                st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

with st.sidebar:
    pdf_query = st.file_uploader(label="Upload a RFP to Retrieve Businesses.", type=["pdf"])

with bottom():
    row2 = row.row([8, 2], vertical_align="center")
    user_query = row2.chat_input("Type a message...")
    sb = row2.button("Clear History", use_container_width=True)

if sb:
    st.session_state.chat_history = []
    st.session_state.chat_history.append(AIMessage(content="Hello! I'm a Match-Making assistant. Write a Query to retrieve the matching Businesses."))

if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        with st.spinner("Retrieving Businesses..."):
            response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))

elif pdf_query is not None:
    st.session_state.chat_history.append(HumanMessage(content="Uploaded File **"+pdf_query.name+"**. Retrieving Businesses for the RFP File Requirements."))
    
    with st.chat_message("Human"):
        st.markdown("Uploaded File **"+pdf_query.name+"**. Retrieving Businesses for the RFP File Requirements.")
        
    with st.chat_message("AI"):
            try:
                with st.spinner("Retrieving Businesses..."):
                    response = get_response(get_pdf_nlp_query(pdf_query), st.session_state.db, st.session_state.chat_history)
            except Exception as e:
                print(e)
                try :
                    with st.spinner("Attempting to Retrieve Businesses..."):
                        response = get_response(get_pdf_nlp_query(pdf_query), st.session_state.db, st.session_state.chat_history)
                except Exception as e:
                        response = "Error: Unable to Retrieve Businesses. Please try again later."
        
            if response:
                st.markdown(response)
            else:
                st.markdown("No Matching Businesses Found.")    
    st.session_state.chat_history.append(AIMessage(content=response))





# """
#     Miscellaneous Codes : Might be useful later

#     # (1) : For connecting to MySQL database - Streamlit Sidebar
#     ---------------------------------------------------------
#     with st.sidebar:
#     st.subheader("Settings")
#     st.write("This is a simple chat application using MySQL. Connect to the database and start chatting.")
    
#     st.text_input("Host", value="localhost", key="Host")
#     st.text_input("Port", value="3306", key="Port")
#     st.text_input("User", value="root", key="User")
#     st.text_input("Password", type="password", value="admin", key="Password")
#     st.text_input("Database", value="Chinook", key="Database")
    
#     if st.button("Connect"):
#         with st.spinner("Connecting to database..."):
#             db = init_database(
#                 st.session_state["User"],
#                 st.session_state["Password"],
#                 st.session_state["Host"],
#                 st.session_state["Port"],
#                 st.session_state["Database"]
#             )
#             st.session_state.db = db
#         st.success("Connected to database!")
#     ---------------------------------------------------------

#     # (2) : LLM Model configuration options
#     ---------------------------------------------------------
#     llm = ChatOpenAI(model="gpt-4-0125-preview")
#     llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
#     ---------------------------------------------------------

#     # (3) : CSS Styling for PDF Uploader
#     ---------------------------------------------------------
#     css = \"""
#         <style>
#             [data-testid='stFileUploader'] {
#                 width: max-content;
#             }
#             [data-testid='stFileUploader'] section {
#                 padding: 0px;
#             }
#             [data-testid='stFileUploader'] section > input + div {
#                 display: none;
#             }
#             [data-testid='stFileUploader'] section + div {
#                 display: none;
#                 padding-top: 0px;
#             }
#         </style>
#         \"""
#     st.markdown(css, unsafe_allow_html=True)
#     ---------------------------------------------------------

#     # (4) : MySQL Database Connection URL Loader
#     ---------------------------------------------------------
#     def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
#         db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}" 
#         return SQLDatabase.from_uri(db_uri)
#     ---------------------------------------------------------


# """
