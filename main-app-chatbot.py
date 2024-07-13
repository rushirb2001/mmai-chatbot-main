from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
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
import zipfile
import sqlite3
from tqdm import tqdm
import tempfile, os
import streamlit as st

st.set_page_config(page_title="Match-Maker-AI", page_icon=":speech_balloon:")

@st.cache_data()
def download_db():    
    if not os.path.exists("supplier-database.db"):
        gdown.download('https://drive.google.com/uc?id=167gji0LKnOJElgIA0flocOI8s_ZFgxGs', 'supplier-database.db', quiet=False)
download_db()

db_uri = f"sqlite:///supplier-database.db"
db = SQLDatabase.from_uri(db_uri)
data = sqlite3.connect("supplier-database.db")
st.session_state.db = db

LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]

def get_schema(_):
    return db.get_table_info()

def get_sql_chain(user_query: str, db: SQLDatabase, chat_history: list):

    with tqdm(total=4, desc="Generating the SQL Query...") as pbar:

        pbar.update(1)
        pbar.set_description("Generating the Template...")
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

            Question: Filter by California State.
            SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE state = 'CA' AND services LIKE '%/creative%production%' OR servicetype LIKE '%production%' OR servicetype LIKE '%/creative%' AND naics LIKE '5414%' OR naics LIKE '7225%';

            Question: Filter by California State.
            SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE state = 'CA' AND services LIKE '%/creative%production%' OR servicetype LIKE '%production%' OR servicetype LIKE '%/creative%' AND naics LIKE '5414%' OR naics LIKE '7225%';

            Question: List companies providing IT services.
            SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE UPPER(services) LIKE UPPER('%IT%') AND naics LIKE '5415%' LIMIT 10;

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
        pbar.update(1)
        pbar.set_description("Generating the Prompt...")
        prompt = ChatPromptTemplate.from_template(template)
        pbar.update(1)
        pbar.set_description("Fetching the LLM Model...")
        llm = ChatGroq(model="llama3-70b-8192", temperature=0, api_key=st.secrets["GROQ_API_KEY"])
        pbar.update(1)
        pbar.set_description("Generating and Sending the Chain...")
        chain = (
            RunnablePassthrough.assign(schema=get_schema)
            | prompt
            | llm
            | StrOutputParser()
        )
        result = chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
        })
        return result
    
def format_businesses_to_markdown(data):
    if not data:
        return "No data available."

    markdown_list = []
    count = 1
    # Loop through each entry and format it
    for item in eval(data):
        print(item)
        if len(item) == 6:  # Ensure each tuple has exactly 6 elements
            company_name, address, city, state, zip_code, services = item
            markdown_list.append(
                f"""
                {count}. **{company_name}**
                    - ***Contact:*** dummy
                    - ***Services Offered:*** {services}\n
                    - ***Address:*** {address}, {city}, {state} - {zip_code}
                """
            )
        else:
            return f"Error: Item at index {count} does not contain exactly 6 elements."
        
        count += 1

    return "\n".join(markdown_list)
    
def get_response(sql_query_response: str):
    if sql_query_response:
        try:
            result = data.execute(sql_query_response).fetchall()
            result = str(result).replace("\\n\\n", "")
            result = str(result).replace("\\n", "")
            
            print(result)

            if result:
                df = pd.DataFrame(eval(result), columns=["Company Name", "Address", "City", "State", "Zip", "Services Offered"], index=np.arange(1, len(eval(result))+1))
                if not len(df) == 0:
                    return format_businesses_to_markdown(result), result, df
                else:
                    return "No Matching Businesses Found.", result, None
            else:
                return "No Matching Businesses Found."
        except Exception as e:
            print(e)
            return "Error: Unable to Retrieve Businesses. Please try again later.1"
    else:
        return "Error: Unable to Retrieve Businesses. Please try again later."

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

@st.experimental_fragment()
def download_file(csv):
    st.download_button("Download Data to CSV File", csv, "data.csv", "csv")

@st.experimental_fragment()
def generate_mk(response):
    st.markdown(response)

@st.experimental_fragment()
def generate_df(df):
    st.dataframe(df, width=2000)

@st.experimental_fragment()
def generate_data(content):
    df = pd.DataFrame(content, columns=["Company Name", "Address", "City", "State", "Zip", "Services Offered"], index=np.arange(1, len(content)+1))
    st.dataframe(df, width=2000)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
    AIMessage(content="Hello! I'm a Match-Making assistant. Write a Query to retrieve the matching Businesses."),
    ]

if "chat_display" not in st.session_state:
    st.session_state.chat_display = [
    AIMessage(content="Hello! I'm a Match-Making assistant. Write a Query to retrieve the matching Businesses."),
    ]

load_dotenv()

with st.sidebar:
    pdf_query_v2 = st.file_uploader(label="Upload a RFP to Retrieve Businesses.", type=["pdf"])
    on = st.toggle("Use Proprietory Databasee")

st.title("Chat with Match-Maker-AI")

# with bottom():
    # row2 = row.row([16, 4], vertical_align="center")
    # user_query = row2.chat_input("Type a message...")
    # sb = row2.button("Clear History", use_container_width=True)

# if sb:
#     st.session_state.chat_history = []
#     st.session_state.chat_display = []
#     st.session_state.chat_history.append(AIMessage(content="Hello! I'm a Match-Making assistant. Write a Query to retrieve the matching Businesses."))
#     st.session_state.chat_display.append(AIMessage(content="Hello! I'm a Match-Making assistant. Write a Query to retrieve the matching Businesses."))


for message in st.session_state.chat_display:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            with st.spinner("Generating Response..."):
                if message.content == "Hello! I'm a Match-Making assistant. Write a Query to retrieve the matching Businesses.":
                    generate_mk(message.content)
                else:
                    mk, _, _ = get_response(message.content)
                    generate_mk(mk)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            generate_mk(message.content)

user_query = st.chat_input("Type your Businesses Query here...", key="user_query")


if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_display.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        generate_mk(user_query)

    with st.chat_message("AI"):
        with st.spinner("Retrieving Businesses..."):
            sql_query_response = get_sql_chain(user_query, st.session_state.db, st.session_state.chat_history)
            # print(sql_query_response)
            response, result, df = get_response(sql_query_response)
            st.session_state.chat_display.append(AIMessage(content=sql_query_response))
        if df is not None:
            st.write("Here are the Matching Businesses:")
        generate_mk(response)
        
        if df is not None:
            csv = df.to_csv().encode("utf-8")
            download_file(csv)

    st.session_state.chat_history.append(BaseMessage(content=sql_query_response, type="AI"))

elif pdf_query_v2 is not None:
    st.session_state.chat_history.append(HumanMessage(content="Uploaded File **"+pdf_query_v2.name+"**. Retrieving Businesses for the RFP File Requirements."))
    st.session_state.chat_display.append(HumanMessage(content="Uploaded File **"+pdf_query_v2.name+"**. Retrieving Businesses for the RFP File Requirements."))
    
    with st.chat_message("Human"):
        generate_mk("Uploaded File **"+pdf_query_v2.name+"**. Retrieving Businesses for the RFP File Requirements.")
        
    with st.chat_message("AI"):
            try:
                with st.spinner("Retrieving Businesses..."):
                    sql_query_response = get_sql_chain(get_pdf_nlp_query(pdf_query_v2), st.session_state.db, st.session_state.chat_history)
                    response, result, df = get_response(sql_query_response)
            except Exception as e:
                try :
                    with st.spinner("Attempting to Retrieve Businesses..."):
                        sql_query_response = get_sql_chain(get_pdf_nlp_query(pdf_query_v2), st.session_state.db, st.session_state.chat_history)
                        response, result, df = get_response(sql_query_response)
                except Exception as e:
                        response = "Error: Unable to Retrieve Businesses. Please try again later."
        
            if response:
                generate_mk(response)
                csv = df.to_csv().encode("utf-8")
                download_file(csv)
                st.session_state.chat_display.append(AIMessage(content=result))
            else:
                st.markdown("No Matching Businesses Found.")

    st.session_state.chat_history.append(BaseMessage(content=sql_query_response, type="AI"))




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
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
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
import zipfile
import sqlite3
from tqdm import tqdm
import tempfile, os
import streamlit as st

st.set_page_config(page_title="Match-Maker-AI", page_icon=":speech_balloon:")

@st.cache_data()
def download_db():    
    if not os.path.exists("supplier-database.db"):
        gdown.download('https://drive.google.com/uc?id=167gji0LKnOJElgIA0flocOI8s_ZFgxGs', 'supplier-database.db', quiet=False)
download_db()

db_uri = f"sqlite:///supplier-database.db"
db = SQLDatabase.from_uri(db_uri)
data = sqlite3.connect("supplier-database.db")
st.session_state.db = db

LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]

def get_schema(_):
    return db.get_table_info()

def get_sql_chain(user_query: str, db: SQLDatabase, chat_history: list):

    with tqdm(total=4, desc="Generating the SQL Query...") as pbar:

        pbar.update(1)
        pbar.set_description("Generating the Template...")
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

            Question: Filter by California State.
            SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE state = 'CA' AND services LIKE '%/creative%production%' OR servicetype LIKE '%production%' OR servicetype LIKE '%/creative%' AND naics LIKE '5414%' OR naics LIKE '7225%';

            Question: Filter by California State.
            SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE state = 'CA' AND services LIKE '%/creative%production%' OR servicetype LIKE '%production%' OR servicetype LIKE '%/creative%' AND naics LIKE '5414%' OR naics LIKE '7225%';

            Question: List companies providing IT services.
            SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE UPPER(services) LIKE UPPER('%IT%') AND naics LIKE '5415%' LIMIT 10;

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
        pbar.update(1)
        pbar.set_description("Generating the Prompt...")
        prompt = ChatPromptTemplate.from_template(template)
        pbar.update(1)
        pbar.set_description("Fetching the LLM Model...")
        llm = ChatGroq(model="llama3-70b-8192", temperature=0, api_key=st.secrets["GROQ_API_KEY"])
        pbar.update(1)
        pbar.set_description("Generating and Sending the Chain...")
        chain = (
            RunnablePassthrough.assign(schema=get_schema)
            | prompt
            | llm
            | StrOutputParser()
        )
        result = chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
        })
        return result
    
def format_businesses_to_markdown(data):
    if not data:
        return "No data available."

    markdown_list = []
    count = 1
    # Loop through each entry and format it
    for item in eval(data):
        print(item)
        if len(item) == 6:  # Ensure each tuple has exactly 6 elements
            company_name, address, city, state, zip_code, services = item
            markdown_list.append(
                f"""
                {count}. **{company_name}**
                    - ***Contact:*** dummy
                    - ***Services Offered:*** {services}\n
                    - ***Address:*** {address}, {city}, {state} - {zip_code}
                """
            )
        else:
            return f"Error: Item at index {count} does not contain exactly 6 elements."
        
        count += 1

    return "\n".join(markdown_list)
    
def get_response(sql_query_response: str):
    if sql_query_response:
        try:
            result = data.execute(sql_query_response).fetchall()
            result = str(result).replace("\\n\\n", "")
            result = str(result).replace("\\n", "")
            
            print(result)

            if result:
                df = pd.DataFrame(eval(result), columns=["Company Name", "Address", "City", "State", "Zip", "Services Offered"], index=np.arange(1, len(eval(result))+1))
                if not len(df) == 0:
                    return format_businesses_to_markdown(result), result, df
                else:
                    return "No Matching Businesses Found.", result, None
            else:
                return "No Matching Businesses Found."
        except Exception as e:
            print(e)
            return "Error: Unable to Retrieve Businesses. Please try again later.1"
    else:
        return "Error: Unable to Retrieve Businesses. Please try again later."

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

@st.experimental_fragment()
def download_file(csv):
    st.download_button("Download Data to CSV File", csv, "data.csv", "csv")

@st.experimental_fragment()
def generate_mk(response):
    st.markdown(response)

@st.experimental_fragment()
def generate_df(df):
    st.dataframe(df, width=2000)

@st.experimental_fragment()
def generate_data(content):
    df = pd.DataFrame(content, columns=["Company Name", "Address", "City", "State", "Zip", "Services Offered"], index=np.arange(1, len(content)+1))
    st.dataframe(df, width=2000)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
    AIMessage(content="Hello! I'm a Match-Making assistant. Write a Query to retrieve the matching Businesses."),
    ]

if "chat_display" not in st.session_state:
    st.session_state.chat_display = [
    AIMessage(content="Hello! I'm a Match-Making assistant. Write a Query to retrieve the matching Businesses."),
    ]

load_dotenv()

with st.sidebar:
    pdf_query_v2 = st.file_uploader(label="Upload a RFP to Retrieve Businesses.", type=["pdf"])
    on = st.toggle("Use Proprietory Databasee")

st.title("Chat with Match-Maker-AI")

# with bottom():
    # row2 = row.row([16, 4], vertical_align="center")
    # user_query = row2.chat_input("Type a message...")
    # sb = row2.button("Clear History", use_container_width=True)

# if sb:
#     st.session_state.chat_history = []
#     st.session_state.chat_display = []
#     st.session_state.chat_history.append(AIMessage(content="Hello! I'm a Match-Making assistant. Write a Query to retrieve the matching Businesses."))
#     st.session_state.chat_display.append(AIMessage(content="Hello! I'm a Match-Making assistant. Write a Query to retrieve the matching Businesses."))


for message in st.session_state.chat_display:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            with st.spinner("Generating Response..."):
                if message.content == "Hello! I'm a Match-Making assistant. Write a Query to retrieve the matching Businesses.":
                    generate_mk(message.content)
                else:
                    mk, _, _ = get_response(message.content)
                    generate_mk(mk)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            generate_mk(message.content)

user_query = st.chat_input("Type your Businesses Query here...", key="user_query")


if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_display.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        generate_mk(user_query)

    with st.chat_message("AI"):
        with st.spinner("Retrieving Businesses..."):
            sql_query_response = get_sql_chain(user_query, st.session_state.db, st.session_state.chat_history)
            # print(sql_query_response)
            response, result, df = get_response(sql_query_response)
            st.session_state.chat_display.append(AIMessage(content=sql_query_response))
        if df is not None:
            st.write("Here are the Matching Businesses:")
        generate_mk(response)
        
        if df is not None:
            csv = df.to_csv().encode("utf-8")
            download_file(csv)

    st.session_state.chat_history.append(BaseMessage(content=sql_query_response, type="AI"))

elif pdf_query_v2 is not None:
    st.session_state.chat_history.append(HumanMessage(content="Uploaded File **"+pdf_query_v2.name+"**. Retrieving Businesses for the RFP File Requirements."))
    st.session_state.chat_display.append(HumanMessage(content="Uploaded File **"+pdf_query_v2.name+"**. Retrieving Businesses for the RFP File Requirements."))
    
    with st.chat_message("Human"):
        generate_mk("Uploaded File **"+pdf_query_v2.name+"**. Retrieving Businesses for the RFP File Requirements.")
        
    with st.chat_message("AI"):
            try:
                with st.spinner("Retrieving Businesses..."):
                    sql_query_response = get_sql_chain(get_pdf_nlp_query(pdf_query_v2), st.session_state.db, st.session_state.chat_history)
                    response, result, df = get_response(sql_query_response)
            except Exception as e:
                try :
                    with st.spinner("Attempting to Retrieve Businesses..."):
                        sql_query_response = get_sql_chain(get_pdf_nlp_query(pdf_query_v2), st.session_state.db, st.session_state.chat_history)
                        response, result, df = get_response(sql_query_response)
                except Exception as e:
                        response = "Error: Unable to Retrieve Businesses. Please try again later."
        
            if response:
                generate_mk(response)
                csv = df.to_csv().encode("utf-8")
                download_file(csv)
                st.session_state.chat_display.append(AIMessage(content=result))
            else:
                st.markdown("No Matching Businesses Found.")

    st.session_state.chat_history.append(BaseMessage(content=sql_query_response, type="AI"))




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
