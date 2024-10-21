from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter             
from langchain_groq import ChatGroq
from streamlit_extras import grid, row
from streamlit_extras.bottom_container import bottom
from streamlit_extras.row import row
import pandas as pd
import numpy as np
import gdown
import zipfile, time
import sqlite3, re
from tqdm import tqdm
import tempfile, os
import streamlit as st
import folium
from streamlit_folium import folium_static, st_folium
import math
import os
from sklearn.cluster import KMeans
from random import random
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import ast
import itertools

#-------------------------------------------------------------------------------------------------------------------------------------#
## Streamlit Configuration and Database Download Functions
# Set the page configuration and API keys
st.set_page_config(
    page_title="Match-Maker-AI", 
    page_icon=":speech_balloon:", 
    layout="wide"
    )
st.title("Match-Maker-AI®")

LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Download the database file, decorated with Streamlit's caching
@st.cache_data()
def download_db():
    """
    Function: Download the database file from Google Drive

    Returns:
        None

    Other:
    # if not os.path.exists("supplier-database.db"):
        # gdown.download('https://drive.google.com/uc?id=167gji0LKnOJElgIA0flocOI8s_ZFgxGs', 'supplier-database.db', 
        # quiet=False)
        # https://drive.google.com/file/d/1XvahREHGxTcQkq1S7ZIT5-7wqH-7I5ai/view?usp=drive_link
    """
    if not os.path.exists("supplier_database-v3.db"):
        gdown.download('https://drive.google.com/uc?id=1XvahREHGxTcQkq1S7ZIT5-7wqH-7I5ai', 'supplier_database-v3.db', quiet=False)
    # link = https://drive.google.com/file/d/18chgV_UwlWSYTEP1W579vYZWQhsBRrEI/view?usp=drive_link
    if not os.path.exists("search_filter_data.csv"):
        gdown.download('https://drive.google.com/uc?id=18chgV_UwlWSYTEP1W579vYZWQhsBRrEI', 'search_filter_data.csv', quiet=False)
download_db()

# Load the database and create a connection
@st.cache_resource()
def load_db(propdefaults):
    """
    Function: Load the database and create a connection

    Returns:
        db: SQLDatabase object
        data: sqlite3 connection object

    Other:
    # db_uri = f"sqlite:///supplier-database.db"
    # data = sqlite3.connect("supplier-database.db")
    """
    if propdefaults:
        db_uri = f"sqlite:///supplier_database-v3.db"
        database = SQLDatabase.from_uri(db_uri)
        conn = sqlite3.connect("supplier_database-v3.db", check_same_thread=False)
        return database, conn
    else:
        db_uri = f"sqlite:///supplier_prop.db"
        database = SQLDatabase.from_uri(db_uri)
        conn = sqlite3.connect("supplier_prop.db", check_same_thread=False)
        return database, conn
st.session_state.db, st.session_state.data = load_db(True)
db, data = st.session_state.db, st.session_state.data
#-------------------------------------------------------------------------------------------------------------------------------------#


#-------------------------------------------------------------------------------------------------------------------------------------#
## SQL Query Generation Functions


# Function to get the table schema
def get_schema(_):
    """
    Function: Get the table schema

    Returns:
        db.get_table_info(): Table schema information
    """
    print(db.get_table_info())
    return db.get_table_info()

# Function to get the SQL query chain

def get_sql_chain(user_query: str, db: SQLDatabase, chat_history: list):

    """
    Function: Get the SQL query chain

    Args:
        user_query: User's query
        db: SQLDatabase object
        chat_history: Chat history
    
    Returns:
        result: SQL query response
    """
    # The SQL Query should be formulated to match the services provided by the companies, and should also match the conditions of search.

    with tqdm(total=4, desc="Generating the SQL Query...") as pbar:

        pbar.update(1)
        pbar.set_description("Generating the Template...")
        template = """
            <SYS> 
            As a Master SQL Generator at a company, you excel in Filtering, Ranking, Sorting, and Retrieving Data from a SQLite3 Database. You are currently interacting with a user who is seeking companies that offer specific services from your company's database. Given the table schema provided below and the details from your ongoing conversation with the user, craft a SQL query in SQLite3 that would accurately respond to the user's inquiry. Ensure the query is tailored to the user's specific needs based on the conversation history, or if the question is new and not previously discussed, tailor the query according to the table schema provided.
            </SYS>
            
            <SCHEMA>{schema}</SCHEMA>
            
            <HISTORY>{chat_history}</HISTORY>
            
            <INST> 
            1. Write only the SQL query (sqlite3) and no other text or explanations. Do not wrap the SQL query in any other text, not even backticks. 
            2. Limit the records retrieved to 10. 
            3. Ensure the query matches only the first FOUR DIGITS of the NAICS code.
            4. Avoid using the AND operator in the SQL query.
            5. Include all requested service keywords and match them with the NAICS code to filter companies providing these services.
            6. Retrieve only the following columns: company, address, city, state, zip, servicetype.
            7. Utilize NESTED SQL queries to filter data based on the search conditions.
            8. Implement the ORDER BY clause to sort the results according to a column relevant to the search.

            </INST>
            
            For example:
            Question: List the companies that provide creative or production services.
            SQL Query:  SELECT company, address, city, state, zip, servicetype 
                        FROM supplierdb 
                        WHERE UPPER(services) LIKE UPPER('%/creative%production%')
                        OR UPPER(servicetype) LIKE UPPER('%production%')
                        OR UPPER(servicetype) LIKE UPPER('%/creative%')
                        OR naics LIKE '5414%' OR naics LIKE '7225%' 
                        ORDER BY company 
                        LIMIT 10;

            Question: Filter by California.
            SQL Query: SELECT company, address, city, state, zip, servicetype 
                        FROM 
                        (
                        SELECT * 
                        FROM supplierdb 
                        WHERE state = 'CA'
                        ) 
                        WHERE UPPER(services) LIKE UPPER('%/creative%production%') 
                        OR UPPER(servicetype) LIKE UPPER('%production%') 
                        OR UPPER(servicetype) LIKE UPPER('%/creative%')
                        OR naics LIKE '5414%' 
                        OR naics LIKE '7225%' 
                        ORDER BY state 
                        LIMIT 10;

            Question: Limit by California.
            SQL Query: SELECT company, address, city, state, zip, servicetype 
                        FROM 
                        (
                        SELECT * 
                        FROM supplierdb 
                        WHERE state = 'CA'
                        ) 
                        WHERE 
                        UPPER(services) LIKE UPPER('%/creative%production%') 
                        OR UPPER(servicetype) LIKE UPPER('%production%') 
                        OR UPPER(servicetype) LIKE UPPER('%/creative%') 
                        OR naics LIKE '5414%' 
                        OR naics LIKE '7225%'
                        ORDER BY state 
                        LIMIT 10;
            
            Question: Filter by New York.
            SQL Query: SELECT company, address, city, state, zip, servicetype 
                        FROM 
                        (
                        SELECT * 
                        FROM supplierdb 
                        WHERE state = 'NY'
                        ) 
                        WHERE 
                        UPPER(services) LIKE UPPER('%/creative%production%') 
                        OR UPPER(servicetype) LIKE UPPER('%production%') 
                        OR UPPER(servicetype) LIKE UPPER('%/creative%') 
                        OR naics LIKE '5414%' 
                        OR naics LIKE '7225%'
                        ORDER BY state 
                        LIMIT 10;
            
            Question: Limit by New York city.
            SQL Query: SELECT company, address, city, state, zip, servicetype
                        FROM
                        (
                        SELECT * 
                        FROM supplierdb 
                        WHERE city = 'New York'
                        ) 
                        WHERE 
                        UPPER(services) LIKE UPPER('%/creative%production%') 
                        OR UPPER(servicetype) LIKE UPPER('%production%') 
                        OR UPPER(servicetype) LIKE UPPER('%/creative%') 
                        OR naics LIKE '5414%' 
                        OR naics LIKE '7225%'
                        ORDER BY state 
                        LIMIT 10;
                        
            Question: Filter by California and women ownership.
            SQL Query: SELECT company, address, city, state, zip, servicetype 
                        FROM 

                        (
                            SELECT * 
                            FROM supplierdb 
                            WHERE state = 'CA'
                            AND ownership = 'Women-owned') 
                            WHERE (UPPER(services) LIKE UPPER('%/creative%production%'
                        ) 

                        OR UPPER(servicetype) LIKE UPPER('%production%') 
                        OR UPPER(servicetype) LIKE UPPER('%/creative%'))
                        OR naics LIKE '5414%' 
                        OR naics LIKE '7225%' 
                        ORDER BY state 
                        LIMIT 10;

            Question: Limit by Los Angeles and ISO Standard 9000.
            SQL Query: SELECT company, address, city, state, zip, servicetype 
                        FROM 

                        (
                        SELECT * 
                        FROM supplierdb 
                        WHERE city = 'Los Angeles' AND "ISO Standard" = 'ISO 9000'
                        ) 

                        WHERE 
                        (UPPER(services) LIKE UPPER('%/creative%production%') 
                        OR UPPER(servicetype) LIKE UPPER('%production%') 
                        OR UPPER(servicetype) LIKE UPPER('%/creative%')) 
                        OR naics LIKE '5414%' 
                        OR naics LIKE '7225%' 
                        ORDER BY city 
                        LIMIT 10;
            
            Question: Filter by ITAR Registered.
            SQL Query: SELECT company, address, city, state, zip, servicetype 
                        FROM 

                        (
                        SELECT * 
                        FROM supplierdb 
                        WHERE "ITAR Registration" = 'Registered'
                        ) 

                        WHERE 
                        (UPPER(services) LIKE UPPER('%/creative%production%') 
                        OR UPPER(servicetype) LIKE UPPER('%production%') 
                        OR UPPER(servicetype) LIKE UPPER('%/creative%'))
                        OR naics LIKE '5414%' 
                        OR naics LIKE '7225%' 
                        ORDER BY company 
                        LIMIT 10;

            Question: Limit by ITAR Registered.
            SQL Query: SELECT company, address, city, state, zip, servicetype 
                    FROM 

                    (
                    SELECT * 
                    FROM supplierdb 
                    WHERE "ITAR Registration" = 'Registered'
                    ) 

                    WHERE (UPPER(services) LIKE UPPER('%/creative%production%') 
                    OR UPPER(servicetype) LIKE UPPER('%production%') 
                    OR UPPER(servicetype) LIKE UPPER('%/creative%'))
                    OR naics LIKE '5414%' 
                    OR naics LIKE '7225%' 
                    ORDER BY company 
                    LIMIT 10;

            Question: List companies providing IT services.
            SQL Query: SELECT company, address, city, state, zip, servicetype 
                    FROM 

                    (
                    SELECT * 
                    FROM supplierdb 
                    WHERE UPPER(services) LIKE UPPER('%IT%')
                    ) 

                    WHERE naics LIKE '5415%'
                    ORDER BY company 
                    LIMIT 10;

            Question: List the next 10 companies providing IT services.
            SQL Query: SELECT company, address, city, state, zip, servicetype 
                    FROM 
                    FROM supplierdb 
                    WHERE 
                    UPPER(services) LIKE UPPER('%IT%') 
                    OR naics LIKE '5415%'
                    ORDER BY company 
                    LIMIT 10, 10;

            Question: List companies with ISO certification.
            SQL Query: SELECT company, address, city, state, zip, servicetype   
                        FROM supplierdb 
                        WHERE "ISO Standard" IS NOT NULL
                        ORDER BY company 
                        LIMIT 10;

            Question: List companies in California.
            SQL Query: SELECT company, address, city, state, zip, servicetype 
                        FROM supplierdb 
                        WHERE state = 'CA' 
                        ORDER BY state 
                        LIMIT 10;

            Question: Give me 10 more companies in California.
            SQL Query: SELECT company, address, city, state, zip, servicetype 
                        FROM supplierdb 
                        WHERE state = 'CA' 
                        ORDER BY state 
                        LIMIT 10, 10;

            Question: Give me 10 more companies in California.
            SQL Query: SELECT company, address, city, state, zip, servicetype   
                        FROM supplierdb 
                        WHERE state = 'CA' 
                        ORDER BY state 
                        LIMIT 20, 10;
            
            Your turn:
            
            Question: {question}. Give me company, address, city, state, zip, servicetype.
            SQL Query:
            """
        pbar.update(1)
        pbar.set_description("Generating the Prompt...")
        prompt = ChatPromptTemplate.from_template(template)
        pbar.update(1)
        pbar.set_description("Fetching the LLM Model...")
        # llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, api_key=GROQ_API_KEY)
        llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, api_key=OPENAI_API_KEY)
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
        print(result)
        return result
#-------------------------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------------------------#
## Business Data Formatting Functions
# Function to format the businesses into a markdown list
def format_businesses_to_markdown(data: str):
    """
    Function: Format the businesses into a markdown list

    Args:
        data: Business data

    Returns:
        markdown_list: Formatted businesses in markdown list
    """

    if not data:
        return "No data available."

    markdown_list = []
    count = 1
    # Loop through each entry and format it
    for item in eval(data):
        # print(item)
        if len(item) == 6:  # Ensure each tuple has exactly 6 elements
            company_name, address, city, state, zip_code, services = item
            contact = "".join(np.random.choice(list("0123456789"), 10))
            markdown_list.append(
                f"""
                {count}. **{company_name}**
                    - ***Contact:*** +1 ({contact[:3]}) {contact[3:6]}-{contact[6:]}
                    - ***Services Offered:*** {services}\n
                    - ***Address:*** {address}, {city}, {state} - {zip_code}
                """
            )
        else:
            return f"Error: Item at index {count} does not contain exactly 6 elements."
        
        count += 1

    return "\n".join(markdown_list)

# Function to get the response from the SQL query

def get_response(sql_query_response: str):
    """
    Function: Get the response from the SQL query

    Args:
        sql_query_response: SQL query response

    Returns:
        result: SQL query result
    """
    # print(st.session_state.db)

    # print(st.session_state.data.execute("""SELECT * 
    #                         FROM 
    #                         (
    #                             SELECT * 
    #                             FROM supplierdb 
    #                             WHERE city = 'Brooklyn' 
    #                             AND ownership LIKE '%Women-Owned%'
    #                         ) 
    #                         WHERE 
    #                         UPPER(services) LIKE UPPER('%advertising%') 
    #                         OR UPPER(services) LIKE UPPER('%marketing%') 
    #                         OR naics LIKE '5418%' 
    #                         OR naics LIKE '5416%'
    #                         ORDER BY company 
    #                         LIMIT 10;""").fetchall())

    if sql_query_response:
        try:
            result = st.session_state.data.execute(sql_query_response).fetchall()
            result = str(result).replace("\\n\\n", "")
            result = str(result).replace("\\n", "")

            df_result = st.session_state.data.execute(str(sql_query_response).replace('LIMIT 10', '')).fetchall()
            df_result = str(df_result).replace("\\n\\n", "")
            df_result = str(df_result).replace("\\n", "")

            if len(eval(result)) > 0 or len(result) > 0:
                df = pd.DataFrame(eval(df_result), columns=["Company Name", "Address", "City", "State", "Zip", "Services Offered"], index=np.arange(1, len(eval(df_result))+1))
                if not len(df) == 0:
                    print("If FRAME 1")
                    return [format_businesses_to_markdown(result), result, df]
                else:
                    print("Else FRAME 1")
                    return ["No Matching Businesses Found.", result, pd.DataFrame()]
            else:
                print("Else FRAME 2")
                return ["No Matching Businesses Found.", result, pd.DataFrame()]
        except Exception as e:
            print(e)
            print("Error: Unable to Retrieve Businesses. Please try again later.")
            return ["Error: Unable to Retrieve Businesses. Please try again later.1", [], pd.DataFrame()]
    else:
        return "Error: Unable to Retrieve Businesses. Please try again later."
#-------------------------------------------------------------------------------------------------------------------------------------#


#-------------------------------------------------------------------------------------------------------------------------------------#
## PDF NLP Query Generation Functions
# Function to save the uploaded file
def save_uploaded_file(uploaded_file):
    """
    Function: Save the uploaded file

    Args:
        uploaded_file: Uploaded file

    Returns:
        temp_file_path: Temporary file path
    """
    try:
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
        return temp_file_path
    except Exception as e:
        return None

# Function to format the documents

def format_docs(docs: str):
    """
    Function: Format the documents

    Args:
        docs: Documents

    Returns:
        Formatted documents
    """
    return "\n\n".join(doc.page_content for doc in docs)

# Function to get the PDF NLP query

def get_pdf_nlp_query(pdf_file: list):

    reader = PdfReader(save_uploaded_file(pdf_file))

    text = ""
    count = 0
    for page in reader.pages:
        text += page.extract_text() + "\n"
        count += 1
        if count == 2:
            break

    question = "Analyze the attached RFP document and extract the Procurement Purpose of Service/Commodity/Information or Commodity Solicitation/Work Category. Provide the list in bullet points, ensuring each point briefly describes a distinct commodity/service required by the RFP without including any descriptive or additional content."

    template = """
    You are an Entity Extractor AI at a match-making company to extract the required type of Procured Service or Commodity or Solicitation of Information/ Commodity or Work Category. You are working with a RFP pdf that a User has uploaded and you are tasked 
    to contextually extract the sought services mentioned in the Request for Proposal (RFP) / Request for Information (RFI) document context provided under statement of procurement purpose.
    Following the extraction of the specified services OR naics codes [Atmost 2 to 3 Codes] extracted from the RFP, create a Natural Language query to find companies offering these services.
-
    <SCHEMA> {schema} </SCHEMA>

    <CONTEXT> {document} </CONTEXT>

    Write only the Natural Language query and nothing else. Do not wrap the query in any other text, not even backticks.

    Question: {question}

    Example:
    Natural Language Query: Find companies offering web development and graphic design services or with possible NAICS codes: [541511] or [541430].

    Your turn:
    Natural Language Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, api_key=GROQ_API_KEY)
    # llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
    
    chain = (
        {"document": RunnablePassthrough(), "schema": get_schema, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = chain.invoke({"document": text, "question": question})
    print(result)
    return result

# def get_pdf_nlp_query(pdf_file: list):

#     loader = PyPDFLoader(save_uploaded_file(pdf_file))
#     pdf_f = loader.load_and_split()

#     faiss_index = FAISS.from_documents(pdf_f, OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536, api_key=OPENAI_API_KEY))

#     question = "Analyze the attached RFP document and extract the Procurement Purpose of Service/Commodity/Information or Commodity Solicitation/Work Category. Provide the list in bullet points, ensuring each point briefly describes a distinct commodity/service required by the RFP without including any descriptive or additional content."

#     template = """
#     You are an Entity Extractor AI at a match-making company to extract the required type of Procured Service or Commodity or Solicitation of Information/ Commodity or Work Category. You are working with a RFP pdf that a User has uploaded and you are tasked 
#     to contextually extract the sought services mentioned in the Request for Proposal (RFP) / Request for Information (RFI) document context provided under statement of procurement purpose.
#     Following the extraction of the specified services OR naics codes [Atmost 2 to 3 Codes] extracted from the RFP, create a Natural Language query to find companies offering these services.
# -
#     <SCHEMA> {schema} </SCHEMA>

#     <CONTEXT> {document_context} </CONTEXT>

#     Write only the Natural Language query and nothing else. Do not wrap the query in any other text, not even backticks.

#     Question: {question}

#     Example:
#     Natural Language Query: Find companies offering web development and graphic design services or with possible NAICS codes: [541511] or [541430].

#     Your turn:
#     Natural Language Query:
#     """
    
#     prompt = ChatPromptTemplate.from_template(template)
    
#     llm = ChatGroq(model="llama3-70b-8192", temperature=0, api_key=GROQ_API_KEY)
#     # llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    
#     # faiss_index.as_retriever() | format_docs
#     chain = (
#         {"document_context": faiss_index.as_retriever(search_type="mmr") | format_docs, "schema": get_schema, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
#     result = chain.invoke(question)
#     # print(result)
#     return result

def get_pdf_query(pdf_file: list):

    loader = PyPDFLoader(save_uploaded_file(pdf_file))
    # loader = PyPDFParser(save_uploaded_file(pdf_file))

    docs = loader.load()

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                        chunk_size=10000, chunk_overlap=0
        )
    split_docs = text_splitter.split_documents(docs)

    prompt_template = """
            Extract the service type/ oppurtunity type (SECTOR) required by the following RFP, ignroe the rest of the content.:
            {text}
            SERVICE TYPE/ OPPURTUNITY TYPE (SECTOR):
            """
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        """
        Your job is to append the list of major service type/ oppurtunity type (SECTOR) requested by the RFP, ignoring other document specifications.
        We have provided an existing list up to a certain point: {existing_answer}
        ------------
        {text}
        ------------
        Given the new context, refine the original list. If the context isn't useful, return the original list.
        """
    )
    refine_prompt = PromptTemplate.from_template(refine_template)

    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, api_key=GROQ_API_KEY)
    # llm = ChatGroq(model="llama3-70b-8192", temperature=0, api_key=GROQ_API_KEY)
    # llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key="input_documents",
        output_key="output_text",
        verbose=True
    )

    result = chain.invoke({"input_documents": split_docs}, return_only_outputs=True)

    return result["output_text"]

#-------------------------------------------------------------------------------------------------------------------------------------#

def haversine_distance(lat1, lon1, lat2, lon2):
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = 3956 * c  # radius of earth in miles

    return distance

def create_map():
    # Create the map with Google Maps
    usa_center = [39.8283, -94.5795]  # Latitude and Longitude
    initial_zoom = 3  # Zoom level suitable for viewing the entire USA

    # Create the map object centered on the USA
    map_obj = folium.Map(
        location=usa_center,
        zoom_start=initial_zoom,
        tiles=None  # We'll add custom tiles next
    )

    folium.TileLayer("https://{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}", 
                    attr="google", 
                    name="Google Maps", 
                    overlay=True, 
                    control=True, 
                    subdomains=["mt0", "mt1", "mt2", "mt3"]).add_to(map_obj)
    return map_obj
    
def add_markers(map_obj, locations, popup_list=None):
    if popup_list is  None:
        # Add markers for each location in the DataFrame
        for lat, lon in locations:
            folium.Marker([lat, lon]).add_to(map_obj)
    else:
        for i in range(len(locations)):
            lat, lon = locations[i]
            popup = popup_list[i]
            folium.Marker([lat, lon], popup=popup).add_to(map_obj)

    # Fit the map bounds to include all markers
    south_west = [min(lat for lat, _ in locations) - 0.02, min(lon for _, lon in locations) - 0.02]
    north_east = [max(lat for lat, _ in locations) + 0.02, max(lon for _, lon in locations) + 0.02]
    map_bounds = [south_west, north_east]
    map_obj.fit_bounds(map_bounds)

    return map_obj

def find_closest_and_furthest_pairs(df, distance_func):

    # Create all possible pairs of points
    pairs = list(itertools.combinations(df['lat_lng'], 2))

    # Calculate distances between each pair of points
    distances = [distance_func(p[0][0], p[0][1], p[1][0], p[1][1]) for p in pairs]

    # Find closest and furthest distances
    closest_distance = min(distances)
    closest_pair = pairs[distances.index(closest_distance)]
    furthest_distance = max(distances)
    furthest_pair = pairs[distances.index(furthest_distance)]

    return (closest_distance, closest_pair, furthest_distance, furthest_pair)

def calculate_center_coords(coords_list):
    lat_mean = sum([coords[0] for coords in coords_list]) / len(coords_list)
    lng_mean = sum([coords[1] for coords in coords_list])  / len(coords_list)

    return [round(lat_mean, 6), round(lng_mean, 6)]

def add_center_marker(map_obj, lat, lon, color='red', icon='star', popup=None):
    new_marker = folium.Marker([lat, lon], icon=folium.Icon(color=color, icon=icon), popup=popup)
    map_obj.add_child(new_marker)

    return map_obj

def add_lines_to_center(map_obj, locations_col, center_lat, center_lon, color='red', group_name='Lines'):
    line_group = folium.FeatureGroup(name=group_name)
    for lat_lng in locations_col:
        folium.PolyLine(locations=[lat_lng, [center_lat, center_lon]], color=color).add_to(line_group)
    map_obj.add_child(line_group)

    return map_obj

def cluster_latlng(coordinates_list, n_clusters):
    # Convert the latitude and longitude coordinates to a NumPy array
    X = np.array(coordinates_list)

    # Initialize the KMeans object with the number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    # Fit the KMeans model to the data
    kmeans.fit(X)

    # Get the cluster labels for each data point
    labels = kmeans.labels_

    # Get the cluster centroids
    centroids = kmeans.cluster_centers_

    # Return the cluster labels and centroids
    return labels, centroids

def map_resize(map_obj, locations):
    # Fit the map bounds to include all markers
    south_west = [min(lat for lat, _ in locations) - 0.02, min(lon for _, lon in locations) - 0.02]
    north_east = [max(lat for lat, _ in locations) + 0.02, max(lon for _, lon in locations) + 0.02]
    map_bounds = [south_west, north_east]
    map_obj.fit_bounds(map_bounds)

    return map_obj

def miles_to_meters(miles):
    return miles * 1609.34

def get_max_by_group(df, group_col, max_col):
    return df.groupby(group_col)[max_col].max()

@st.fragment()
def download_file(df, u_key):
    df = pd.DataFrame(df)
    csv = df.to_csv().encode("utf-8")
    date = pd.Timestamp.now().strftime("%Y-%m-%d")
    filename = f"businesses_{date}.csv"
    st.download_button("Download Data to CSV File", csv, filename, "csv", key=u_key)

# def stream(content):
#     for word in content.split(" "):
#         yield word + " "
#         time.sleep(0.001)

@st.fragment()
def generate_mk_ai(response, check):
    if check >= 5:
        infor = st.container(height=685)
        with infor:
            infor.markdown(response)
    else:
        st.markdown(response)

def generate_mk(content):
    st.markdown(content)

@st.fragment()
def generate_df(df):
    st.dataframe(df, width=2000)

@st.fragment()
def generate_data(content):
    df = pd.DataFrame(content, columns=["Company Name", "Address", "City", "State", "Zip", "Services Offered"], index=np.arange(1, len(content)+1))
    st.dataframe(df, width=2000)

@st.fragment()
def create_map_whole(m, df):

    with tqdm(total=3, desc="Generating the Map...") as pbar:
        # df['coordinates'] = df['coordinates'].apply(eval)
        pbar.update(1)
        if len(df) < 1000:
            m = add_markers(m, df['coordinates'], popup_list=None)
        else:
            m = map_resize(m, df['coordinates'])
            labels, centroids = cluster_latlng(df['coordinates'].to_list(), 10)
            df['cluster_label'] = labels
            df['cluster_center'] = df['cluster_label'].apply(lambda x: [round(centroids[x][0], 6), round(centroids[x][1], 6)])
            df = df.sort_values('cluster_label')
            df['distance_cluster_center_miles'] = df.apply(lambda x: haversine_distance(x['coordinates'][0], x['coordinates'][1], x['cluster_center'][0], x['cluster_center'][1]), axis=1)
            
            max_distances = get_max_by_group(df, 'cluster_label', 'distance_cluster_center_miles')
            # Add cluster center
            for i in range(len(centroids)):
                c = centroids[i]
                add_center_marker(m, c[0], c[1], popup='Cluster {} Center Point'.format(i))
            
            # Add circle markers for each centroid
            for i in range(len(centroids)):
                center = centroids[i]
                radius = miles_to_meters(max_distances[i]) + 500
                folium.Circle(location=center, radius=radius, color='red', fill_color='red', fill_opacity=0.2).add_to(m)

        pbar.update(1)
        folium_static(m, height=460, width=540)
        pbar.update(1)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
    AIMessage(content="Hello! I'm a Match-Making assistant. Describe your requirements to retrieve the matching Businesses."),
    ]

if "chat_display" not in st.session_state:
    st.session_state.chat_display = [
    AIMessage(content="Hello! I'm a Match-Making assistant. Describe your requirements to retrieve the matching Businesses."),
    ]

if "pdf_query" not in st.session_state:
    st.session_state.pdf_query = None

load_dotenv()

with st.sidebar:
    HORIZONTAL = "talin_labs_logo-horizontal.png"
    ICON = "talin_labs_logo.png"

    st.logo(HORIZONTAL, icon_image=ICON)
    
    st.markdown("---")
    st.write("# Welcome to Match-Maker-AI®")

    st.markdown("---")
    st.write("# A Chatbot Assistant to Retrieve Matching Businesses.")
    st.write("© Developed by Talin Labs®.")

    st.markdown("---")
    st.header("Instructions")
    st.markdown("""
                1. Type a Query to retrieve the matching Businesses.
                2. Upload a PDF RFP to retrieve Businesses.
                3. Use the Database Selection to retrieve Businesses.
                4. Use the AI Chatbot to interact with the Assistant.
                5. Use the Advanced Search to perform advanced queries.
                6. Use the Regions Tab to view the Businesses on the Map.
                7. Use the Business Economics Tab to view the Business Economics.
                8. Use the File Upload Tab to upload a PDF RFP.""")
    
    # Draw a line
    st.markdown("---")

    st.write("## Select the Database")
    db_select = st.radio("Select the Database", ["Use Public Database", "Proprietory Database"], key="db_select", label_visibility="collapsed")

    st.markdown("---")

    # st.button("New Session")
    # st.button("Clear Chat History")


    # pdf_query = st.file_uploader(label="Upload a RFP to Retrieve Businesses.", type=["pdf"], key="pdf_query")

# with bottom():
    # row2 = row.row([16, 4], vertical_align="center")
    # user_query = row2.chat_input("Type a message...")
    # sb = row2.button("Clear History", use_container_width=True)

# if sb:
#     st.session_state.chat_history = []
#     st.session_state.chat_display = []
#     st.session_state.chat_history.append(AIMessage(content="Hello! I'm a Match-Making assistant. Write a Query to retrieve the matching Businesses."))
#     st.session_state.chat_display.append(AIMessage(content="Hello! I'm a Match-Making assistant. Write a Query to retrieve the matching Businesses."))

query_global = "SELECT * from supplierdb LIMIT 2;"
xlsx_query = None
xlsx_data = None
tab1, tab2 = st.tabs(["Advanced AI Chatbot", "Basic Search"])
pdf_upl_prpt = True
db_upl_prpt = True

with tab1:
    col1, col2 = tab1.columns([0.7, 0.3])
    mapping = col2.container(height=615, border=False)
    tabbing = col2.container(height=390, border=False)
    
    prop_upload = tabbing.expander("Data-File Upload",expanded= True, icon=":material/file_upload:")
    file_upload = tabbing.expander("RFP-File Upload", expanded= True, icon=":material/file_upload:")

    with file_upload:
        flu = file_upload.container(height=115, border=False)
        with flu:
            pdf_query = flu.file_uploader(label="Upload a RFP to Retrieve Businesses.", type=["pdf"], label_visibility="collapsed")
    
    if db_select == "Proprietory Database":
        time.sleep(0.5)
        with prop_upload:
            flq = prop_upload.container(height=125, border=False)
            with flq:
                xlsx_query = flq.file_uploader(label="Upload a XLSX to load the Proprietory Supplier List.", type=["xlsx"], label_visibility="collapsed")
        if xlsx_query is not None:
            data_file = save_uploaded_file(xlsx_query)
            xlsx_data = pd.read_excel(data_file)
            conn = sqlite3.connect('supplier_prop.db')
            xlsx_data.to_sql('supplierdb', conn, if_exists='replace', index=False)
            st.session_state.db, st.session_state.data = load_db(False)
    elif db_select == "Use Public Database":
        st.session_state.db, st.session_state.data = load_db(True)

    with col1:
        upper = st.container(height=960)

        with upper:
            for message in st.session_state.chat_display:
                if isinstance(message, AIMessage):
                    with st.chat_message("AI"):
                        with st.spinner("Generating Response..."):
                            if message.content == "Hello! I'm a Match-Making assistant. Describe your requirements to retrieve the matching Businesses.":
                                generate_mk(message.content)
                            elif "The Suppliers List has been Loaded Successfully." in message.content:
                                st.success("The Suppliers List has been Loaded Successfully.")
                                st.dataframe(xlsx_data, hide_index=True, selection_mode="single")
                            else:
                                # mk, _, _ = get_response(message.content)
                                mk = message.content
                                if "No Matching Businesses Found." not in mk[0]:
                                    generate_mk("Here are the Matching Businesses:")
                                    generate_mk_ai(mk[0], len(mk[0]))
                                    download_file(mk[1], u_key=np.random.randint(1000, 9999))
                                elif "No Matching Businesses Found." in mk[0]:
                                    generate_mk("No Matching Businesses Found.")
                elif isinstance(message, HumanMessage):
                    with st.chat_message("Human"):
                        generate_mk(message.content)

        user_query = col1.chat_input("Type your Businesses Query here...", key="user_query")

        with upper:
            if user_query is not None and user_query.strip() != "":
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                st.session_state.chat_display.append(HumanMessage(content=user_query))
                
                with st.chat_message("Human"):
                    generate_mk(user_query) 

                with st.chat_message("AI"):
                    with st.spinner("Retrieving Businesses..."):
                        sql_query_response = get_sql_chain(user_query, st.session_state.db, st.session_state.chat_history)
                        sql_query_response = sql_query_response.replace('servicetype, services', 'services')
                        query_global = sql_query_response
                        # print(sql_query_response)
                        response, result, df = get_response(sql_query_response)
                        if len(df) > 0:
                            st.success("Here are the Matching Businesses:")
                            generate_mk_ai(response, len(df))
                            download_file(df, u_key=np.random.randint(1000, 9999))
                            st.session_state.chat_display.append(AIMessage(content=[response, df]))
                        else:
                            st.error("No Matching Businesses Found.")
                            # st.session_state.chat_display.append(AIMessage(content="No Matching Businesses Found."))
                
                st.session_state.chat_history.append(BaseMessage(content=sql_query_response, type="AI"))

            elif pdf_query is not None and user_query is None:

                st.session_state.chat_history.append(HumanMessage(content="Uploaded File **"+pdf_query.name+"**. Retrieving Businesses for the RFP File Requirements."))
                st.session_state.chat_display.append(HumanMessage(content="Uploaded File **"+pdf_query.name+"**. Retrieving Businesses for the RFP File Requirements."))
            
                
                with st.chat_message("Human"):
                    generate_mk("Uploaded File **"+pdf_query.name+"**. Retrieving Businesses for the RFP File Requirements.")
                    
                with st.chat_message("AI"):
                        try:
                            with st.status("Retrieving Businesses..."):
                                st.write("Analysing the RFP Document...")
                                nlp_2 = get_pdf_nlp_query(pdf_query)
                                st.write("Extracting the Services from the RFP Document...")
                                sql_query_response = get_sql_chain(nlp_2, st.session_state.db, st.session_state.chat_history)
                                query_global = sql_query_response
                                st.write("Retrieving the Matching Businesses...")
                            
                            response, result, df = get_response(sql_query_response)
                        except Exception as e:
                            st.error(f"Error: {e}")
                            try :
                                with st.status("Re-attempting to Retrieve Businesses..."):
                                    st.write("Analysing the RFP Document...")
                                    nlp_2 = get_pdf_nlp_query(pdf_query) 
                                    st.write("Extracting the Services from the RFP Document...")
                                    sql_query_response = get_sql_chain(nlp_2, st.session_state.db, st.session_state.chat_history)
                                    query_global = sql_query_response
                                    st.write("Retrieving the Matching Businesses...")
                                response, result, df = get_response(sql_query_response)
                            except Exception as e:
                                    response = "Error: Unable to Retrieve Businesses. Please try again later."
                    
                        if len(df) > 0:
                            st.success("Here are the Matching Businesses:")
                            generate_mk_ai(response, len(df))
                            download_file(df, u_key=np.random.randint(1000, 9999))
                            st.session_state.chat_display.append(AIMessage(content=[response, df]))
                        else:
                            st.error("No Matching Businesses Found.")

                        # a = time.time()
                        # nlp_1 = get_pdf_query(pdf_query)
                        # b = time.time()
                        # print(nlp_1, f"\n\n Time Taken: {b-a}")

                st.session_state.chat_history.append(BaseMessage(content=sql_query_response, type="AI"))
            
            elif xlsx_query is not None and user_query is None and pdf_query is None:
                st.session_state.chat_history.append(HumanMessage(content="Uploaded Data-File **"+xlsx_query.name+"**. Loading Suppliers from the XLSX File."))
                st.session_state.chat_display.append(HumanMessage(content="Uploaded Data-File **"+xlsx_query.name+"**. Loading Suppliers from the XLSX File."))
                
                with st.chat_message("Human"):
                    time.sleep(0.5)
                    generate_mk("Uploaded Data-File **"+xlsx_query.name+"**. Loading Suppliers from the XLSX File.")
                    
                with st.chat_message("AI"):
                    time.sleep(0.5)
                    with st.status("Loading Businesses...", expanded=True):
                        st.write("Analysing the XLSX Document Structure and Data Attributes...")
                        time.sleep(2)
                        st.write("Extracting the Suppliers List and their attributes from the XLSX Document...")
                        time.sleep(4)
                        st.write("Loading the Supplier Information...")
                        time.sleep(2)
                        
                    
                    # Show a AI Message that the File has been converted and loaded. Start enterring the Query to retrieve the Businesses.
                    st.success("The Suppliers List has been Loaded Successfully.")
                    st.session_state.chat_display.append(AIMessage(content="The Suppliers List has been Loaded Successfully."))
                    st.dataframe(xlsx_data, hide_index=True, selection_mode="single")
                    


    with col2:
        full_query = ''
        # repl = 'company, address, city, state, zip'
        # if 'servicetype' in full_query:
        #     repl += ', servicetype'
        # elif 'services' in full_query:
        #     repl += ', services'
        if db_select != "Proprietory Database":
            full_query = str(query_global).replace('LIMIT 10', '')
            full_query = full_query.replace('company, address, city, state, zip, service_type', 'address, coordinates')
        else:
            full_query = query_global
        # print(full_query)
        run_sync = 0
        m = create_map()
        with mapping:
            t1, t2 = st.tabs(["Regions", "Business Economics"])
            with t1:
                map_container = t1.container(height=480, border=False)
                if full_query is not None and """(
    SELECT *"""  in full_query:
                    print("Full Query: ", full_query)
                    try:                    
                        full_data = data.execute(full_query).fetchall()
                    except Exception as e:
                        full_data = []

                    if len(full_data) > 0:
                        
                        t1.info(f"##### Total \"{len(full_data)}\" Matching Businesses found.", icon=":material/find_in_page:")
                        if db_select != "Proprietory Database":
                            df = pd.DataFrame(eval(str(full_data)), columns=["address", "coordinates"], index=np.arange(1, len(full_data)+1))
                            
                            # Remove none values from dataframe
                            df = df.dropna()
                            print("Dataframe: ", df['coordinates'])

                            df['coordinates'] = df['coordinates'].apply(lambda x: ast.literal_eval(x))


                            df['latitude'] = df['coordinates'].apply(lambda x: x[0])
                            df['longitude'] = df['coordinates'].apply(lambda x: x[1])

                            # Convert latitude and longitude to float
                            df['latitude'] = df['latitude'].astype(float)
                            df['longitude'] = df['longitude'].astype(float)

                            # Define the bounds for the contiguous 48 states
                            lat_min, lat_max = 24.396308, 49.384358
                            lon_min, lon_max = -125.0, -66.93457

                            # Filter the DataFrame to keep only the rows within these bounds
                            map_df = df[(df['latitude'] >= lat_min) & (df['latitude'] <= lat_max) &
                                        (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max)]

                            # Drop the 'coordinates' column if no longer needed
                            map_df = map_df.drop(columns=['latitude', 'longitude'])
                        else:
                            with map_container:
                                folium_static(m, height=460, width=585)

                        
                        with map_container:
                            if db_select != "Proprietory Database" and len(map_df) > 0:
                                create_map_whole(m, map_df)
                            else:
                                folium_static(m, height=460, width=585)

                    else:
                        print("No Matching Businesses Found.")
                        with map_container:
                            folium_static(m, height=460, width=585)
                        t1.error("##### No Matching Businesses Found.")
                else:
                    print("No Matching Businesses Found.")
                    with map_container:
                        folium_static(m, height=460, width=585)
                    t1.info('##### Write a Query to Find Matching Businesses.', icon=":material/find_in_page:")
            with t2:
                st.write("### Business Economics")

        
        with tabbing:
            
            if st.session_state.pdf_query is not None:
                st.success("File Uploaded Successfully.")
            # # dbs = tabbing.expander("Database Selection", expanded=True, icon=":material/database:")
            # with dbs:

with tab2:
    col1, col2 = st.columns([0.7, 0.3])
    dt = pd.read_csv('search_filter_data.csv', low_memory=False)


    resp = 0
    text = "##### **Search Filters** "
    with col1:
        dfc = col1.container(height=960, border=False)
        msc = col1.empty()
    with col2:
        f1 = st.form(key="search_form", border=False)
        with f1:
            f1.info(text, icon=":material/tune:")
            with f1.container(height=900, border=False):
                flds = st.expander("Keyword Search", icon=":material/text_fields:", expanded=True)
                with flds:
                    keyw = st.text_input("Enter Keywords or Phrases...", placeholder="Enter Keywords or Phrases...", key="user_query_basic", label_visibility="collapsed")
                    # fields = st.multiselect("Select the Fields to Search", ["company", "address", "city", "state", "zip", "services"], key="fields")
                with st.expander("Search Ranking Filters", icon=":material/filter_list:"):
                    rows = st.columns([1,1,1,1])
                    rev_up = rows[0].number_input("Annual-Revenue", value=0, placeholder="Min...")
                    emp_up = rows[2].number_input("Employee-Count", value=0, placeholder="Min...")
                    rev_down = rows[1].number_input("Annual Revenue", value=None, placeholder="Max ($100M)...", label_visibility="hidden")
                    emp_down = rows[3].number_input("Employees", value=None, placeholder="Max (200)...", label_visibility="hidden")
                    c_1, c_2 = st.columns([0.5, 0.5])
                    with c_1:
                        est_up = st.number_input("Establishment Year", value=1999, placeholder="Min...")
                    with c_2:
                        est_down = st.number_input("Established", value=None, placeholder="Max (2024)...", label_visibility="hidden")
                with st.expander("Business Demographics", icon=":material/business:", expanded=True):
                    ncss = st.container(height=100, border=False)
                    naics = ncss.multiselect("Select NAICS Code(s)", [code for code in dt['NAICS_Description'].unique() if not pd.isnull(code)], key="naics")
                with st.expander("Geographic Information", icon=":material/globe:", expanded=True):    
                    c_one, c_two, c_three = st.columns([0.33, 0.33, 0.33])
                    city = c_one.multiselect("Select City(s)", [code for code in sorted(dt['city'].unique()) if not pd.isnull(code)], key="city")
                    states = c_two.multiselect("Select State(s)", [code for code in sorted(dt['state'].unique()) if not pd.isnull(code)], key="states")
                    zipc = c_three.multiselect("Select Zip Code(s)", [code for code in sorted(dt['zip'].apply(lambda x: str(x)[:5]).unique()) if not pd.isnull(code)], key="zipc") 
                with st.expander("Demographics", icon=":material/demography:", expanded=True):
                    cone, ctwo = st.columns([0.5, 0.5])
                    ethnicity = cone.multiselect("Ethnicity Type(s)", ['ASIAN', 'General', 'BLACK', 'NON-MINORITY', 'HISPANIC', 'NATIVE AMERICAN'], key="ethnicity")
                    ownership = ctwo.multiselect("Ownership Type(s)", ['Minority-Owned', 'General-Ownership', 'Minority-owned', 'Women-Owned', 'Veteran-owned', 'Woman-Owned', 'Veteran-owned, Minority-owned', 'Women-owned, Minority-owned', 'Family-Owned', 'Women-owned', 'Service-Disabled-Veteran-Owned', 'Minority-owned, Woman-Owned', 'Minority-Owned, Women-Owned'], key="ownership")
                with st.expander("Certifications and Compliance", icon=":material/verified_user:", expanded=True):
                    # ISO Standard, CMMI, ITAR Registration, Compliance
                    one, two = st.columns([0.5, 0.5])
                    iso = one.multiselect("ISO Standard(s)", ['ISO 9000', 'ISO 27001, ISO 9000', 'Non-ISO Standard', 'ISO 27001'], key="ISO Standard")
                    compliance = two.multiselect("Compliance Type(s)", ['NIST 800-171 compliance', 'PIPEDA compliance', 'No Compliance Conditions', 'NIST 800-171 compliance, PIPEDA compliance'], key="Compliance")
                    cmmi = two.multiselect("CMMI Integration(s)", ['Not-Integrated', 'Integrated'], key="CMMI")
                    itar = one.multiselect("ITAR Registration(s)", ['Registered','Not Registered'], key="ITAR Registration")

            one, two = f1.columns([0.5, 0.5])
            search = one.form_submit_button("Search", use_container_width=True)      
            reset = two.form_submit_button("Reset", use_container_width=True)      
            
    # Generate a SQL Query using the User Input Fields
    # print(search, fields, rev_up, rev_down, emp_up, emp_down, est_up, est_down, naics, city, states, zipc)

    if search and keyw:

        words = re.split(r'[,\s\-]+', keyw.strip())
    
        # Generate a list of words with '%' appended at front and end
        words = [f"%{word}%" for word in words if word]

        QUERY = "SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE ("
    
        # Generate separate LIKE conditions for each word
        if words:
            conditions = " OR ".join([f"(UPPER(services) LIKE UPPER('{word}'))" for word in words])
            QUERY += conditions

        QUERY += ")"

        # Additional conditions
        conditions = []

        # # Annual Revenue
        # if rev_up:
        #     conditions.append(f"annual_revenue >= {rev_up}")
        # if rev_down:
        #     conditions.append(f"annual_revenue <= {rev_down}")

        # # Employee Count
        # if emp_up:
        #     conditions.append(f"employee_count >= {emp_up}")
        # if emp_down:
        #     conditions.append(f"employee_count <= {emp_down}")

        # # Establishment Year
        # if est_up:
        #     conditions.append(f"establishment_year >= {est_up}")
        # if est_down:
        #     conditions.append(f"establishment_year <= {est_down}")

        # NAICS Code
        if naics:
            naics_condition = " OR ".join([f"naics = '{code[:6]}'" for code in naics])
            conditions.append(f"({naics_condition})")

        # City
        if city:
            city_condition = " OR ".join([f"city = '{c}'" for c in city])
            conditions.append(f"({city_condition})")

        # State
        if states:
            state_condition = " OR ".join([f"state = '{s}'" for s in states])
            conditions.append(f"({state_condition})")

        # Zip Code
        if zipc:
            zip_condition = " OR ".join([f"zip LIKE '{z}%'" for z in zipc])
            conditions.append(f"({zip_condition})")

        # Ethnicity
        if ethnicity:
            ethnicity_condition = " OR ".join([f"ethnicity = '{eth}'" for eth in ethnicity])
            conditions.append(f"({ethnicity_condition})")

        # Ownership
        if ownership:
            ownership_condition = " OR ".join([f"ownership = '{own}'" for own in ownership])
            conditions.append(f"({ownership_condition})")

        # ISO Standards
        if iso:
            iso_condition = " OR ".join([f"\"ISO Standard\" LIKE '%{std}%'" for std in iso])
            conditions.append(f"({iso_condition})")

        # Compliance Types
        if compliance:
            compliance_condition = " OR ".join([f"\"Compliance\" LIKE '%{comp}%'" for comp in compliance])
            conditions.append(f"({compliance_condition})")

        # CMMI Integration
        if cmmi:
            cmmi_condition = " OR ".join([f"\"CMMI\" = '{integration}'" for integration in cmmi])
            conditions.append(f"({cmmi_condition})")

        # ITAR Registration
        if itar:
            itar_condition = " OR ".join([f"\"ITAR Registration\" = '{reg}'" for reg in itar])
            conditions.append(f"({itar_condition})")

        # Append all conditions
        if conditions:
            QUERY += " AND " + " AND ".join(conditions)

        print(QUERY)

        resp = data.execute(QUERY).fetchall()
        if resp:
            df = pd.DataFrame(resp, columns=["Company Name", "Address", "City", "State", "Zip", "Services Offered"], index=np.arange(1, len(resp)+1))
            dfc.dataframe(df, height=950, width=1400)
            text = f"##### \"{len(df)}\" Results Found."
            with bottom():
                msc.info(text, icon=":material/find_in_page:")
        else:
            msc.error("No Matching Businesses Found.")
            time.sleep(5)
            msc.warning("Modify your Search Filters to find suitable Matching Businesses.", icon=":material/dashboard:")
            time.sleep(10)
            msc.info("Use the Search Filters to find the Matching Businesses.", icon=":material/dashboard:")
    elif reset:
        st.rerun()
    elif search and not keyw:
        with bottom():
            msc.error("Please Enter a Keyword to Search.", icon=":material/error_outline:")
            time.sleep(5)
            msc.info("Use the Search Filters to find the Matching Businesses.", icon=":material/dashboard:")
    else:
        with dfc:
            co = st.columns([1, 1, 1])
            co[1].image("manage_search.png", use_column_width=True, output_format="PNG")
        msc.info("Use the Search Filters to find the Matching Businesses.", icon=":material/dashboard:")
    # print(fields, naics)

    





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
#
#
#
#     # (5) : LLM Prompting and Response Generation Template
#     ---------------------------------------------------------
#       template = """
        # <SYS> 
        # You are a Master SQL Generator at a company, you are very well versed with Filtering, Ranking, Sorting and Retrieving Data from a sqlite3 Database. You are interacting with a user who is asking for companies matching the services he is interested from company's database. Based on the table schema below and Conversation History, write a SQL query (sqlite3) that would answer the user's question in the most perfect manner possible. Retain the conversation history context to generate the SQL query. If the question is new and not in the conversation history, then generate the SQL query based on the table schema.
        # </SYS>
        
        # <SCHEMA>{schema}</SCHEMA>
        
        # <HISTORY>{chat_history}</HISTORY>
        
        # <INST> 
        # - Write only the SQL query (sqlite3) and nothing else. 
        # - Do not wrap the SQL query in any other text, not even backticks. 
        # - Limit the records to 10.
        # - Formulate the SQL query (sqlite3) to match only the first FOUR DIGITS of the NAICS code. 
        # - DO NOT use AND operator in the SQL query.
        # - Use all words of the services requested, and the NAICS code to match the services provided by the companies. - Retrieve only the following columns: company, address, city, state, zip, servicetype.
        # - Use nested SQL queries to filter the data based on the conditions of search. 
        # - Use the ORDER BY clause to sort the data based on the column relevant to the search.
        # </INST>
        
        # For example:
        # Question: List the companies that provide creative or production services.
        # SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE (UPPER(services) LIKE UPPER('%/creative%production%') OR UPPER(servicetype) LIKE UPPER('%production%') OR UPPER(servicetype) LIKE UPPER('%/creative%')) OR naics LIKE '5414%' OR naics LIKE '7225%' ORDER BY company;

        # Question: Filter by California State.
        # SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE state = 'CA' AND (UPPER(services) LIKE UPPER('%/creative%production%') OR UPPER(servicetype) LIKE UPPER('%production%') OR UPPER(servicetype) LIKE UPPER('%/creative%')) OR naics LIKE '5414%' OR naics LIKE '7225%' ORDER BY state;

        # Question: Limit by California State.
        # SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE state = 'CA' AND (UPPER(services) LIKE UPPER('%/creative%production%') OR UPPER(servicetype) LIKE UPPER('%production%') OR UPPER(servicetype) LIKE UPPER('%/creative%')) OR naics LIKE '5414%' OR naics LIKE '7225%' ORDER BY state;

        # Question: Filter by California State.
        # SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE state = 'CA' AND (UPPER(services) LIKE UPPER('%/creative%production%') OR UPPER(servicetype) LIKE UPPER('%production%') OR UPPER(servicetype) LIKE UPPER('%/creative%')) OR naics LIKE '5414%' OR naics LIKE '7225%' ORDER BY state;

        # Question: Limit by Los Angeles City.
        # SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE city = 'Los Angeles' AND (UPPER(services) LIKE UPPER('%/creative%production%') OR UPPER(servicetype) LIKE UPPER('%production%') OR UPPER(servicetype) LIKE UPPER('%/creative%')) OR naics LIKE '5414%' OR naics LIKE '7225%' ORDER BY city;
        
        # Question: Filter by ITAR Registered.
        # SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE 'ITAR Registration' = 'Registered' AND (UPPER(services) LIKE UPPER('%/creative%production%') OR UPPER(servicetype) LIKE UPPER('%production%') OR UPPER(servicetype) LIKE UPPER('%/creative%')) OR naics LIKE '5414%' OR naics LIKE '7225%' ORDER BY company;

        # Question: Limit by ITAR Registered.
        # SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE 'ITAR Registration' = 'Registered' AND (UPPER(services) LIKE UPPER('%/creative%production%') OR UPPER(servicetype) LIKE UPPER('%production%') OR UPPER(servicetype) LIKE UPPER('%/creative%')) OR naics LIKE '5414%' OR naics LIKE '7225%' ORDER BY company;

        # Question: List companies providing IT services.
        # SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE (UPPER(services) LIKE UPPER('%IT%')) OR naics LIKE '5415%' LIMIT 10 ORDER BY company;

        # Question: List the next 10 companies providing IT services.
        # SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE (UPPER(services) LIKE UPPER('%IT%')) OR naics LIKE '5415%' LIMIT 10 OFFSET 10 ORDER BY company;

        # Question: List companies with ISO certification.
        # SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE "ISO Standard" IS NOT NULL;

        # Question: List companies in California.
        # SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE state = 'CA' ORDER BY state LIMIT 10;

        # Question: Give me 10 more companies in California.
        # SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE state = 'CA' ORDER BY state LIMIT 10, 10;

        # Question: Give me 10 more companies in California.
        # SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE state = 'CA' ORDER BY state LIMIT 20, 10;
        
        # Your turn:
        
        # Question: {question}. Give me Company Names, Address [Address, City, State, Zip] and services Offered.
        # SQL Query:
        # """
#    ---------------------------------------------------------
# """
