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
import pandas as pd
import numpy as np
import gdown
import zipfile, time
import sqlite3
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
st.set_page_config(page_title="Match-Maker-AI", page_icon=":speech_balloon:", layout="wide")
st.title("Match-Maker-AI")
LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]

# Download the database file, decorated with Streamlit's caching
@st.cache_data()
def download_db():
    """
    Function: Download the database file from Google Drive

    Returns:
        None

    Other:
    # if not os.path.exists("supplier-database.db"):
        # gdown.download('https://drive.google.com/uc?id=167gji0LKnOJElgIA0flocOI8s_ZFgxGs', 'supplier-database.db', quiet=False)
    """
    if not os.path.exists("supplier_database-v3.db"):
        gdown.download('https://drive.google.com/uc?id=1IbzFyzO55siAUs6sUQ0JPJaFnsUUZhcU', 'supplier_database-v3.db', quiet=False)
download_db()

# Load the database and create a connection
@st.cache_resource()
def load_db():
    """
    Function: Load the database and create a connection

    Returns:
        db: SQLDatabase object
        data: sqlite3 connection object

    Other:
    # db_uri = f"sqlite:///supplier-database.db"
    # data = sqlite3.connect("supplier-database.db")
    """
    db_uri = f"sqlite:///supplier_database-v3.db"
    database = SQLDatabase.from_uri(db_uri)
    conn = sqlite3.connect("supplier_database-v3.db", check_same_thread=False)
    return database, conn
db, data = load_db()
st.session_state.db = db
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
            SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE UPPER(services) LIKE UPPER('%/creative%production%') OR UPPER(servicetype) LIKE UPPER('%production%') OR UPPER(servicetype) LIKE UPPER('%/creative%') AND naics LIKE '5414%' OR naics LIKE '7225%';

            Question: Filter by California State.
            SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE state = 'CA' AND UPPER(services) LIKE UPPER('%/creative%production%') OR UPPER(servicetype) LIKE UPPER('%production%') OR UPPER(servicetype) LIKE UPPER('%/creative%') AND naics LIKE '5414%' OR naics LIKE '7225%';

            Question: Filter by California State.
            SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE state = 'CA' AND UPPER(services) LIKE UPPER('%/creative%production%') OR UPPER(servicetype) LIKE UPPER('%production%') OR UPPER(servicetype) LIKE UPPER('%/creative%') AND naics LIKE '5414%' OR naics LIKE '7225%';

            Question: List companies providing IT services.
            SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE UPPER(services) LIKE UPPER('%IT%') AND naics LIKE '5415%' LIMIT 10;

            Question: List the next 10 companies providing IT services.
            SQL Query: SELECT company, address, city, state, zip, servicetype FROM supplierdb WHERE UPPER(services) LIKE UPPER('%IT%') AND naics LIKE '5415%' LIMIT 10 OFFSET 10;

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

# Function to get the response from the SQL query

def get_response(sql_query_response: str):
    """
    Function: Get the response from the SQL query

    Args:
        sql_query_response: SQL query response

    Returns:
        result: SQL query result
    """

    if sql_query_response:
        try:
            result = data.execute(sql_query_response).fetchall()
            result = str(result).replace("\\n\\n", "")
            result = str(result).replace("\\n", "")

            df_result = data.execute(str(sql_query_response).replace('LIMIT 10', '')).fetchall()
            df_result = str(df_result).replace("\\n\\n", "")
            df_result = str(df_result).replace("\\n", "")

            if len(eval(result)) > 0 or len(result) > 0:
                df = pd.DataFrame(eval(df_result), columns=["Company Name", "Address", "City", "State", "Zip", "Services Offered"], index=np.arange(1, len(eval(df_result))+1))
                if not len(df) == 0:
                    return format_businesses_to_markdown(result), result, df
                else:
                    return "No Matching Businesses Found.", result, pd.DataFrame()
            else:
                return "No Matching Businesses Found."
        except Exception as e:
            print(e)
            return "Error: Unable to Retrieve Businesses. Please try again later.1"
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
    Following the extraction of the specified services and NAICS codes [Atmost 2 to 3 Codes] extracted from the RFP, create a Natural Language query to find companies offering these services.
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
    
    llm = ChatGroq(model="llama3-70b-8192", temperature=0, api_key=st.secrets["GROQ_API_KEY"])
    # llm = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_API_KEY"])
    
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

#     faiss_index = FAISS.from_documents(pdf_f, OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536, api_key=st.secrets["OPENAI_API_KEY"]))

#     question = "Analyze the attached RFP document and extract the Procurement Purpose of Service/Commodity/Information or Commodity Solicitation/Work Category. Provide the list in bullet points, ensuring each point briefly describes a distinct commodity/service required by the RFP without including any descriptive or additional content."

#     template = """
#     You are an Entity Extractor AI at a match-making company to extract the required type of Procured Service or Commodity or Solicitation of Information/ Commodity or Work Category. You are working with a RFP pdf that a User has uploaded and you are tasked 
#     to contextually extract the sought services mentioned in the Request for Proposal (RFP) / Request for Information (RFI) document context provided under statement of procurement purpose.
#     Following the extraction of the specified services and NAICS codes [Atmost 2 to 3 Codes] extracted from the RFP, create a Natural Language query to find companies offering these services.
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
    
#     llm = ChatGroq(model="llama3-70b-8192", temperature=0, api_key=st.secrets["GROQ_API_KEY"])
#     # llm = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_API_KEY"])
    
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

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=st.secrets["GROQ_API_KEY"])
    # llm = ChatGroq(model="llama3-70b-8192", temperature=0, api_key=st.secrets["GROQ_API_KEY"])
    # llm = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_API_KEY"])
    
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
    map_obj = folium.Map(tiles=None)
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

@st.experimental_fragment()
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

@st.experimental_fragment()
def generate_mk_ai(response, check):
    if check >= 5:
        infor = st.container(height=685)
        with infor:
            infor.markdown(response)
    else:
        st.markdown(response)

def generate_mk(content):
    st.markdown(content)

@st.experimental_fragment()
def generate_df(df):
    st.dataframe(df, width=2000)

@st.experimental_fragment()
def generate_data(content):
    df = pd.DataFrame(content, columns=["Company Name", "Address", "City", "State", "Zip", "Services Offered"], index=np.arange(1, len(content)+1))
    st.dataframe(df, width=2000)

@st.experimental_fragment()
def create_map_whole(m, df):

    with tqdm(total=3, desc="Generating the Map...") as pbar:
        df['coordinates'] = df['coordinates'].apply(eval)
        pbar.update(1)
        if len(df) < 1000:
            m = add_markers(m, df['coordinates'], popup_list=None)
        else:
            m = map_resize(m, df['coordinates'])
            labels, centroids = cluster_latlng(df['coordinates'].to_list(), 10)
            df['cluster_label'] = labels
            df['cluster_center'] = df['cluster_label'].apply(lambda x: [round(centroids[x][0], 6), round(centroids[x][1], 6)])
            df = df.sort_values('cluster_label')
            df['distance_cluster_center_miles'] = df.apply(lambda x: haversine_distance(x['coordinates'][0], 
                                                                            x['coordinates'][1], 
                                                                            x['cluster_center'][0], 
                                                                            x['cluster_center'][1]), axis=1)
            
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
        folium_static(m, height=470, width=540)
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
    HORIZONTAL = "/Users/rushirbhavsar/Pictures/talin_labs_logo-horizontal.jpg"
    ICON = "/Users/rushirbhavsar/Pictures/talin_labs_logo.jpg"

    st.logo(HORIZONTAL, icon_image=ICON)
    
    st.write("## Welcome to Match-Maker-AI")
    st.write("### A Chatbot Assistant to Retrieve Matching Businesses.")
    st.write(" Developed by Talin Labs.")

    st.subheader("Instructions")
    st.markdown("""
                1. Type a Query to retrieve the matching Businesses.
                2. Upload a PDF RFP to retrieve Businesses.
                3. Use the Database Selection to retrieve Businesses.
                4. Use the AI Chatbot to interact with the Assistant.
                5. Use the Advanced Search to perform advanced queries.
                6. Use the Regions Tab to view the Businesses on the Map.
                7. Use the Business Economics Tab to view the Business Economics.
                8. Use the File Upload Tab to upload a PDF RFP.""")



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
# pdf_query = st.session_state.pdf_query
tab1, tab2 = st.tabs(["AI Chatbot", "Advanced Search"])

with tab1:
    col1, col2 = tab1.columns([0.7, 0.3])
    mapping = col2.container(height=675, border=False)
    tabbing = col2.container(height=325, border=False)
    tb1, tb2 = tabbing.tabs(["File Upload", "Database Selection"])
    pdf_query = tb1.file_uploader(label="Upload a RFP to Retrieve Businesses.", type=["pdf"])
    with col1:
        upper = st.container(height=960)

        with upper:
            for message in st.session_state.chat_display:
                if isinstance(message, AIMessage):
                    with st.chat_message("AI"):
                        with st.spinner("Generating Response..."):
                            if message.content == "Hello! I'm a Match-Making assistant. Describe your requirements to retrieve the matching Businesses.":
                                generate_mk(message.content)
                            else:
                                # mk, _, _ = get_response(message.content)
                                mk = message.content
                                if "No Matching Businesses Found." not in mk[0]:
                                    st.success("Here are the Matching Businesses:")
                                    generate_mk_ai(mk[0], len(mk[0]))
                                    download_file(mk[1], u_key=len(mk[0])+len(st.session_state.chat_display))
                                elif "No Matching Businesses Found." in mk[0]:
                                    st.error("No Matching Businesses Found.")
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
                            download_file(df, u_key = len(response)+len(st.session_state.chat_display))
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
                            download_file(df, u_key = len(response)+len(st.session_state.chat_display))
                            st.session_state.chat_display.append(AIMessage(content=[response, df]))
                        else:
                            st.error("No Matching Businesses Found.")

                        # a = time.time()
                        # nlp_1 = get_pdf_query(pdf_query)
                        # b = time.time()
                        # print(nlp_1, f"\n\n Time Taken: {b-a}")

                st.session_state.chat_history.append(BaseMessage(content=sql_query_response, type="AI"))
    with col2:
        full_query = str(query_global).replace('LIMIT 10', '')
        full_query = full_query.replace('company, address, city, state, zip, services', 'address, coordinates')
        run_sync = 0
        m = create_map()
        with mapping:
            t1, t2 = st.tabs(["Regions", "Business Economics"])
            with t1:
                map_container = t1.container(height=520)
                if full_query is not None and "*" not in full_query:
                    full_data = data.execute(full_query).fetchall()
                    if len(full_data) > 0:
                        t1.write(f"> ### :grey-background[Total \"{len(full_data)}\" Matching Businesses found.]")
                        df = pd.DataFrame(eval(str(full_data)), columns=["address", "coordinates"], index=np.arange(1, len(full_data)+1))
                        with map_container:
                            create_map_whole(m, df)
                    else:
                        with map_container:
                            folium_static(m, height=470, width=560)
                        t1.write("#### No Matching Businesses Found.")
                else:
                    with map_container:
                        folium_static(m, height=470, width=560)
                    t1.markdown('> ### :grey-background[Write a Query to Find Matching Businesses.]')
            with t2:
                st.write("### Business Economics")

        
        with tabbing:
            
            
            if st.session_state.pdf_query is not None:
                st.success("File Uploaded Successfully.")
            on = tb2.toggle("Use Proprietory Databasee")
            # debug_container = st.container(height=350)
            # with debug_container:
            #     st.write(st.session_state.chat_history)
            # tb2.write(st.session_state.chat_history)
with tab2:
    st.write("### Advanced Search")
    import streamlit as st

    st.bar_chart({"data": [1, 5, 2, 6, 2, 1]})

    with st.expander("See explanation"):
        st.write('''
            The chart above shows some numbers I picked for you.
            I rolled actual dice for these, so they're *guaranteed* to
            be random.
        ''')
        st.image("https://static.streamlit.io/examples/dice.jpg")





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
