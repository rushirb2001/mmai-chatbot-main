import streamlit as st

# Create the new page object
new_page = st.Page("test_page.py", title="New Chat", icon=":material/chat:")

# Append the new page to the list
st.session_state.pages["All Chat Messages"].append(st.Page("test_page.py", title="New Chat", icon=":material/chat:"))