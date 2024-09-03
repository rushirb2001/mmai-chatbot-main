import streamlit as st

if "pages" not in st.session_state:
    st.session_state.pages = {}

st.session_state.pages = {
    "All Chat Messages": [
        st.Page("add_page.py", title="Create Chat", icon=":material/add_box:"),
        st.Page(title="New Chat", icon=":material/chat:"),
    ]
}

bt = st.sidebar.button("Add Page")

if bt:
    st.session_state.pages["All Chat Messages"].append(st.Page("test_page.py", title="New Chat", icon=":material/chat:"))

pg = st.navigation(st.session_state.pages)
pg.run()