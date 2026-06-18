import streamlit as st
from rag import process_urls, generate_answer

st.title("Real Estate Research Tool")

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

placeholder = st.empty()

process_url_button = st.sidebar.button("Process URLs")
if process_url_button:
    urls = [url for url in (url1, url2, url3) if url!='']
    if len(urls) == 0:
        placeholder.text("You must provide at least one valid url")
    else:
        for status in process_urls(urls):
            placeholder.text(status)

with placeholder.form("question_form"):
    query = st.text_input("Question")
    submitted = st.form_submit_button("Ask")
    
if submitted and query:
    try:
        answer, sources = generate_answer(query)
        st.session_state["last_answer"] = answer
        st.session_state["last_sources"] = sources
    except RuntimeError:
        st.session_state["last_answer"] = None
        placeholder.text("You must process urls first")

if st.session_state.get("last_answer"):
    st.header("Answer:")
    st.write(st.session_state["last_answer"])
    sources = st.session_state.get("last_sources")
    if sources:
        st.subheader("Sources:")
        for source in sources.split("\n"):
            st.write(source)
