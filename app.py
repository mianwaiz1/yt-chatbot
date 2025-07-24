import streamlit as st
import os
import pandas as pd
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load API Key
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")
os.environ["GOOGLE_API_KEY"] = google_api_key

# --- YouTube Scraper ---
def scrape_youtube(query):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--log-level=3")
    driver = webdriver.Chrome(options=options)

    search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
    driver.get(search_url)

    for _ in range(5):
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
        time.sleep(2)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    video_elements = soup.find_all('ytd-video-renderer')

    data = []
    for video in video_elements:
        try:
            a_tag = video.find('a', id='video-title') # type: ignore
            if not a_tag:
                continue
            title = a_tag.get('title') or a_tag.text.strip() # type: ignore
            url = "https://www.youtube.com" + a_tag['href'] # type: ignore

            channel_tag = video.find('ytd-channel-name') # type: ignore
            channel = channel_tag.text.strip() if channel_tag else "" # type: ignore

            meta = video.find('div', id='metadata-line').find_all('span') # type: ignore
            views = meta[0].text.strip() if len(meta) > 0 else ""
            upload_date = meta[1].text.strip() if len(meta) > 1 else ""

            img_tag = video.find('img') # type: ignore
            thumbnail = (
                img_tag.get('src') # type: ignore
                or img_tag.get('data-thumb') # type: ignore
                or img_tag.get('data-src') # type: ignore
                or ""
            )

            data.append({
                'Title': title.strip(), # type: ignore
                'Video URL': url,
                'Channel': channel,
                'Views': views,
                'Upload Date': upload_date,
                'Thumbnail URL': thumbnail,
            })
        except:
            continue

    driver.quit()
    return pd.DataFrame(data)

# --- LangChain Setup ---
@st.cache_resource(show_spinner="🔄 Embedding video data...")
def create_vectorstore(df):
    docs = []
    for _, row in df.iterrows():
        metadata = {
            "title": row["Title"],
            "url": row.get("Video URL", ""),
            "channel": row.get("Channel", ""),
            "upload_date": row.get("Upload Date", ""),
            "views": row.get("Views", ""),
        }
        content = (
            f"Video Title: {row['Title']}\n"
            f"Channel: {metadata['channel']}\n"
            f"Upload Date: {metadata['upload_date']}\n"
            f"Views: {metadata['views']}"
        )
        docs.append(Document(page_content=content, metadata=metadata))

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

# --- Streamlit UI ---
st.set_page_config(page_title="YouTube Scraper + Chatbot", layout="wide")
st.title("🎬 YouTube Video Scraper + 🤖 Videos Chatbot ")
st.markdown("---")

# --- Scraping Section ---
st.subheader("🔍 Scrape YouTube")
search_query = st.text_input("Enter search term:", key="search_query")
search_btn = st.button("🔎 Search YouTube", key="search_button")

# Session state to store scraped data and chatbot
if "video_df" not in st.session_state:
    st.session_state.video_df = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if search_btn and search_query.strip():
    with st.spinner("Scraping YouTube..."):
        df = scrape_youtube(search_query)
        if not df.empty:
            st.session_state.video_df = df
            st.session_state.vectorstore = create_vectorstore(df)
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3),
                retriever=st.session_state.vectorstore.as_retriever(search_type="similarity", k=10),
                return_source_documents=True
            )
            st.success(f"✅ Found {len(df)} videos.")
            st.dataframe(df[['Title', 'Video URL', 'Channel', 'Views', 'Upload Date', 'Thumbnail URL']])
            # Prepare downloadable CSV
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name= "youtube_videos_data.csv",
                mime="text/csv"
            )
        else:
            st.error("❌ No videos found or scraping failed.")

# --- Chatbot Section ---
st.markdown("---")
st.subheader("💬 Chat with Gemini")

if st.session_state.qa_chain:
    user_question = st.text_input("Ask a question about the videos:", key="chat_input")
    ask_btn = st.button("🤖 Ask Gemini", key="ask_button")

    if ask_btn and user_question.strip():
        with st.spinner("Gemini is thinking..."):
            result = st.session_state.qa_chain({"query": user_question})
            st.markdown("### 🧠 Answer:")
            st.write(result["result"])

            st.markdown("### 📺 Source Videos")
            for doc in result["source_documents"]:
                meta = doc.metadata
                st.markdown(
                    f"**{meta['title']}**  \n[Watch here]({meta['url']})  \n📅 {meta.get('upload_date')} | 👁️ {meta.get('views')}"
                )
else:
    st.info("ℹ️ Scrape videos first to enable chat.")
