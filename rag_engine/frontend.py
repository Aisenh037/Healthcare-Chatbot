import streamlit as st
import requests
import json

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Medical Assistant RAG",
    page_icon="M",
    layout="wide"
)

# Title and description
st.title("Medical Research Assistant")
st.markdown("""
**System Architecture:** Streamlit (UI) → FastAPI (Gateway) → RAG Engine → LLM
""")

# Check authentication
if "auth" not in st.session_state:
    st.session_state.auth = None

def login():
    with st.sidebar:
        st.header("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Log In"):
            try:
                # Use Basic Auth to check credentials
                res = requests.get(f"{API_URL}/auth/login", auth=(username, password))
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.auth = {
                        "username": username,
                        "password": password,
                        "role": data["role"]
                    }
                    st.success(f"Welcome {username} ({data['role']})")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            except Exception as e:
                st.error(f"Connection failed: {e}")

def logout():
    st.session_state.auth = None
    st.rerun()

# --- MAIN APP LOGIC ---

if not st.session_state.auth:
    login()
    st.info("Please log in to access the Medical Assistant.")
    st.markdown("Try: `admin`/`admin123` or `doctor`/`doc123`")
else:
    user = st.session_state.auth
    
    # Sidebar: User Info & Actions
    with st.sidebar:
        st.success(f"👤 Logged in as: **{user['username'].upper()}**")
        st.caption(f"Role: {user['role']}")
        if st.button("Log Out"):
            logout()
        
        st.markdown("---")
        
        # --- ROLE-BASED FEATURE: UPLOAD (Admin Only) ---
        if user['role'] == 'admin':
            st.header("📂 Ingestion (Admin)")
            uploaded_file = st.file_uploader("Upload Medical PDF", type=["pdf"])
            
            if uploaded_file is not None:
                if st.button("Index Document"):
                    with st.spinner("Ingesting..."):
                        try:
                            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                            # Pass credentials
                            res = requests.post(
                                f"{API_URL}/upload", 
                                files=files,
                                auth=(user['username'], user['password'])
                            )
                            if res.status_code == 200:
                                st.success(f"Indexed! Total: {res.json()['total_chunks']}")
                            else:
                                st.error(f"Error: {res.text}")
                        except Exception as e:
                            st.error(f"Error: {e}")
        else:
            st.info("👋 Uploads are restricted to Admins.")

    # Main Chat Area
    st.header(f"🩺 Medical Assistant ({user['role'].title()} View)")
    
    # Init messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a medical question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Consulting database..."):
                try:
                    # Send query to chat endpoint (Form data)
                    res = requests.post(
                        f"{API_URL}/chat/chat",
                        data={"message": prompt},
                        auth=(user['username'], user['password'])
                    )
                    
                    if res.status_code == 200:
                        data = res.json()
                        answer = data["answer"]
                        sources = data.get("sources", [])
                        
                        full_resp = answer
                        if sources:
                            full_resp += "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in sources])
                        
                        placeholder.markdown(full_resp)
                        st.session_state.messages.append({"role": "assistant", "content": full_resp})
                    else:
                        err = f"Error {res.status_code}: {res.text}"
                        placeholder.error(err)
                except Exception as e:
                    placeholder.error(f"Connection error: {e}")
