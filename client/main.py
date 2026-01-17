import streamlit as st
import requests
from requests.auth import HTTPBasicAuth
import os
from dotenv import load_dotenv

load_dotenv()

SERVER_API_URL = os.getenv("SERVER_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Healthcare Chatbot",
    page_icon="🏥",
    layout="wide"  # Improved layout
)

## Session State Init
if "username" not in st.session_state:
    st.session_state.username = ""
    st.session_state.password = ""
    st.session_state.role = ""
    st.session_state.logged_in = False
    st.session_state.messages = []  # Chat history

def get_auth():
    return HTTPBasicAuth(st.session_state.username, st.session_state.password)

def auth_ui():
    st.title("🏥 Medical Assistant System")
    st.subheader("Professional Access Portal")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🔐 Secure Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Access Dashboard", type="primary"):
            try:
                res = requests.get(f"{SERVER_API_URL}/auth/login", auth=HTTPBasicAuth(username, password))
                if res.status_code == 200:
                    user_data = res.json()
                    st.session_state.username = username
                    st.session_state.password = password
                    st.session_state.role = user_data["role"]
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid Credentials. Please check your username/password.")
            except Exception as e:
                st.error(f"Connection Error: {e}")

    with col2:
        st.info("ℹ️ Default Credentials (Seeded):")
        st.code("""
Admin:   admin / admin123
Doctor:  doctor / doc123
Nurse:   nurse / nurse123
Patient: patient / patient123
        """, language="text")

def upload_interface():
    st.sidebar.markdown("---")
    st.sidebar.header("📂 Document Ingestion")
    st.sidebar.info("Admin Privileges Active")
    
    uploaded_file = st.sidebar.file_uploader("Upload Medical PDF", type=["pdf"])
    role_for_doc = st.sidebar.selectbox("Access Level:", ["doctor", "nurse", "patient", "public"])
    
    if st.sidebar.button("Process & Index Document"):
        if uploaded_file:
            with st.spinner("Encrypting and Indexing..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                data = {"role": role_for_doc}
                try:
                    res = requests.post(f"{SERVER_API_URL}/upload_docs", files=files, data=data, auth=get_auth())
                    if res.status_code == 200:
                        doc_info = res.json()
                        st.sidebar.success(f"✅ Indexed {uploaded_file.name}")
                        st.sidebar.json(doc_info)
                    else:
                        st.sidebar.error(f"Upload Failed: {res.text}")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")

def chat_interface():
    st.header(f"🩺 Clinical Assistant ({st.session_state.role.title()} View)")
    
    # Check System Status
    try:
        status = requests.get(f"{SERVER_API_URL}/health").json()
        st.sidebar.success(f"Backend Status: {status['status'].upper()} 🟢")
        st.sidebar.caption(f"Engine: {status.get('engine', 'Unknown')}")
    except:
        st.sidebar.error("Backend Status: OFFLINE 🔴")

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Enter clinical query or patient question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("analyzing medical records..."):
                try:
                    # Match backend endpoint structure
                    res = requests.post(
                        f"{SERVER_API_URL}/chat/chat", 
                        data={"message": prompt}, 
                        auth=get_auth()
                    )
                    
                    if res.status_code == 200:
                        reply = res.json()
                        answer = reply["answer"]
                        sources = reply.get("sources", [])
                        
                        full_resp = answer
                        if sources:
                            full_resp += "\n\n**📚 Verification Sources:**\n" + "\n".join([f"- {s}" for s in sources])
                        
                        placeholder.markdown(full_resp)
                        st.session_state.messages.append({"role": "assistant", "content": full_resp})
                    else:
                        placeholder.error(f"Error {res.status_code}: {res.text}")
                except Exception as e:
                    placeholder.error(f"Network Error: {e}")

if not st.session_state.logged_in:
    auth_ui()
else:
    # Sidebar Profile
    with st.sidebar:
        st.success(f"👤 {st.session_state.username.upper()}")
        if st.button("Log Out"):
            st.session_state.logged_in = False
            st.rerun()
    
    # Admin Features
    if st.session_state.role == "admin":
        upload_interface()
    
    # Main Chat
    chat_interface()

