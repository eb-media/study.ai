import streamlit as st
import json, os
from openai import OpenAI
from dotenv import load_dotenv

# ==========================
# 🔑 SETUP
# ==========================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Arabic Study Assistant", layout="centered")

# ==========================
# 📚 Load book data
# ==========================
with open("data/arabic_book_text.txt", "r", encoding="utf-8") as f:
    book_text = f.read()
with open("data/embeddings.json", "r", encoding="utf-8") as f:
    embeddings = json.load(f)

chapters = {
    "📖 القراءة العربية": [
        "إرادة التغيير", "أبو الريحان البيروني", "القدس مدينة عربية إسلامية",
        "العلم في الإسلام", "قيم إنسانية"
    ],
    "✍️ الأدب والنصوص": [
        "مدرسة الإحياء والبعث", "أحمد شوقي", "المدارس الرومانتيكية", "المقال",
        "القصة القصيرة", "التمثيلية"
    ],
    "🎨 البلاغة": ["التجربة الشعرية", "الوحدة الفنية"],
    "🧠 التدريبات اللغوية": [
        "النطق والإملاء", "الأبنية", "التراكيب", "إعراب الاسم",
        "إعراب الفعل", "الأدوات", "الممنوع من الصرف"
    ],
}

# ==========================
# 💬 Chat Interface
# ==========================
st.title("🧠 Arabic Study Assistant")
st.caption("Chat with your Arabic book — ask questions, get summaries or generate quizzes.")

st.sidebar.header("📚 Choose Chapter")
chapter = st.sidebar.selectbox("Main Section:", list(chapters.keys()))
unit = st.sidebar.selectbox("Unit:", chapters[chapter])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask something about this unit...")

if user_input:
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Display "thinking..." before the AI responds
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("🤔 **Thinking...**")

        try:
            # Generate response using OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"You are a helpful Arabic study assistant. Context: chapter '{chapter}', unit '{unit}'."},
                    *st.session_state.messages
                ]
            )
            answer = response.choices[0].message.content

            # Replace "thinking..." with final answer
            thinking_placeholder.markdown(answer)

            # Save AI message to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            thinking_placeholder.markdown(f"⚠️ Error: {e}")

