import os
import json
import numpy as np
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# ==============================================
# ✅ API CONFIG — Load securely from Streamlit secrets
# ==============================================
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error("⚠️ Could not load OpenAI API key. Make sure it's set in Streamlit Secrets.")
    st.stop()

# ==============================================
# 🧱 PAGE CONFIG
# ==============================================
st.set_page_config(page_title="🧠 Arabic Study Assistant", layout="wide")

# ==============================================
# 📘 LOAD DATA
# ==============================================
try:
    with open("data/arabic_book_text.txt", "r", encoding="utf-8") as f:
        book_text = f.read()

    with open("data/embeddings.json", "r", encoding="utf-8") as f:
        embeddings_data = json.load(f)
except FileNotFoundError:
    st.error("❌ Data files not found. Please ensure 'data/arabic_book_text.txt' and 'data/embeddings.json' exist.")
    st.stop()

# ==============================================
# 📚 STRUCTURED CHAPTERS
# ==============================================
chapters = {
    "📖 القراءة العربية": [
        "إرادة التغيير",
        "أبو الريحان البيروني",
        "القدس مدينة عربية إسلامية",
        "العلم في الإسلام",
        "قيم إنسانية"
    ],
    "✍️ الأدب والنصوص": [
        "مدرسة الإحياء والبعث وجيل التطوير",
        "أحمد شوقي وجيل التطوير",
        "المدارس الرومانسية في الشعر العربي",
        "الاتجاه الرومانسي",
        "المرأة - خليل مطران",
        "رفاء القاتل",
        "أهوارك يا وطني - محمود حسن إسماعيل",
        "من أنت يا نفسي - ميخائيل نعيمة"
    ],
    "📗 النثر وفنونه": [
        "المقال",
        "الرواية",
        "القصة القصيرة",
        "الكنيسة نورت - إبراهيم أصلان",
        "المسرحية"
    ],
    "💬 البلاغة: مفاهيم نقدية": [
        "التجربة الشعرية",
        "الرحلة الفنية"
    ],
    "🧩 التدريبات اللغوية": [
        "النطق والإملاء",
        "الأبنية",
        "الترادف",
        "إعراب الاسم",
        "إعراب الفعل",
        "الأدوات",
        "الممنوع من الصرف"
    ]
}

# ==============================================
# 🧠 AI FUNCTIONS
# ==============================================
def ask_question(prompt, chapter_text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an Arabic language study assistant. Answer in Arabic clearly and helpfully."},
                {"role": "user", "content": f"Text: {chapter_text}\n\nQuestion: {prompt}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ==============================================
# 💬 CHAT INTERFACE
# ==============================================
st.sidebar.title("📘 Choose Chapter")

main_section = st.sidebar.selectbox("Main Section:", list(chapters.keys()))
unit = st.sidebar.selectbox("Unit:", chapters[main_section])

st.title("🧠 Arabic Study Assistant")
st.markdown("Chat with your Arabic book — ask questions, get summaries or generate quizzes.")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Get the selected chapter content
chapter_text = f"Extracted text for: {unit}\n\n{book_text[:2000]}"

# Chat UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask something about this unit..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            response = ask_question(prompt, chapter_text)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

