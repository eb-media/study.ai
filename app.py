import streamlit as st
import os, json, numpy as np, pypdfium2
from openai import OpenAI
from dotenv import load_dotenv

# ========================
# ⚙️ SETUP
# ========================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Arabic Study Assistant", layout="wide")

# ========================
# 📘 Load your data
# ========================
with open("data/arabic_book_text.txt", "r", encoding="utf-8") as f:
    book_text = f.read()

with open("data/embeddings.json", "r", encoding="utf-8") as f:
    embeddings = json.load(f)

# ========================
# 📚 Define Chapters
# ========================
chapters = {
    "القراءة العربية": [
        "الموضوع الأول : إرادة التغيير",
        "الموضوع الثاني : أثر الإيمان باليوم الآخر",
        "الموضوع الثالث : القدس مدينة عربية إسلامية",
        "الموضوع الرابع : العلم في الإسلام",
        "الموضوع الخامس : قيم إنسانية"
    ],
    "الأدب والنصوص": [
        "مدرسة الإحياء والبعث وجيل التطوير",
        "المدارس الرومانتيكية في الشعر العربي",
        "الواقعية والشعر الحديث",
        "المقال",
        "مثال : التكافل الاجتماعي في الإسلام – أحمد حسن الزيات",
        "الرواية",
        "القصة القصيرة",
        "قصة : الكنيسة نورت – إبراهيم أصلان",
        "المسرحية"
    ],
    "البلاغة": ["التجربة الشعرية", "الوحدة الفنية"],
    "التدريبات اللغوية": [
        "الوحدة الأولى : النطق والإملاء",
        "الوحدة الثانية : الأبنية",
        "الوحدة الثالثة : التراكيب",
        "الوحدة الرابعة : إعراب الاسم",
        "الوحدة الخامسة : إعراب الفعل",
        "الوحدة السادسة : الأدوات",
        "الوحدة السابعة : الممنوع من الصرف"
    ]
}

# ========================
# 🔍 Helper functions
# ========================
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_best_section(question):
    query_emb = client.embeddings.create(
        input=question, model="text-embedding-3-small"
    ).data[0].embedding
    best_title = max(embeddings.items(), key=lambda x: cosine_similarity(query_emb, x[1]["embedding"]))[0]
    return best_title, embeddings[best_title]["text"]

# ========================
# 🧠 OpenAI actions
# ========================
def ask_question(question, context):
    prompt = f"""You are a smart Arabic study assistant.
Answer the following question in Arabic using only the provided context:

Context:
{context}

Question:
{question}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def summarize(text):
    prompt = f"Summarize the following Arabic educational text in Arabic and English:\n\n{text}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def generate_quiz(text):
    prompt = f"""
Based on this Arabic educational text, create 5 multiple-choice questions in Arabic.
Each question must have 1 ✅ correct answer and 3 ❌ wrong answers.
{text}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ========================
# 🖥️ STREAMLIT UI
# ========================
st.title("📚 Arabic Study Assistant")

section = st.selectbox("Select Main Section:", list(chapters.keys()))
subunit = st.selectbox("Select Sub-Unit:", chapters[section])

st.divider()
action = st.radio("Choose an action:", ["❓ Ask a Question", "🧾 Get Summary", "🧠 Generate Quiz (Arabic)"])

if st.button("Run"):
    with st.spinner("Processing..."):
        chapter_text = embeddings.get(subunit, {}).get("text", "")
        if not chapter_text:
            st.error("No text found for this section.")
        else:
            if action == "❓ Ask a Question":
                question = st.text_input("Enter your question in Arabic:")
                if question:
                    st.write("🧠 **Answer:**")
                    st.markdown(ask_question(question, chapter_text))
            elif action == "🧾 Get Summary":
                st.write(summarize(chapter_text))
            elif action == "🧠 Generate Quiz (Arabic)":
                st.write(generate_quiz(chapter_text))
