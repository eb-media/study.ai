import os
import json
import numpy as np
import pypdfium2
from openai import OpenAI
from dotenv import load_dotenv

# ==========================
# 🔑 API & ENV SETUP
# ==========================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pdf_path = "data/Arabic_language_Sec3.pdf"
text_path = "data/arabic_book_text.txt"
embeddings_path = "data/embeddings.json"
answers_path = "data/answers.txt"

# ==========================
# 📚 STRUCTURED CHAPTERS
# ==========================
chapters = {
    "القراءة العربية": [
        "الموضوع الأول : إرادة التغيير",
        "الموضوع الثاني : أثر الإيمان باليوم الآخر",
        "الموضوع الثالث : القدس مدينة عربية إسلامية",
        "الموضوع الرابع : العلم في الإسلام",
        "الموضوع الخامس : قيم إنسانية"
    ],

    # الشعر + النثر تحت نفس المظلة "الأدب والنصوص"
    "الأدب والنصوص": [
        # الشعر ومدارسه
        "مدرسة الإحياء والبعث وجيل التطوير",
        "المدارس الرومانتيكية في الشعر العربي",
        "الواقعية والشعر الحديث",

        # النثر وفنونه
        "المقال",
        "مثال : التكافل الاجتماعي في الإسلام – أحمد حسن الزيات",
        "الرواية",
        "القصة القصيرة",
        "قصة : الكنيسة نورت – إبراهيم أصلان",
        "المسرحية"
    ],

    "البلاغة": [
        "التجربة الشعرية",
        "الوحدة الفنية"
    ],

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

# ==========================
# 📄 LOAD PDF TEXT
# ==========================
def load_pdf_text():
    if os.path.exists(text_path):
        print("📖 Book text already exists, loading...")
        with open(text_path, "r", encoding="utf-8") as f:
            return f.read()

    print("📘 Extracting text from PDF...")
    pdf = pypdfium2.PdfDocument(pdf_path)
    text = ""
    for i, page in enumerate(pdf):
        page_text = page.get_textpage().get_text_range()
        text += f"\n--- PAGE {i+1} ---\n" + page_text
        print(f"✅ Page {i+1}/{len(pdf)} processed")

    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"💾 Text saved to {text_path}")
    return text

# ==========================
# 🧩 SPLIT INTO CHAPTERS
# ==========================
def split_into_chapters(text):
    print("📚 Splitting book by chapters...")
    chapters_dict = {}
    current = None
    buffer = []

    # Flatten all subchapter titles for search
    all_titles = [title for sublist in chapters.values() for title in sublist]

    def normalize(s):
        return s.replace(" ", "").replace(":", "").replace("ـ", "").replace("\n", "")

    normalized_text = text.replace("\r", "")
    lines = normalized_text.splitlines()

    for line in lines:
        clean_line = normalize(line)
        for title in all_titles:
            if normalize(title[:10]) in clean_line:  # fuzzy matching
                if current and buffer:
                    chapters_dict[current] = "\n".join(buffer).strip()
                    buffer = []
                current = title
                print(f"🟢 Found section: {title}")
                break
        else:
            if current:
                buffer.append(line)

    if current and buffer:
        chapters_dict[current] = "\n".join(buffer).strip()

    print(f"✅ Found {len(chapters_dict)} total sections.")
    return chapters_dict

# ==========================
# 🧠 CREATE EMBEDDINGS
# ==========================
def create_embeddings(chapters_dict):
    if os.path.exists(embeddings_path):
        print("💾 Embeddings already exist, loading...")
        with open(embeddings_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("🧩 Creating embeddings...")
    embeddings = {}

    def chunk_text(text, max_chars=6000):
        for i in range(0, len(text), max_chars):
            yield text[i:i + max_chars]

    for title, content in chapters_dict.items():
        print(f"🔹 Processing: {title}")
        chunks = list(chunk_text(content))
        chunk_embeddings = []

        for i, chunk in enumerate(chunks):
            emb = client.embeddings.create(
                input=chunk,
                model="text-embedding-3-small"
            ).data[0].embedding
            chunk_embeddings.append(emb)
            print(f"  ✅ Chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")

        mean_emb = np.mean(chunk_embeddings, axis=0).tolist()
        embeddings[title] = {"embedding": mean_emb, "text": content}

    with open(embeddings_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)
    print("💾 Embeddings saved successfully.")
    return embeddings

# ==========================
# 🔍 COSINE SIMILARITY
# ==========================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ==========================
# 💬 ASK QUESTION
# ==========================
def ask_question(question, chapter_text):
    prompt = f"""
You are a smart Arabic study assistant.
Answer the following question in Arabic using only the provided text.

Text:
{chapter_text}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    print("\n🗣️ Answer:\n", answer)

    with open(answers_path, "a", encoding="utf-8") as f:
        f.write(f"❓ Question: {question}\n💬 Answer: {answer}\n{'='*70}\n")

# ==========================
# 🧾 SUMMARY (ENG + AR)
# ==========================
def summarize_chapter(chapter_text):
    prompt = f"""
Summarize the following Arabic educational text in both English and Arabic.
Make it concise and educational:

{chapter_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    summary = response.choices[0].message.content
    print("\n📚 Summary:\n", summary)
    return summary

# ==========================
# 🧩 QUIZ GENERATION (Arabic)
# ==========================
def generate_quiz(chapter_text):
    prompt = f"""
Based on the following Arabic educational text, create 5 multiple-choice questions in Arabic.
Each question should have:
- 1 correct answer
- 3 wrong answers
Mark the correct answer with (✅) and the others with (❌).

Text:
{chapter_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    quiz = response.choices[0].message.content
    print("\n🧠 Quiz (Arabic):\n", quiz)

    with open(answers_path, "a", encoding="utf-8") as f:
        f.write(f"🧩 Quiz Generated:\n{quiz}\n{'='*70}\n")

# ==========================
# 🚀 MAIN MENU
# ==========================
def main():
    print("\n🚀 START STUDY ASSISTANT\n")

    text = load_pdf_text()
    chapters_dict = split_into_chapters(text)
    embeddings = create_embeddings(chapters_dict)

    print("\n✅ Knowledge base ready!\n")

    while True:
        print("\n📖 Main Sections:")
        main_titles = list(chapters.keys())
        for i, section in enumerate(main_titles, 1):
            print(f"{i}. {section}")

        choice = input("\nSelect a main section number (or type 'exit'): ").strip()
        if choice.lower() == "exit":
            print("👋 Goodbye!")
            break

        try:
            selected_main = main_titles[int(choice) - 1]
        except (IndexError, ValueError):
            print("⚠️ Invalid choice, try again.")
            continue

        print(f"\n📘 Selected main section: {selected_main}")
        sub_units = chapters[selected_main]

        while True:
            print("\n📚 Sub-units:")
            for j, sub in enumerate(sub_units, 1):
                print(f"{j}. {sub}")
            print("0. 🔙 Back to main sections")

            sub_choice = input("\nSelect a sub-unit: ").strip()
            if sub_choice == "0":
                break
            try:
                selected_sub = sub_units[int(sub_choice) - 1]
            except (IndexError, ValueError):
                print("⚠️ Invalid option.")
                continue

            chapter_text = chapters_dict.get(selected_sub, "")
            if not chapter_text:
                print("⚠️ Text for this section not found.")
                continue

            while True:
                print("\nOptions:")
                print("1️⃣ Ask a question")
                print("2️⃣ Get summary")
                print("3️⃣ Generate quiz (Arabic)")
                print("4️⃣ Back to sub-units")
                print("5️⃣ Exit")

                opt = input("\nChoose an option: ").strip()
                if opt == "1":
                    q = input("\n❓ Enter your question: ")
                    ask_question(q, chapter_text)
                elif opt == "2":
                    summarize_chapter(chapter_text)
                elif opt == "3":
                    generate_quiz(chapter_text)
                elif opt == "4":
                    break
                elif opt == "5":
                    print("👋 Goodbye!")
                    return
                else:
                    print("⚠️ Invalid option, try again.")

if __name__ == "__main__":
    main()
