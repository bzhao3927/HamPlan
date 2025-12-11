#!/usr/bin/env python3
"""
RAG System - Complete Version with Conversation Memory
Catalog + Syllabi + Department Overviews + Prerequisites + Memory
"""

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os
import pickle
from pathlib import Path
import PyPDF2
import json

# Load environment
load_dotenv(find_dotenv())
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =============================================
# EMBEDDING FUNCTION
# =============================================
def get_openai_embedding(text):
    """Get OpenAI embedding."""
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text[:8000]
    )
    return np.array(response.data[0].embedding)

# =============================================
# LOAD DEPARTMENT OVERVIEWS
# =============================================
def load_department_overviews(txt_folder, model):
    """Load department overview text files with major/minor requirements."""
    print(f"\n📚 Loading department overviews: {txt_folder}")

    txt_path = Path(txt_folder)
    if not txt_path.exists():
        print(f"⚠️  Department overviews folder not found")
        return []

    txt_files = list(txt_path.glob("*.txt"))
    print(f"✅ Found {len(txt_files)} department files")
    print("🔄 Creating embeddings...")

    dept_docs = []

    for i, txt_file in enumerate(txt_files, 1):
        dept_name = txt_file.stem  # filename without .txt

        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()

            # Skip the header lines (Department: ... URL: ... ===)
            lines = text.split('\n')
            content_start = 0
            for idx, line in enumerate(lines):
                if '=' * 50 in line:  # Find the separator
                    content_start = idx + 1
                    break

            clean_text = '\n'.join(lines[content_start:]).strip()

            if len(clean_text) > 200:  # Only if substantial content
                embedding = get_openai_embedding(clean_text)

                dept_docs.append({
                    "text": clean_text,
                    "embedding": embedding,
                    "source": f"Department Overview: {dept_name.replace('-', ' ').title()}",
                    "type": "department_overview",
                    "department": dept_name
                })

            if i % 10 == 0:
                print(f"  Processed {i}/{len(txt_files)}...")

        except Exception as e:
            print(f"  ⚠️  Error reading {txt_file.name}: {e}")
            continue

    print(f"✅ Created {len(dept_docs)} department overview embeddings\n")
    return dept_docs

# =============================================
# LOAD COURSE CATALOG
# =============================================
def load_course_catalog(json_path, model=None):
    """Load course catalog with prerequisites from DisplayText."""
    print(f"\n📚 Loading course catalog: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        courses = json.load(f)

    print(f"✅ Loaded {len(courses)} courses")
    print("🔄 Creating embeddings...")

    catalog_docs = []

    for i, course in enumerate(courses):
        course_obj = course.get('Course', {})

        course_name = course.get('CourseName', '')
        title = course_obj.get('Title', '')
        description = course_obj.get('Description', '')
        credits = course.get('MinimumCredits', '')
        subject = course_obj.get('SubjectCode', '')

        # Instructor info
        faculty = course.get('FacultyDisplay', [])
        instructor_text = ""
        if faculty:
            instructor_text = f"Instructor: {', '.join(faculty)}\n"

        # Meeting times and location
        meetings = course.get('MeetingsDisplay', [])
        meeting_text = ""
        if meetings:
            meeting_text = f"Meeting times: {'; '.join(meetings)}\n"

        # Course types (major requirements, distribution)
        course_types = course.get('CourseTypesDisplay', [])
        requirements_text = ""
        if course_types:
            requirements_text = f"Satisfies: {', '.join(course_types)}\n"

        # Extract prerequisites from DisplayText
        prerequisites = course.get('Prerequisites', [])
        prereq_text = ""
        if prerequisites:
            prereq_list = [prereq.get('DisplayText', '') for prereq in prerequisites if prereq.get('DisplayText')]
            if prereq_list:
                prereq_text = f"Prerequisites: {'; '.join(prereq_list)}\n"

        # Build searchable text
        text = f"{course_name}: {title}\n"
        if instructor_text:
            text += instructor_text
        if meeting_text:
            text += meeting_text
        if description:
            text += f"Description: {description}\n"
        if prereq_text:
            text += prereq_text
        if requirements_text:
            text += requirements_text
        if credits:
            text += f"Credits: {credits}"

        if text.strip():
            embedding = get_openai_embedding(text)

            catalog_docs.append({
                "text": text,
                "embedding": embedding,
                "source": f"Catalog: {course_name}",
                "subject": subject,
                "type": "catalog",
                "instructor": faculty[0] if faculty else None
            })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(courses)}...")

    print(f"✅ Created {len(catalog_docs)} catalog embeddings\n")
    return catalog_docs

# =============================================
# LOAD SYLLABI
# =============================================
def load_syllabi(syllabi_folder, model):
    """Load and embed syllabi PDFs."""
    print(f"📚 Loading syllabi from: {syllabi_folder}")

    syllabi_path = Path(syllabi_folder)
    if not syllabi_path.exists():
        print(f"⚠️  Folder not found")
        return []

    pdf_files = []
    for subject_folder in syllabi_path.iterdir():
        if subject_folder.is_dir():
            for pdf_file in subject_folder.glob("*.pdf"):
                pdf_files.append(pdf_file)

    print(f"✅ Found {len(pdf_files)} PDFs")
    print("🔄 Creating embeddings...")

    syllabus_docs = []

    for idx, pdf_path in enumerate(pdf_files, 1):
        subject = pdf_path.parent.name
        pdf_name = pdf_path.name

        text = ""
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"  ⚠️  Error reading {pdf_name}: {e}")
            continue

        if not text.strip():
            continue

        chunks = [c.strip() for c in text.split('\n\n') if len(c.strip()) >= 50]

        print(f"  [{idx}/{len(pdf_files)}] {subject}/{pdf_name} - {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            try:
                embedding = get_openai_embedding(chunk)

                syllabus_docs.append({
                    "text": chunk,
                    "embedding": embedding,
                    "source": f"{subject}/{pdf_name} (chunk {i+1})",
                    "subject": subject,
                    "type": "syllabus"
                })
            except Exception as e:
                print(f"    ⚠️  Error on chunk {i+1}: {e}")
                continue

    print(f"✅ Created {len(syllabus_docs)} syllabus embeddings\n")
    return syllabus_docs

# =============================================
# MEMORY MANAGEMENT
# =============================================
def trim_conversation_history(history, max_tokens=4000):
    """Keep only recent conversation within token limit."""
    # Rough estimate: 1 token ≈ 4 characters
    total_chars = sum(len(msg["content"]) for msg in history)

    while total_chars > max_tokens * 4 and len(history) > 2:
        # Remove oldest exchange (user + assistant)
        history.pop(0)
        history.pop(0)
        total_chars = sum(len(msg["content"]) for msg in history)

    return history

def summarize_old_conversation(history, client):
    """Summarize older parts of conversation."""
    if len(history) < 10:  # Don't summarize if short
        return history

    # Take first half of conversation
    old_messages = history[:len(history)//2]
    recent_messages = history[len(history)//2:]

    # Summarize old messages
    old_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in old_messages])

    summary_prompt = f"""Summarize this conversation concisely, preserving key questions and answers:

{old_text}

Provide a brief summary of what was discussed."""

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.3,
        max_tokens=200
    )

    summary = response.choices[0].message.content

    # Return summary + recent messages
    return [{"role": "system", "content": f"Previous conversation summary: {summary}"}] + recent_messages

# =============================================
# SEARCH & ANSWER
# =============================================
def search_documents(query, documents, top_k=50):
    """Search documents using OpenAI embeddings."""
    query_emb = get_openai_embedding(query)

    scores = []
    for doc in documents:
        similarity = np.dot(query_emb, doc["embedding"])
        scores.append((similarity, doc))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scores[:top_k]]

def answer_question_with_memory(query, documents, conversation_history, use_summary=False):
    """Answer question using GPT-4 with conversation memory."""
    relevant_docs = search_documents(query, documents)

    context = "\n\n---\n\n".join([
        f"From {doc['source']}:\n{doc['text']}"
        for doc in relevant_docs
    ])

    # Manage conversation history
    if use_summary and len(conversation_history) > 10:
        conversation_history = summarize_old_conversation(conversation_history, client)
    else:
        conversation_history = trim_conversation_history(conversation_history)

    # Build messages with history
    messages = [
        {"role": "system", "content": """You are a helpful academic advisor at Hamilton College. 

Guidelines:
- Answer questions based ONLY on the course information provided
- If Hamilton doesn't offer courses in a specific area, say so clearly and suggest related alternatives
- Always cite specific course codes when discussing prerequisites
- Be precise about requirements (e.g., "CPSCI-101 OR placement" not just "CPSCI-101")
- If information is uncertain or incomplete, acknowledge it
- Maintain context from previous questions in the conversation"""}
    ]

    # Add conversation history
    messages.extend(conversation_history)

    # Add current query with context
    prompt = f"""Based on the following course information, answer the question concisely.

Question: {query}

Course Information:
{context}

Guidelines:
- If the question asks about a subject Hamilton doesn't offer, clearly state this and suggest related alternatives
- When discussing prerequisites, always cite the specific course code and be precise about OR conditions
- Be accurate about course requirements
- If information is incomplete, acknowledge limitations

Provide a clear, direct answer based only on the information above. If referencing previous questions, be explicit about what you're referring to."""

    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        temperature=0.3,
        max_tokens=500
    )

    answer = response.choices[0].message.content
    sources = [doc["source"] for doc in relevant_docs]

    # Update conversation history
    conversation_history.append({"role": "user", "content": query})
    conversation_history.append({"role": "assistant", "content": answer})

    return answer, sources, conversation_history

# =============================================
# MAIN
# =============================================
def main():
    print("="*70)
    print("🎓 COMPLETE COURSE RAG SYSTEM WITH MEMORY")
    print("="*70)

    model = None

    # Paths
    catalog_json = "../course-catalog-scraper/courses_with_prerequisites.json"
    department_overviews_txt = "/Users/CS/Documents/GitHub/Final-Project/course-catalog-scraper/department_overviews/txt"
    syllabi_folder = "/Users/CS/Documents/GitHub/Final-Project/RAG/syllabi"
    cache_file = "cache/complete_system_v3.pkl"

    # Check cache
    if Path(cache_file).exists():
        print(f"\n✅ Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            documents = pickle.load(f)
        print(f"✅ Loaded {len(documents)} documents")
    else:
        all_documents = []

        # Load catalog
        catalog_docs = load_course_catalog(catalog_json, model)
        all_documents.extend(catalog_docs)

        # Load department overviews
        if Path(department_overviews_txt).exists():
            dept_docs = load_department_overviews(department_overviews_txt, model)
            all_documents.extend(dept_docs)

        # Load syllabi
        if Path(syllabi_folder).exists():
            syllabus_docs = load_syllabi(syllabi_folder, model)
            all_documents.extend(syllabus_docs)

        print(f"\n🎉 Total documents: {len(all_documents)}")
        print(f"   Catalog: {len([d for d in all_documents if d.get('type') == 'catalog'])}")
        print(f"   Department overviews: {len([d for d in all_documents if d.get('type') == 'department_overview'])}")
        print(f"   Syllabi: {len([d for d in all_documents if d.get('type') == 'syllabus'])}")

        Path("cache").mkdir(exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(all_documents, f)
        print(f"💾 Saved to cache")

        documents = all_documents

    # Initialize conversation memory
    conversation_history = []

    # Q&A loop
    print("\n" + "="*70)
    print("💡 Ask questions! (I'll remember our conversation)")
    print("   - What are the requirements for a CS major?")
    print("   - What about their prerequisites? (follow-up)")
    print("   - What classes does Professor Kuruwita teach?")
    print("\nCommands:")
    print("   'clear' - Reset conversation memory")
    print("   'quit' or 'exit' - Exit the system")
    print("="*70 + "\n")

    while True:
        q = input("❓ Question: ").strip()

        if q.lower() in {"quit", "exit", "q"}:
            print("\n👋 Goodbye!")
            break

        if q.lower() == "clear":
            conversation_history = []
            print("🔄 Conversation memory cleared!\n")
            continue

        if not q:
            continue

        print("\n🔍 Searching...")
        try:
            answer, sources, conversation_history = answer_question_with_memory(
                q, documents, conversation_history, use_summary=True
            )

            print(f"\n💡 Answer:")
            print("-" * 70)
            print(answer)
            print("-" * 70)
            print(f"\n📚 Top Sources:")
            for i, source in enumerate(sources[:5], 1):
                print(f"  {i}. {source}")
            print(f"\n💬 Conversation exchanges: {len(conversation_history)//2}")
            print("\n" + "=" * 70 + "\n")

        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    main()