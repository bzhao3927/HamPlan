#!/usr/bin/env python3
"""
RAG System - Complete Version
Catalog + Syllabi + Department Overviews + Prerequisites
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
                    "type": "department_overview"
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
def load_course_catalog(json_path, requirements_path=None, model=None):
    """Load course catalog with prerequisites resolved."""
    print(f"\n📚 Loading course catalog: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        courses = json.load(f)
    
    # Load requirements mapping if provided
    req_mapping = {}
    if requirements_path and Path(requirements_path).exists():
        print(f"📋 Loading requirements mapping: {requirements_path}")
        with open(requirements_path, 'r', encoding='utf-8') as f:
            requirements = json.load(f)
            for req in requirements:
                req_code = req.get('RequirementCode', '')
                req_courses = req.get('Courses', [])
                if req_code:
                    req_mapping[req_code] = req_courses
    
    print(f"✅ Loaded {len(courses)} courses")
    if req_mapping:
        print(f"✅ Loaded {len(req_mapping)} prerequisite mappings")
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
        
        # Resolve prerequisites
        requisites = course_obj.get('Requisites', [])
        prereq_text = ""
        if requisites and req_mapping:
            prereq_courses = []
            for req in requisites:
                req_code = req.get('RequirementCode', '')
                if req_code in req_mapping:
                    prereq_courses.extend(req_mapping[req_code])
            if prereq_courses:
                prereq_text = f"Prerequisites: {', '.join(set(prereq_courses))}\n"
        
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

def answer_question(query, documents):
    """Answer question using GPT-4."""
    relevant_docs = search_documents(query, documents)
    
    context = "\n\n---\n\n".join([
        f"From {doc['source']}:\n{doc['text']}" 
        for doc in relevant_docs
    ])
    
    prompt = f"""Based on the following course information, answer the question concisely.

Question: {query}

Course Information:
{context}

Provide a clear, direct answer based only on the information above."""

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful academic advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    
    answer = response.choices[0].message.content
    sources = [doc["source"] for doc in relevant_docs]
    
    return answer, sources

# =============================================
# MAIN
# =============================================
def main():
    print("="*70)
    print("🎓 COMPLETE COURSE RAG SYSTEM")
    print("="*70)
    
    model = None
    
    # Paths
    catalog_json = "../course-catalog-scraper/courses_fall_2025.json"
    requirements_json = "../course-catalog-scraper/requirements_data.json"
    department_overviews_txt = "../course-catalog-scraper/department_overviews/txt"
    syllabi_folder = "/Users/CS/Documents/GitHub/Final-Project/RAG/syllabi"
    cache_file = "cache/complete_system_v2.pkl"  # New cache name
    
    # Check cache
    if Path(cache_file).exists():
        print(f"\n✅ Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            documents = pickle.load(f)
        print(f"✅ Loaded {len(documents)} documents")
    else:
        all_documents = []
        
        # Load catalog
        catalog_docs = load_course_catalog(catalog_json, requirements_json, model)
        all_documents.extend(catalog_docs)
        
        # Load department overviews
        if Path(department_overviews_txt).exists():
            dept_docs = load_department_overviews(department_overviews_txt, model)  # CHANGED
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
    
    # Q&A loop
    print("\n" + "="*70)
    print("💡 Ask questions!")
    print("   - What are the requirements for a CS major?")
    print("   - What classes does Professor Kuruwita teach?")
    print("   - What statistics courses are available?")
    print("\nType 'quit' to exit")
    print("="*70 + "\n")
    
    while True:
        q = input("❓ Question: ").strip()
        
        if q.lower() in {"quit", "exit", "q"}:
            print("\n👋 Goodbye!")
            break
        
        if not q:
            continue
        
        print("\n🔍 Searching...")
        try:
            answer, sources = answer_question(q, documents)
            
            print(f"\n💡 Answer:")
            print("-" * 70)
            print(answer)
            print("-" * 70)
            print(f"\n📚 Top Sources:")
            for i, source in enumerate(sources[:5], 1):
                print(f"  {i}. {source}")
            print("\n" + "=" * 70 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    main()