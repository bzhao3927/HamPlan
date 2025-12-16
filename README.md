# HamPlan - Hamilton College Course Recommendation System

**Authors:** Cade Boiney, Ken Lam, Ognian Trajanov, Benjamin Zhao  
**Institution:** Hamilton College, Clinton, NY, USA

A RAG (Retrieval-Augmented Generation) system for Hamilton College courses using course catalog data, syllabi, and department overviews to answer student questions with conversation memory.

**Try it live:** [hamiltoncourses.com](http://hamiltoncourses.com)

## Prerequisites
- Python 3.8+
- OpenAI API key
- Course catalog JSON file (`courses_with_prerequisites.json`)
- Course syllabi PDFs (optional but recommended)
- Department overview text files

## Setup

**1. Navigate to the RAG directory:**
```bash
cd RAG
```
**Important:** All subsequent commands must be run from the `RAG/` directory due to relative file paths.

**2. Install dependencies:**
```bash
pip install openai numpy python-dotenv PyPDF2
```

**3. Set up OpenAI API key:**

Create a `.env` file in the current directory (`RAG/`):
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```
Replace `your-api-key-here` with your actual OpenAI API key.

**4. Verify your data structure:**
```
HamPlan/
├── RAG/                          # ← You should be here
│   ├── inference.py
│   ├── .env
│   ├── syllabi/                  # PDF syllabi by subject
│   └── cache/                    # Auto-created for embeddings
├── course-catalog-scraper/
│   ├── courses_with_prerequisites.json
│   └── department_overviews/txt/
├── prompt_history/               # AI usage documentation
├── CONTRIBUTION.md
└── HamPlan_Report.pdf
```

## Usage

From the `RAG/` directory, run:
```bash
python inference.py
```

**What happens:**
- **First run:** Creates embeddings for all documents (~5-10 minutes)
- **Subsequent runs:** Loads from cache (instant)
- Ask questions interactively with conversation memory

**Example questions:**
- "What are the requirements for a CS major?"
- "What about their prerequisites?" (follow-up)
- "What classes does Professor Kuruwita teach?"

**Commands:**
- `clear` - Reset conversation memory
- `quit` or `exit` - Exit

## Features
- Course catalog search with prerequisites
- Syllabus content retrieval
- Department requirements lookup
- Conversation memory (remembers context)
- Automatic embedding caching
- Multi-source answers (catalog + syllabi + department info)

## Technical Details
- **Embeddings:** OpenAI `text-embedding-3-large`
- **LLM:** GPT-4 Turbo
- **Vector Search:** Cosine similarity (top-k=50)
- **Memory:** Automatic conversation trimming and summarization

## License
Academic project - Hamilton College

**Questions?** Contact the authors at Hamilton College.
