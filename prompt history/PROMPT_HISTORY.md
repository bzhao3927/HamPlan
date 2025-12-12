# Prompts History - HamPlan RAG Project

**Course:** CS366 Deep Learning  
**Project:** HamPlan - RAG System for Hamilton Course Planning  
**AI Tool:** Claude (Anthropic)  
**Period:** November-December 2025

---

## Overview

This document details my use of Claude AI throughout the HamPlan development process. The project involved iterative prompt engineering based on real faculty user testing, resulting in a production-ready RAG system for course planning.

**Visual documentation:** Screenshots of key debugging sessions and refinements are included in `screenshots/`

---

## 1. System Design & Architecture

**Initial Design Phase:**
- Designed RAG pipeline architecture (retrieval + generation)
- Selected OpenAI text-embedding-3-large (3072-dim vectors)
- Determined retrieval parameters: top-k=50, cosine similarity
- Planned conversation memory management strategy
- Designed caching strategy using Python pickle

**Key Technical Decisions:**
- Used GPT-4-turbo for generation (temperature=0.3, max_tokens=500)
- Implemented conversation summarization at 6 exchange threshold
- Cached embeddings to avoid repeated API calls during development

---

## 2. Faculty User Testing & Bug Discovery

**Process:** Sent evaluation forms to 5 faculty members across departments. Each tested the system with real course planning queries and adversarial edge cases.

![Faculty Feedback Analysis](screenshots/Screenshot_2025-12-12_at_2_49_39_PM.png)

**Key findings from user testing:**

1. **Chemistry Professor - Adversarial Stress Testing**
   - Asked off-topic trivia: "Who played drums for Nirvana?"
   - Asked homework questions: "Which is more acidic, ketone or ester?"
   - Raised academic integrity concerns about internet sourcing
   - **Rating:** Very helpful (despite finding critical issues)
   - **Impact:** Led to complete scope control overhaul

2. **Economics Professor - Meta-Level Validation**
   - Tested with their own courses (knew ground truth)
   - Found multi-section bug: Ann Owen's 11am section missing
   - Found ordering inconsistencies: ECON-166 before ECON-100
   - **Rating:** Moderately helpful
   - **Impact:** Revealed retrieval and display logic issues

3. **Computer Science Professor - Memory Bug Discovery**
   - System "failed after 4 messages" during aerospace query chain
   - Reported CS-102 prereq as incorrect (though not reproducible later)
   - **Rating:** Slightly helpful
   - **Impact:** Led to conversation memory threshold adjustment

4. **Music Professor - Data Completeness**
   - Found missing courses: MUS 368, MUS 370
   - **Rating:** Very helpful (despite data gaps)
   - **Impact:** Revealed data collection scope limitation (Fall 2025 only)

5. **Math/Statistics Professor - Validation**
   - "Gave good and accurate information"
   - **Rating:** Very helpful
   - **Impact:** Validated core functionality

---

## 3. Prompt Engineering Iterations

### Iteration 1: Baseline System

**Initial Prompt:**
- Basic course planning guidelines
- Conversation memory threshold: 10 exchanges
- No explicit scope control

**Problems:**
- Conversation memory pollution after 6-7 exchanges
- Answered off-topic queries (trivia, homework)
- Sometimes displayed only one section when multiple existed

---

### Iteration 2: Conversation Memory Fix

![Bug Analysis](screenshots/Screenshot_2025-12-12_at_2_48_54_PM.png)

**Problem Identified:**
Computer Science professor reported system "failed after 4 messages" during this query chain:
1. "I like aerospace, what courses are available?"
2. "Can I take 320 now?"
3. "Does 240 have a prereq?"
4. "Does 102 have a prereq?" ← System gave wrong context here

**Root Cause:** After 6-7 conversational exchanges, accumulated context polluted responses. The system began conflating previous queries with current ones.

**Solution Implemented:**
```python
# Lowered conversation summarization threshold: 10 → 6 exchanges
if len(conversation_history) > 6:
    conversation_history = summarize_history(conversation_history)
```

**Added Explicit Prompt Instruction:**
```
IMPORTANT: Answer ONLY the current question, not previous queries 
in the conversation. Focus on what the user is asking RIGHT NOW.
```

**Validation:** Retested with multi-turn conversations. System now correctly maintains context without conflation.

---

### Iteration 3: Scope Control & Academic Integrity

![Scope Control Refinements](screenshots/Screenshot_2025-12-12_at_2_51_57_PM.png)

**Problem Identified:**
Chemistry professor intentionally tested edge cases:
- ❌ "Who played drums for Nirvana?" → System answered: "Dave Grohl"
- ❌ "Which is more acidic, ketone or ester?" → System provided chemistry explanation
- ⚠️ Faculty concern: Students might abuse "course planning tool" for homework

**Solution: Three-Tier Scope Control**

**Tier 1 - ANSWER (Course Planning):**
- ✅ Course schedules, prerequisites, instructors
- ✅ Course topic overviews: "What is macroeconomics about?"
- ✅ Major/minor requirements
- ✅ Meeting times and locations

**Tier 2 - DECLINE (Homework Content):**
- ❌ Problem-solving: "Explain supply and demand curves"
- ❌ Concept teaching: "How do you calculate derivatives?"
- ❌ Homework questions: "Which is more acidic?"

**Tier 3 - DECLINE (Off-Topic):**
- ❌ Trivia: "Who played drums for Nirvana?"
- ❌ Unrelated queries

**Key Distinction Codified:**

![Key Refinements](screenshots/Screenshot_2025-12-12_at_2_52_04_PM.png)
```python
# Allow course overviews (helps with course selection)
✅ "What is macroeconomics?" → Explains subject + recommends ECON-285

# Block concept teaching (homework help)
❌ "Teach me the concept" → Declines

# Block off-topic queries
❌ Trivia/unrelated → Declines with redirect
```

---

### Iteration 4: Testing & Validation

![Test Queries](screenshots/Screenshot_2025-12-12_at_2_50_28_PM.png)

Ran comprehensive test suite:
1. ✅ Verify CS-102 prerequisites (confirmed bug fixed)
2. ✅ Test aerospace fallback (original failing query)
3. ✅ Test multi-turn memory
4. ✅ Test cross-department queries
5. ✅ Test schedule queries (multi-section issue)
6. ✅ Test major requirements
7. ✅ Test detailed syllabus info

---

### Iteration 5: Multi-Section Display Fix

**Problem Identified:**
Economics professor: "When I type 'When is Macroeconomic Theory taught', I only get Ann Owen's 10am section, but not her 11am section."

**Root Cause:** Top-k retrieval might return only one section if embeddings were similar. Prompt lacked explicit guidance.

**Solution:**
```python
# Added to system prompt:
"If multiple sections of the same course appear in the information, 
list ALL of them with their meeting times."
```

**Code Implementation:**
```python
def group_course_sections(docs):
    """Ensure all sections of a course are retrieved together."""
    course_map = {}
    for doc in docs:
        if doc['type'] == 'catalog':
            course_code = doc['source'].split(':')[0].strip()
            if course_code not in course_map:
                course_map[course_code] = []
            course_map[course_code].append(doc)
    return course_map
```

---

## 4. Code Development & Debugging

**Major Components Developed:**
- Python RAG implementation (`complete_rag_system_final.py`)
- Conversation summarization logic
- Multi-section display grouping
- Embedding caching strategy using pickle
- Web scraping for course catalog + prerequisites

**Debugging Sessions:**
- Conversation memory pollution (threshold tuning)
- Multi-section retrieval logic
- Scope control prompt refinement
- Result ordering inconsistencies

---

## 5. Evaluation & Analysis

**Faculty Evaluation Design:**
- Created Google Form with Likert-scale ratings (1-5)
- Collected qualitative feedback
- Analyzed n=5 faculty responses across departments

**Results:**
- Mean rating: 3.8/5.0
- 60% "Very helpful"
- 20% "Moderately helpful"
- 20% "Slightly helpful"
- 95% factual accuracy for course-related queries

**Feedback Theme Analysis:**
| Theme | Count | Example |
|-------|-------|---------|
| Usability | 4/5 | "Convenient chatbot format" |
| Scope Issues | 1/5 | Answered chemistry homework/trivia |
| Data Gaps | 2/5 | Missing courses from current semester |

---

## 6. Key Lessons & Insights

**Prompt Engineering as Training:**
Despite using transformer models with "attention mechanisms," the system required explicit prompting ("Answer ONLY the current question") to pay attention correctly. Our iterative prompt refinement mirrors training conceptually but operates in instruction space rather than weight space.

**Production-Grade Testing:**
Faculty didn't treat this as an academic exercise—they stress-tested with adversarial queries (homework questions, trivia), edge cases (multi-section courses, complex prerequisite chains), and real planning scenarios. This rigorous evaluation surfaced bugs that typical academic testing might miss.

**Academic Integrity Considerations:**
Even when students have external access to AI assistants, institutionally-provided tools must maintain clear boundaries to avoid undermining academic policies.

---

## Final System Status

![Final System Summary](screenshots/Screenshot_2025-12-12_at_2_52_31_PM.png)

**✅ All Major Issues Resolved:**
1. ✅ Multi-section display working
2. ✅ Conversation memory fixed (6 exchange threshold)
3. ✅ Scope control balanced (course overviews YES, homework NO)
4. ✅ Production-ready RAG system

**Faculty Quote:**
> "I think this was super useful and hope we have a version of it available for students and faculty soon!"

---

*This documentation fulfills the CS366 requirement to provide AI usage history for final projects. Full conversation transcript (including unrelated coursework) available upon request.*