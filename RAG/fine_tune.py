#!/usr/bin/env python3
"""
Fine-tune SentenceTransformer on BOTH Course Catalog JSON + Syllabus PDFs
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from pathlib import Path
import pickle
import random
import json
from datetime import datetime

# Optional: W&B
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# =============================================
# CONFIGURATION
# =============================================
CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "epochs": 10,
    "batch_size": 16,
    "warmup_steps": 100,
    "evaluation_steps": 50,
    "output_path": "./finetuned_model",
    "cache_path": "cache/",
    "catalog_json": "../course-catalog-scraper/courses_fall_2025.json",  # ADD THIS
    "learning_rate": 2e-5,
}

# =============================================
# LOAD COURSE CATALOG JSON
# =============================================
def load_course_catalog(json_path):
    """Load course catalog from JSON."""
    print(f"\n📚 Loading course catalog from: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        courses = json.load(f)
    
    print(f"✅ Loaded {len(courses)} courses from catalog")
    
    # Extract useful info from each course
    catalog_docs = []
    for course in courses:
        # Extract key fields
        subject = course.get('Subject', '')
        course_number = course.get('CourseNumber', '')
        title = course.get('CourseTitle', '')
        description = course.get('CourseDescription', '')
        credits = course.get('Credits', '')
        prereqs = course.get('Prerequisites', '')
        
        # Create searchable text
        text_parts = []
        if subject and course_number:
            text_parts.append(f"{subject} {course_number}: {title}")
        if description:
            text_parts.append(f"Description: {description}")
        if prereqs:
            text_parts.append(f"Prerequisites: {prereqs}")
        if credits:
            text_parts.append(f"Credits: {credits}")
        
        full_text = "\n".join(text_parts)
        
        if full_text.strip():
            catalog_docs.append({
                "text": full_text,
                "source": f"Catalog: {subject} {course_number}",
                "type": "catalog",
                "subject": subject,
                "course_number": course_number
            })
    
    print(f"✅ Processed {len(catalog_docs)} course catalog entries")
    return catalog_docs

# =============================================
# LOAD SYLLABUS PDFs (from cache)
# =============================================
def load_cached_syllabi(cache_path="cache/"):
    """Load pre-embedded syllabi from cache."""
    cache_file = Path(cache_path) / "documents.pkl"
    
    if not cache_file.exists():
        print(f"⚠️  No syllabus cache found at {cache_file}")
        return []
    
    with open(cache_file, "rb") as f:
        documents = pickle.load(f)
    
    print(f"✅ Loaded {len(documents)} syllabus chunks from cache")
    return documents

# =============================================
# GENERATE TRAINING PAIRS (ENHANCED)
# =============================================
def generate_training_pairs(catalog_docs, syllabus_docs):
    """
    Generate training pairs from BOTH catalog and syllabi.
    """
    
    print("\n📝 Generating training pairs...")
    
    # Question templates
    templates = {
        "prerequisites": [
            "What are the prerequisites for {}?",
            "What do I need before taking {}?",
            "Requirements for {}?",
        ],
        "description": [
            "What is {} about?",
            "Tell me about {}",
            "What does {} cover?",
        ],
        "grading": [
            "How is {} graded?",
            "What's the grading policy for {}?",
        ],
        "credits": [
            "How many credits is {}?",
            "Credit hours for {}?",
        ]
    }
    
    training_pairs = []
    
    # 1. CATALOG PAIRS
    print("  Generating catalog pairs...")
    for doc in catalog_docs:
        text = doc["text"]
        subject = doc.get("subject", "")
        course_num = doc.get("course_number", "")
        course_name = f"{subject} {course_num}"
        
        text_lower = text.lower()
        
        # Match question type to content
        if "prerequisite" in text_lower or "requirement" in text_lower:
            questions = templates["prerequisites"]
        elif "credit" in text_lower:
            questions = templates["credits"]
        else:
            questions = templates["description"]
        
        # Create 2-3 pairs per course
        for _ in range(min(2, len(questions))):
            question = random.choice(questions).format(course_name)
            training_pairs.append((question, text, "catalog"))
    
    # 2. SYLLABUS PAIRS
    print("  Generating syllabus pairs...")
    syllabus_sample = random.sample(syllabus_docs, min(200, len(syllabus_docs)))
    
    for doc in syllabus_sample:
        text = doc["text"]
        subject = doc.get("subject", "this course")
        
        text_lower = text.lower()
        
        if "grade" in text_lower or "exam" in text_lower:
            questions = templates["grading"]
        elif "prerequisite" in text_lower:
            questions = templates["prerequisites"]
        else:
            questions = templates["description"]
        
        question = random.choice(questions).format(subject)
        training_pairs.append((question, text, "syllabus"))
    
    print(f"✅ Generated {len(training_pairs)} training pairs")
    print(f"   Catalog pairs: {len([p for p in training_pairs if p[2] == 'catalog'])}")
    print(f"   Syllabus pairs: {len([p for p in training_pairs if p[2] == 'syllabus'])}")
    
    return training_pairs

# =============================================
# REST OF THE CODE (same as before)
# =============================================
def prepare_training_data(training_pairs, split_ratio=0.8):
    """Convert to InputExample and split train/val."""
    random.shuffle(training_pairs)
    
    examples = []
    for question, text, source_type in training_pairs:
        example = InputExample(texts=[question, text], label=1.0)
        examples.append(example)
    
    split_idx = int(len(examples) * split_ratio)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    print(f"📊 Training: {len(train_examples)}, Validation: {len(val_examples)}")
    return train_examples, val_examples

def finetune_model(train_examples, val_examples, config):
    """Fine-tune model."""
    print("\n" + "="*70)
    print("🚀 STARTING FINE-TUNING")
    print("="*70)
    
    if WANDB_AVAILABLE:
        wandb.init(project="syllabus-catalog-rag", config=config)
    
    model = SentenceTransformer(config['model_name'])
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=config['batch_size'])
    train_loss = losses.CosineSimilarityLoss(model)
    
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples, name='validation'
    )
    
    print(f"\n📊 Epochs: {config['epochs']}, Batch: {config['batch_size']}, LR: {config['learning_rate']}")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=config['epochs'],
        evaluation_steps=config['evaluation_steps'],
        warmup_steps=config['warmup_steps'],
        output_path=config['output_path'],
        optimizer_params={'lr': config['learning_rate']},
        show_progress_bar=True
    )
    
    print(f"✅ Model saved to: {config['output_path']}")
    if WANDB_AVAILABLE:
        wandb.finish()
    
    return model

# =============================================
# MAIN
# =============================================
def main():
    print("="*70)
    print("🎯 FINE-TUNING ON COURSE CATALOG + SYLLABI")
    print("="*70)
    
    # Load course catalog JSON
    catalog_docs = load_course_catalog(CONFIG["catalog_json"])
    
    # Load syllabus PDFs (from cache)
    syllabus_docs = load_cached_syllabi(CONFIG["cache_path"])
    
    if not catalog_docs and not syllabus_docs:
        print("❌ No data found! Need either catalog or syllabi.")
        return
    
    # Generate training pairs from BOTH sources
    training_pairs = generate_training_pairs(catalog_docs, syllabus_docs)
    
    # Prepare data
    train_examples, val_examples = prepare_training_data(training_pairs)
    
    # Fine-tune
    model = finetune_model(train_examples, val_examples, CONFIG)
    
    print("\n✅ DONE!")
    print(f"📁 Model: {CONFIG['output_path']}")
    print(f"🔄 Update RAG to use: SentenceTransformer('{CONFIG['output_path']}')")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()