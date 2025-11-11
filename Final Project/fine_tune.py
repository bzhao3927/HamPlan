#!/usr/bin/env python3
"""
Fine-tune SentenceTransformer Embeddings on Syllabus Data
This script trains the embedding model to better understand syllabus-specific queries
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from pathlib import Path
import pickle
import random
from datetime import datetime
import json

# Optional: Weights & Biases for tracking (install: pip install wandb)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Install with: pip install wandb")
    print("   Continuing without W&B tracking...\n")

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
    "learning_rate": 2e-5,
}

# =============================================
# LOAD CACHED DOCUMENTS
# =============================================
def load_cached_documents(cache_path="cache/"):
    """Load pre-embedded documents from cache."""
    cache_file = Path(cache_path) / "documents.pkl"
    
    if not cache_file.exists():
        print(f"❌ No cache found at {cache_file}")
        print("Run your RAG script first to create embeddings!")
        return None
    
    with open(cache_file, "rb") as f:
        documents = pickle.load(f)
    
    print(f"✅ Loaded {len(documents)} documents from cache")
    return documents

# =============================================
# GENERATE TRAINING DATA
# =============================================
def generate_training_pairs(documents):
    """
    Generate training pairs from syllabus documents.
    Creates (question, relevant_text) pairs for contrastive learning.
    """
    
    print("\n📝 Generating training pairs from documents...")
    
    # Template questions for different syllabus sections
    question_templates = {
        "prerequisites": [
            "What are the prerequisites for {}?",
            "What background is needed for {}?",
            "What courses should I take before {}?",
            "What are the requirements for {}?"
        ],
        "grading": [
            "What is the grading policy for {}?",
            "How is {} graded?",
            "What is the grade breakdown for {}?",
            "How are assignments weighted in {}?"
        ],
        "topics": [
            "What topics are covered in {}?",
            "What will I learn in {}?",
            "What does {} teach?",
            "What is the curriculum for {}?"
        ],
        "logistics": [
            "When does {} meet?",
            "What are the office hours for {}?",
            "What is the attendance policy for {}?",
            "What are the deadlines for {}?"
        ],
        "materials": [
            "What textbooks are required for {}?",
            "What materials do I need for {}?",
            "What software is used in {}?",
            "What resources are needed for {}?"
        ]
    }
    
    training_pairs = []
    
    # Group documents by subject
    subjects = {}
    for doc in documents:
        subject = doc.get("subject", "Unknown")
        if subject not in subjects:
            subjects[subject] = []
        subjects[subject].append(doc)
    
    # Generate pairs for each subject
    for subject, docs in subjects.items():
        # Sample some documents from this subject
        sample_size = min(20, len(docs))
        sampled_docs = random.sample(docs, sample_size)
        
        for doc in sampled_docs:
            text = doc["text"]
            
            # Infer question type from content
            text_lower = text.lower()
            
            if any(word in text_lower for word in ["prerequisite", "require", "background"]):
                questions = question_templates["prerequisites"]
            elif any(word in text_lower for word in ["grade", "exam", "assignment", "homework"]):
                questions = question_templates["grading"]
            elif any(word in text_lower for word in ["cover", "learn", "topic", "introduce"]):
                questions = question_templates["topics"]
            elif any(word in text_lower for word in ["office hour", "meet", "schedule", "deadline"]):
                questions = question_templates["logistics"]
            elif any(word in text_lower for word in ["textbook", "material", "software", "tool"]):
                questions = question_templates["materials"]
            else:
                questions = question_templates["topics"]  # Default
            
            # Create a question
            question = random.choice(questions).format(subject)
            training_pairs.append((question, text))
    
    print(f"✅ Generated {len(training_pairs)} training pairs")
    return training_pairs

# =============================================
# PREPARE TRAINING DATA
# =============================================
def prepare_training_data(training_pairs, split_ratio=0.8):
    """
    Convert training pairs to InputExample format and split train/val.
    """
    
    # Shuffle pairs
    random.shuffle(training_pairs)
    
    # Convert to InputExample format
    examples = []
    for question, text in training_pairs:
        # Create positive pair (question, relevant text, score=1.0)
        example = InputExample(texts=[question, text], label=1.0)
        examples.append(example)
    
    # Split into train/val
    split_idx = int(len(examples) * split_ratio)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    print(f"📊 Training examples: {len(train_examples)}")
    print(f"📊 Validation examples: {len(val_examples)}")
    
    return train_examples, val_examples

# =============================================
# FINE-TUNING
# =============================================
def finetune_model(train_examples, val_examples, config):
    """
    Fine-tune SentenceTransformer model on syllabus data.
    """
    
    print("\n" + "="*70)
    print("🚀 STARTING EMBEDDING FINE-TUNING")
    print("="*70)
    
    # Initialize W&B if available
    if WANDB_AVAILABLE:
        wandb.init(
            project="syllabus-rag-finetuning",
            config=config,
            name=f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Load base model
    print(f"\n📥 Loading base model: {config['model_name']}")
    model = SentenceTransformer(config['model_name'])
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_examples, 
        shuffle=True, 
        batch_size=config['batch_size']
    )
    
    # Define loss function - CosineSimilarityLoss for semantic similarity
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Create evaluator for validation
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples,
        name='validation'
    )
    
    # Calculate total steps
    num_train_steps = len(train_dataloader) * config['epochs']
    
    print(f"\n📊 Training Configuration:")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Warmup steps: {config['warmup_steps']}")
    print(f"   Total steps: {num_train_steps}")
    print(f"   Evaluation steps: {config['evaluation_steps']}")
    
    # Custom callback to log to W&B
    class WandbCallback:
        def __init__(self):
            self.step = 0
        
        def __call__(self, score, epoch, steps):
            self.step = steps
            if WANDB_AVAILABLE:
                wandb.log({
                    "validation_score": score,
                    "epoch": epoch,
                    "step": steps
                })
    
    callback = WandbCallback() if WANDB_AVAILABLE else None
    
    # Fine-tune the model
    print(f"\n🔄 Starting training...")
    print("-"*70)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=config['epochs'],
        evaluation_steps=config['evaluation_steps'],
        warmup_steps=config['warmup_steps'],
        output_path=config['output_path'],
        optimizer_params={'lr': config['learning_rate']},
        callback=callback,
        show_progress_bar=True
    )
    
    print("-"*70)
    print(f"✅ Training complete!")
    print(f"💾 Model saved to: {config['output_path']}")
    
    if WANDB_AVAILABLE:
        wandb.finish()
    
    return model

# =============================================
# EVALUATION
# =============================================
def evaluate_model(model, val_examples):
    """
    Evaluate the fine-tuned model.
    """
    print("\n" + "="*70)
    print("📊 EVALUATING MODEL")
    print("="*70)
    
    # Create evaluator
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples,
        name='final_evaluation'
    )
    
    # Evaluate
    score = evaluator(model)
    
    # Handle if score is a dict or float
    if isinstance(score, dict):
        # Extract the main score (usually 'cosine_spearman' or similar)
        main_score = score.get('cosine_spearman', score.get('cosine_pearson', list(score.values())[0] if score else 0.0))
        print(f"\n✅ Final Validation Scores:")
        for key, val in score.items():
            print(f"   {key}: {val:.4f}")
    else:
        main_score = score
        print(f"\n✅ Final Validation Score: {main_score:.4f}")
        print("   (Higher is better, max = 1.0)")
    
    return main_score

# =============================================
# COMPARISON TEST
# =============================================
def compare_models(base_model_name, finetuned_model_path, test_queries, documents):
    """
    Compare base model vs fine-tuned model on test queries.
    """
    print("\n" + "="*70)
    print("🔍 COMPARING BASE vs FINE-TUNED MODEL")
    print("="*70)
    
    # Load models
    print(f"\n📥 Loading base model...")
    base_model = SentenceTransformer(base_model_name)
    
    print(f"📥 Loading fine-tuned model...")
    finetuned_model = SentenceTransformer(finetuned_model_path)
    
    # Sample some test documents
    test_docs = random.sample(documents, min(100, len(documents)))
    
    results = []
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        
        # Base model retrieval
        query_emb_base = base_model.encode(query, normalize_embeddings=True)
        scores_base = []
        for doc in test_docs:
            doc_text = doc["text"]
            doc_emb = base_model.encode(doc_text, normalize_embeddings=True)
            score = np.dot(query_emb_base, doc_emb)
            scores_base.append((score, doc))
        
        scores_base.sort(key=lambda x: x[0], reverse=True)
        top_doc_base = scores_base[0][1]
        top_score_base = scores_base[0][0]
        
        # Fine-tuned model retrieval
        query_emb_ft = finetuned_model.encode(query, normalize_embeddings=True)
        scores_ft = []
        for doc in test_docs:
            doc_text = doc["text"]
            doc_emb = finetuned_model.encode(doc_text, normalize_embeddings=True)
            score = np.dot(query_emb_ft, doc_emb)
            scores_ft.append((score, doc))
        
        scores_ft.sort(key=lambda x: x[0], reverse=True)
        top_doc_ft = scores_ft[0][1]
        top_score_ft = scores_ft[0][0]
        
        print(f"   Base model top score: {top_score_base:.4f}")
        print(f"   Fine-tuned top score: {top_score_ft:.4f}")
        print(f"   Improvement: {top_score_ft - top_score_base:.4f}")
        
        results.append({
            "query": query,
            "base_score": float(top_score_base),
            "finetuned_score": float(top_score_ft),
            "improvement": float(top_score_ft - top_score_base)
        })
    
    # Summary
    avg_base = np.mean([r["base_score"] for r in results])
    avg_ft = np.mean([r["finetuned_score"] for r in results])
    avg_improvement = np.mean([r["improvement"] for r in results])
    
    print(f"\n{'='*70}")
    print("📊 SUMMARY")
    print(f"{'='*70}")
    print(f"Average base model score: {avg_base:.4f}")
    print(f"Average fine-tuned score: {avg_ft:.4f}")
    print(f"Average improvement: {avg_improvement:.4f} ({avg_improvement/avg_base*100:.1f}%)")
    
    # Save results
    results_file = "comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "summary": {
                "avg_base_score": float(avg_base),
                "avg_finetuned_score": float(avg_ft),
                "avg_improvement": float(avg_improvement),
                "improvement_percentage": float(avg_improvement/avg_base*100)
            },
            "per_query_results": results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")

# =============================================
# MAIN
# =============================================
def main():
    print("="*70)
    print("🎯 SENTENCE TRANSFORMER FINE-TUNING FOR SYLLABUS RAG")
    print("="*70)
    
    # Load documents
    documents = load_cached_documents(CONFIG["cache_path"])
    if not documents:
        return
    
    # Generate training pairs
    training_pairs = generate_training_pairs(documents)
    
    # Prepare training data
    train_examples, val_examples = prepare_training_data(training_pairs)
    
    # Fine-tune model
    model = finetune_model(train_examples, val_examples, CONFIG)
    
    # Evaluate
    evaluate_model(model, val_examples)
    
    # Compare with base model
    test_queries = [
        "What are the prerequisites for machine learning?",
        "What is the grading policy?",
        "Which courses cover neural networks?",
        "What textbooks are required?",
        "What are the office hours?"
    ]
    
    compare_models(
        CONFIG["model_name"],
        CONFIG["output_path"],
        test_queries,
        documents
    )
    
    print("\n" + "="*70)
    print("✅ FINE-TUNING COMPLETE!")
    print("="*70)
    print(f"\n💡 Next steps:")
    print(f"   1. Your fine-tuned model is saved at: {CONFIG['output_path']}")
    print(f"   2. Update your RAG script to use this model:")
    print(f"      model = SentenceTransformer('{CONFIG['output_path']}')")
    print(f"   3. Check comparison_results.json for performance improvements")
    print(f"   4. If using W&B, check your dashboard for learning curves")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    main()