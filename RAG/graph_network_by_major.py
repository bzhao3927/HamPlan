#!/usr/bin/env python3
"""
Course Prerequisite Network Analysis - BY MAJOR
Analyzes Hamilton College course networks organized by major/concentration
Shows requirements and pathways to complete each major
"""

import json
import re
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
from pathlib import Path

# For better visualizations (optional)
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  plotly not installed. Install with: pip install plotly")
    print("   Continuing with matplotlib only...\n")

# =============================================
# CONFIGURATION
# =============================================
CONFIG = {
    "catalog_json": "../course-catalog-scraper/courses_with_prerequisites.json",
    "department_overviews": "../course-catalog-scraper/department_overviews/txt",
    "output_dir": "network_analysis_by_major/",
}

# Subject code to department name mapping
SUBJECT_TO_DEPT = {
    'CPSCI': 'Computer Science',
    'ECON': 'Economics',
    'MATH': 'Mathematics',
    'PHYS': 'Physics',
    'CHEM': 'Chemistry',
    'BIO': 'Biology',
    'BICHM': 'Biochemistry',
    'NEURO': 'Neuroscience',
    'PSYCH': 'Psychology',
    'GOVT': 'Government',
    'HIST': 'History',
    'PHIL': 'Philosophy',
    'LIT': 'Literature',
    'CRWR': 'Creative Writing',
    'ART': 'Art',
    'MUSIC': 'Music',
    'THETR': 'Theatre',
    'AFRST': 'Africana Studies',
    'AMST': 'American Studies',
    'ENVST': 'Environmental Studies',
    'WOMST': 'Women\'s Studies',
    'PPOL': 'Public Policy',
    'JLLJS': 'Jurisprudence',
    'ASIAN': 'Asian Studies',
    'MEDVL': 'Medieval Studies',
}

# =============================================
# LOAD DATA
# =============================================
def load_catalog(json_path):
    """Load course catalog from JSON."""
    print(f"📚 Loading catalog: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        courses = json.load(f)

    print(f"✅ Loaded {len(courses)} courses")
    return courses

def extract_prerequisites(prereq_text):
    """Extract course codes from prerequisite text."""
    if not prereq_text or prereq_text == "None":
        return []

    # Pattern: SUBJ-NUM or SUBJ NUM
    pattern = r'\b([A-Z]{2,4})[-\s]?(\d{3})\b'
    matches = re.findall(pattern, prereq_text.upper())

    # Convert to standard format: SUBJ-NUM
    prereqs = [f"{subj}-{num}" for subj, num in matches]
    return list(set(prereqs))

def parse_courses(courses):
    """Parse courses into structured format."""
    print("\n🔍 Parsing course data...")

    course_dict = {}

    for course in courses:
        course_obj = course.get('Course', {})

        subject = course_obj.get('SubjectCode', '').strip()
        number = course_obj.get('Number', '').strip()
        title = course_obj.get('Title', 'Untitled')
        description = course_obj.get('Description', '')
        credits = course.get('MinimumCredits', '')

        # Extract prerequisites from DisplayText
        prerequisites = course.get('Prerequisites', [])
        prereq_texts = []
        for prereq in prerequisites:
            display_text = prereq.get('DisplayText', '')
            if display_text:
                prereq_texts.append(display_text)

        prereq_text = '; '.join(prereq_texts) if prereq_texts else ''

        if not subject or not number:
            continue

        course_id = f"{subject}-{number}"
        prereqs = extract_prerequisites(prereq_text)

        course_dict[course_id] = {
            'id': course_id,
            'subject': subject,
            'number': number,
            'title': title,
            'description': description,
            'prereqs': prereqs,
            'prereq_text': prereq_text,
            'credits': credits,
        }

    print(f"✅ Parsed {len(course_dict)} courses")
    return course_dict

def load_department_requirements(dept_dir):
    """Load and parse department overview files for major requirements."""
    print("\n📖 Loading department requirements...")

    dept_path = Path(dept_dir)
    if not dept_path.exists():
        print(f"⚠️  Department overviews folder not found")
        return {}

    txt_files = list(dept_path.glob("*.txt"))
    print(f"✅ Found {len(txt_files)} department files")

    requirements = {}

    for txt_file in txt_files:
        dept_name = txt_file.stem.replace('-', ' ').title()

        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()

            # Extract course codes mentioned in requirements
            course_pattern = r'\b([A-Z]{2,4})[-\s]?(\d{3})\b'
            matches = re.findall(course_pattern, text)
            required_courses = [f"{subj}-{num}" for subj, num in matches]

            # Look for concentration/major requirements section
            major_section = ""
            if "concentration" in text.lower() or "major" in text.lower():
                lines = text.split('\n')
                in_requirements = False
                for line in lines:
                    if 'concentration' in line.lower() or 'major' in line.lower():
                        in_requirements = True
                    if in_requirements:
                        major_section += line + '\n'
                        if len(major_section) > 1000:  # Limit size
                            break

            requirements[dept_name] = {
                'required_courses': list(set(required_courses)),
                'description': major_section[:500] if major_section else text[:500],
                'file': txt_file.name
            }

        except Exception as e:
            print(f"  ⚠️  Error reading {txt_file.name}: {e}")
            continue

    print(f"✅ Loaded requirements for {len(requirements)} departments\n")
    return requirements

# =============================================
# ORGANIZE BY MAJOR
# =============================================
def organize_courses_by_major(course_dict):
    """Organize courses by their subject/major."""
    print("\n📊 Organizing courses by major...")

    by_major = defaultdict(list)

    for course_id, info in course_dict.items():
        subject = info['subject']
        by_major[subject].append(course_id)

    # Sort majors by number of courses
    sorted_majors = sorted(by_major.items(), key=lambda x: len(x[1]), reverse=True)

    print(f"✅ Found {len(by_major)} majors/subjects")
    print("   Top majors by course count:")
    for subject, courses in sorted_majors[:10]:
        dept_name = SUBJECT_TO_DEPT.get(subject, subject)
        print(f"   {subject:10} ({dept_name:25}) - {len(courses):3} courses")

    return dict(by_major)

# =============================================
# BUILD MAJOR-SPECIFIC NETWORKS
# =============================================
def build_major_network(course_dict, major_courses, all_course_dict):
    """Build network for a specific major including prerequisites from other majors."""
    G = nx.DiGraph()

    # Add major courses as nodes
    for course_id in major_courses:
        if course_id in course_dict:
            info = course_dict[course_id]
            G.add_node(
                course_id,
                subject=info['subject'],
                title=info['title'],
                in_major=True
            )

    # Add prerequisite edges (may include courses from other majors)
    for course_id in major_courses:
        if course_id not in course_dict:
            continue

        info = course_dict[course_id]
        for prereq_id in info['prereqs']:
            # Add prerequisite node if not already in graph
            if prereq_id not in G and prereq_id in all_course_dict:
                prereq_info = all_course_dict[prereq_id]
                G.add_node(
                    prereq_id,
                    subject=prereq_info['subject'],
                    title=prereq_info['title'],
                    in_major=False  # Prerequisite from another major
                )

            # Add edge
            if prereq_id in all_course_dict:
                G.add_edge(course_id, prereq_id)

    return G

# =============================================
# ANALYZE MAJOR
# =============================================
def analyze_major_network(G, major_name, course_dict):
    """Analyze a major's network and return statistics."""
    if G.number_of_nodes() == 0:
        return None

    analysis = {
        'major': major_name,
        'total_courses': G.number_of_nodes(),
        'major_courses': len([n for n in G.nodes() if G.nodes[n].get('in_major', False)]),
        'prerequisite_edges': G.number_of_edges(),
    }

    # In-degree: foundational courses for this major
    in_degree = dict(G.in_degree())
    analysis['foundational'] = sorted(
        [(course_id, degree) for course_id, degree in in_degree.items() if degree > 0],
        key=lambda x: x[1],
        reverse=True
    )[:10]

    # Out-degree: advanced courses with most prerequisites
    out_degree = dict(G.out_degree())
    analysis['advanced'] = sorted(
        [(course_id, degree) for course_id, degree in out_degree.items() if degree > 0],
        key=lambda x: x[1],
        reverse=True
    )[:10]

    # Courses with no prerequisites (entry points)
    analysis['entry_points'] = [
        course_id for course_id, degree in out_degree.items()
        if degree == 0 and G.nodes[course_id].get('in_major', False)
    ]

    # Courses with no dependents (terminal/capstone courses)
    analysis['terminal'] = [
        course_id for course_id, degree in in_degree.items()
        if degree == 0 and G.nodes[course_id].get('in_major', False)
    ]

    return analysis

def compute_course_levels(G, major_name):
    """
    Compute 'level' of each course based on prerequisite depth.
    Level 0 = no prerequisites, Level 1 = depends on level 0, etc.
    """
    if G.number_of_nodes() == 0:
        return {}

    levels = {}

    # Get courses that are in the major
    major_nodes = [n for n in G.nodes() if G.nodes[n].get('in_major', False)]

    # Start with courses with no prerequisites (level 0)
    out_degree = dict(G.out_degree())
    for node in major_nodes:
        if out_degree.get(node, 0) == 0:
            levels[node] = 0

    # Iteratively assign levels
    max_iterations = 20
    for iteration in range(max_iterations):
        changed = False
        for node in major_nodes:
            if node in levels:
                continue

            # Get all prerequisites
            prereqs = list(G.successors(node))
            if not prereqs:
                levels[node] = 0
                changed = True
                continue

            # Check if all prerequisites have been assigned levels
            prereq_levels = []
            all_assigned = True
            for prereq in prereqs:
                if prereq in levels:
                    prereq_levels.append(levels[prereq])
                else:
                    all_assigned = False
                    break

            # If all prerequisites have levels, assign this course's level
            if all_assigned and prereq_levels:
                levels[node] = max(prereq_levels) + 1
                changed = True

        if not changed:
            break

    # Assign remaining courses (those with circular dependencies or isolated) to level 0
    for node in major_nodes:
        if node not in levels:
            levels[node] = 0

    return levels

def analyze_major_pathway(G, major_name, course_dict):
    """Analyze the pathway/progression through a major."""
    if G.number_of_nodes() == 0:
        return None

    # Compute course levels
    levels = compute_course_levels(G, major_name)

    # Group courses by level
    by_level = defaultdict(list)
    for course, level in levels.items():
        by_level[level].append(course)

    # Sort levels
    sorted_levels = sorted(by_level.items())

    pathway = {
        'levels': sorted_levels,
        'max_level': max(by_level.keys()) if by_level else 0,
        'level_map': levels
    }

    return pathway

# =============================================
# VISUALIZATIONS
# =============================================
def visualize_major_network(G, major_name, course_dict, output_dir):
    """Create visualization for a major's network."""
    if G.number_of_nodes() == 0:
        return

    plt.figure(figsize=(16, 12))

    # Separate nodes by whether they're in the major or prerequisites from other majors
    major_nodes = [n for n in G.nodes() if G.nodes[n].get('in_major', False)]
    prereq_nodes = [n for n in G.nodes() if not G.nodes[n].get('in_major', False)]

    # Layout
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)

    # Draw prerequisite nodes (from other majors) in gray
    if prereq_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=prereq_nodes,
            node_color='lightgray',
            node_size=300,
            alpha=0.6,
            label='Prerequisites from other majors'
        )

    # Draw major nodes colored by in-degree (how many courses depend on them)
    if major_nodes:
        in_degree = dict(G.in_degree())
        node_colors = [in_degree.get(node, 0) for node in major_nodes]

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=major_nodes,
            node_color=node_colors,
            node_size=500,
            cmap='YlOrRd',
            alpha=0.8,
            label='Major courses'
        )

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        alpha=0.3,
        arrows=True,
        arrowsize=15,
        width=1.5,
        edge_color='gray'
    )

    # Labels for all nodes
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

    plt.title(f"{major_name} Course Network\n(Node color intensity = # of dependent courses)",
              fontsize=16, fontweight='bold')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()

    # Save
    safe_name = major_name.lower().replace(' ', '_').replace('/', '_')
    plt.savefig(f"{output_dir}/{safe_name}_network.png", dpi=150, bbox_inches='tight')
    plt.close()

def create_major_summary_report(major_analyses, major_pathways, requirements, output_dir):
    """Create a comprehensive summary report of all majors."""
    report_path = f"{output_dir}/major_analysis_summary.txt"

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("HAMILTON COLLEGE MAJOR NETWORK ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")

        # Sort majors by number of courses
        sorted_majors = sorted(major_analyses.items(),
                              key=lambda x: x[1]['total_courses'] if x[1] else 0,
                              reverse=True)

        for major_name, analysis in sorted_majors:
            if not analysis:
                continue

            f.write(f"\n{'='*80}\n")
            f.write(f"{major_name.upper()}\n")
            f.write(f"{'='*80}\n\n")

            f.write(f"Total courses in network: {analysis['total_courses']}\n")
            f.write(f"Courses in major: {analysis['major_courses']}\n")
            f.write(f"Prerequisite relationships: {analysis['prerequisite_edges']}\n\n")

            # Course progression pathway
            pathway = major_pathways.get(major_name)
            if pathway and pathway['levels']:
                f.write(f"COURSE PROGRESSION PATHWAY (by prerequisite depth):\n")
                f.write(f"Maximum depth: {pathway['max_level']} levels\n\n")

                for level, courses in pathway['levels']:
                    if level == 0:
                        f.write(f"  Level {level} (Entry-level - no prerequisites):\n")
                    else:
                        f.write(f"  Level {level} (Build on Level {level-1} courses):\n")

                    for course_id in sorted(courses)[:15]:  # Show up to 15 courses per level
                        f.write(f"    • {course_id}\n")

                    if len(courses) > 15:
                        f.write(f"    ... and {len(courses)-15} more\n")
                    f.write("\n")

            f.write(f"ENTRY POINTS (no prerequisites):\n")
            if analysis['entry_points']:
                for course_id in analysis['entry_points'][:10]:
                    f.write(f"  • {course_id}\n")
            else:
                f.write("  None\n")
            f.write("\n")

            f.write(f"FOUNDATIONAL COURSES (most depended upon):\n")
            if analysis['foundational']:
                for course_id, count in analysis['foundational'][:5]:
                    f.write(f"  • {course_id:15} ({count} courses depend on this)\n")
            else:
                f.write("  None\n")
            f.write("\n")

            f.write(f"ADVANCED COURSES (most prerequisites):\n")
            if analysis['advanced']:
                for course_id, count in analysis['advanced'][:5]:
                    f.write(f"  • {course_id:15} (requires {count} prerequisites)\n")
            else:
                f.write("  None\n")
            f.write("\n")

            f.write(f"TERMINAL COURSES (no dependents - potential capstones):\n")
            if analysis['terminal']:
                for course_id in analysis['terminal'][:10]:
                    f.write(f"  • {course_id}\n")
            else:
                f.write("  None\n")
            f.write("\n")

    print(f"✅ Saved summary report: {report_path}")

# =============================================
# MAIN
# =============================================
def main():
    print("="*80)
    print("🎓 COURSE NETWORK ANALYSIS BY MAJOR")
    print("="*80)

    # Load data
    courses = load_catalog(CONFIG['catalog_json'])
    course_dict = parse_courses(courses)
    requirements = load_department_requirements(CONFIG['department_overviews'])

    # Organize by major
    by_major = organize_courses_by_major(course_dict)

    # Create output directory
    Path(CONFIG['output_dir']).mkdir(exist_ok=True)
    print(f"\n📁 Output directory: {CONFIG['output_dir']}")

    # Analyze each major
    print("\n🔍 Analyzing networks for each major...")
    major_analyses = {}
    major_pathways = {}

    # Focus on majors with substantial courses
    major_subjects = [(subj, courses) for subj, courses in by_major.items() if len(courses) >= 10]
    major_subjects = sorted(major_subjects, key=lambda x: len(x[1]), reverse=True)

    for subject, major_courses in major_subjects[:20]:  # Top 20 majors
        major_name = SUBJECT_TO_DEPT.get(subject, subject)
        print(f"\n  📚 {major_name} ({subject})...")

        # Build network
        G = build_major_network(course_dict, major_courses, course_dict)

        # Analyze
        analysis = analyze_major_network(G, major_name, course_dict)
        major_analyses[major_name] = analysis

        # Compute pathway
        pathway = analyze_major_pathway(G, major_name, course_dict)
        major_pathways[major_name] = pathway

        if analysis:
            print(f"     Courses: {analysis['major_courses']}, Prerequisites: {analysis['prerequisite_edges']}")
            print(f"     Entry points: {len(analysis['entry_points'])}, Terminal: {len(analysis['terminal'])}")
            if pathway:
                print(f"     Progression levels: {pathway['max_level'] + 1}")

        # Visualize
        visualize_major_network(G, major_name, course_dict, CONFIG['output_dir'])
        print(f"     ✅ Saved visualization")

    # Create summary report
    print("\n📝 Creating summary report...")
    create_major_summary_report(major_analyses, major_pathways, requirements, CONFIG['output_dir'])

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📁 Results saved to: {CONFIG['output_dir']}")
    print("\nKey outputs:")
    print(f"  • major_analysis_summary.txt - Comprehensive analysis of all majors")
    print(f"  • [major_name]_network.png - Network visualization for each major")

if __name__ == "__main__":
    main()
