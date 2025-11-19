#!/usr/bin/env python3
"""
Course Prerequisite Network Analysis
Builds and analyzes the Hamilton College course network
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
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  plotly not installed. Install with: pip install plotly")
    print("   Continuing with matplotlib only...\n")

# For community detection (optional)
try:
    import community as community_louvain
    COMMUNITY_AVAILABLE = True
except ImportError:
    COMMUNITY_AVAILABLE = False
    print("⚠️  python-louvain not installed. Install with: pip install python-louvain")
    print("   Skipping community detection...\n")

# =============================================
# CONFIGURATION
# =============================================
CONFIG = {
    "catalog_json": "courses_fall_2025.json",
    "output_dir": "network_analysis/",
}

# =============================================
# LOAD AND PARSE CATALOG
# =============================================
def load_catalog(json_path):
    """Load course catalog from JSON."""
    print(f"📚 Loading catalog: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        courses = json.load(f)
    
    print(f"✅ Loaded {len(courses)} courses")
    return courses

def extract_prerequisites(prereq_text):
    """
    Extract course codes from prerequisite text.
    Returns list of course codes like ['MATH-113', 'CPSC-220']
    """
    if not prereq_text or prereq_text == "None":
        return []
    
    # Pattern: SUBJ-NUM or SUBJ NUM
    pattern = r'\b([A-Z]{2,4})[-\s]?(\d{3})\b'
    matches = re.findall(pattern, prereq_text.upper())
    
    # Convert to standard format: SUBJ-NUM
    prereqs = [f"{subj}-{num}" for subj, num in matches]
    return list(set(prereqs))  # Remove duplicates

def parse_courses(courses):
    """
    Parse courses into structured format.
    Returns dict: {course_id: {info}}
    """
    print("\n🔍 Parsing course data...")
    
    course_dict = {}
    
    for course in courses:
        # Extract fields
        subject = course.get('Subject', '').strip()
        number = course.get('CourseNumber', '').strip()
        title = course.get('CourseTitle', 'Untitled')
        description = course.get('CourseDescription', '')
        prereq_text = course.get('Prerequisites', '')
        credits = course.get('Credits', '')
        
        if not subject or not number:
            continue
        
        # Create course ID
        course_id = f"{subject}-{number}"
        
        # Parse prerequisites
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
    
    # Count courses with prerequisites
    with_prereqs = sum(1 for c in course_dict.values() if c['prereqs'])
    print(f"   Courses with prerequisites: {with_prereqs}")
    
    return course_dict

# =============================================
# BUILD NETWORK GRAPH
# =============================================
def build_network(course_dict):
    """
    Build directed graph: course -> prerequisites
    Edge direction: A -> B means "B is a prerequisite for A"
    """
    print("\n🕸️  Building network graph...")
    
    G = nx.DiGraph()
    
    # Add all courses as nodes
    for course_id, info in course_dict.items():
        G.add_node(
            course_id,
            subject=info['subject'],
            number=info['number'],
            title=info['title'],
            credits=info['credits']
        )
    
    # Add prerequisite edges
    edge_count = 0
    for course_id, info in course_dict.items():
        for prereq_id in info['prereqs']:
            # Only add edge if prerequisite exists in catalog
            if prereq_id in course_dict:
                G.add_edge(course_id, prereq_id)  # course depends on prereq
                edge_count += 1
    
    print(f"✅ Network built:")
    print(f"   Nodes (courses): {G.number_of_nodes()}")
    print(f"   Edges (prerequisites): {G.number_of_edges()}")
    print(f"   Density: {nx.density(G):.4f}")
    
    return G

# =============================================
# NETWORK ANALYSIS
# =============================================
def analyze_network(G, course_dict):
    """Compute network statistics."""
    print("\n📊 Analyzing network...")
    
    results = {}
    
    # 1. DEGREE CENTRALITY
    # In-degree = how many courses depend on this one (foundational courses)
    # Out-degree = how many prerequisites this course has
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    
    # Top foundational courses (most courses depend on them)
    top_foundational = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:20]
    
    print("\n🏛️  TOP FOUNDATIONAL COURSES (most depended upon):")
    for course_id, count in top_foundational[:10]:
        title = course_dict.get(course_id, {}).get('title', 'Unknown')
        print(f"   {course_id:15} ({count:2} courses) - {title}")
    
    results['top_foundational'] = top_foundational
    
    # Top advanced courses (most prerequisites)
    top_advanced = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:20]
    
    print("\n🎓 TOP ADVANCED COURSES (most prerequisites):")
    for course_id, count in top_advanced[:10]:
        title = course_dict.get(course_id, {}).get('title', 'Unknown')
        print(f"   {course_id:15} ({count:2} prereqs) - {title}")
    
    results['top_advanced'] = top_advanced
    
    # 2. BETWEENNESS CENTRALITY
    # Courses that are "bridges" in the prerequisite chain
    print("\n🌉 Computing betweenness centrality...")
    betweenness = nx.betweenness_centrality(G)
    top_bridges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:20]
    
    print("   TOP BRIDGE COURSES:")
    for course_id, score in top_bridges[:10]:
        title = course_dict.get(course_id, {}).get('title', 'Unknown')
        print(f"   {course_id:15} ({score:.4f}) - {title}")
    
    results['top_bridges'] = top_bridges
    
    # 3. PAGERANK
    # Importance based on graph structure
    print("\n⭐ Computing PageRank...")
    pagerank = nx.pagerank(G)
    top_important = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:20]
    
    print("   TOP IMPORTANT COURSES (PageRank):")
    for course_id, score in top_important[:10]:
        title = course_dict.get(course_id, {}).get('title', 'Unknown')
        print(f"   {course_id:15} ({score:.4f}) - {title}")
    
    results['top_important'] = top_important
    
    # 4. CONNECTED COMPONENTS
    print("\n🔗 Analyzing connectivity...")
    
    # Convert to undirected for component analysis
    G_undirected = G.to_undirected()
    components = list(nx.connected_components(G_undirected))
    
    print(f"   Connected components: {len(components)}")
    print(f"   Largest component size: {len(max(components, key=len))}")
    
    results['components'] = components
    
    # 5. SUBJECT ANALYSIS
    print("\n📚 Analyzing by subject...")
    
    subjects = defaultdict(list)
    for node in G.nodes():
        subject = G.nodes[node].get('subject', 'Unknown')
        subjects[subject].append(node)
    
    print(f"   Number of subjects: {len(subjects)}")
    print("   Top subjects by course count:")
    top_subjects = sorted(subjects.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for subject, courses in top_subjects:
        avg_prereqs = np.mean([out_degree.get(c, 0) for c in courses])
        print(f"   {subject:10} - {len(courses):3} courses, avg {avg_prereqs:.1f} prereqs")
    
    results['subjects'] = subjects
    results['top_subjects'] = top_subjects
    
    return results

# =============================================
# COMMUNITY DETECTION
# =============================================
def detect_communities(G, course_dict):
    """Detect course communities using Louvain algorithm."""
    if not COMMUNITY_AVAILABLE:
        print("\n⚠️  Skipping community detection (library not available)")
        return None
    
    print("\n🏘️  Detecting communities...")
    
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()
    
    # Detect communities
    communities = community_louvain.best_partition(G_undirected)
    
    # Group courses by community
    community_groups = defaultdict(list)
    for course_id, community_id in communities.items():
        community_groups[community_id].append(course_id)
    
    print(f"   Found {len(community_groups)} communities")
    
    # Analyze each community
    print("\n   TOP COMMUNITIES:")
    sorted_communities = sorted(community_groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    for i, (comm_id, courses) in enumerate(sorted_communities[:5], 1):
        print(f"\n   Community {i} ({len(courses)} courses):")
        
        # Count subjects in this community
        subject_counts = Counter([course_dict[c]['subject'] for c in courses if c in course_dict])
        top_subjects = subject_counts.most_common(3)
        
        print(f"     Top subjects: {', '.join([f'{s}({c})' for s, c in top_subjects])}")
        
        # Show sample courses
        sample = courses[:5]
        for course_id in sample:
            if course_id in course_dict:
                title = course_dict[course_id]['title']
                print(f"     - {course_id}: {title}")
    
    return communities

# =============================================
# VISUALIZATIONS
# =============================================
def create_output_dir(output_dir):
    """Create output directory."""
    Path(output_dir).mkdir(exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")

def visualize_network_matplotlib(G, course_dict, results, output_dir):
    """Create static network visualization with matplotlib."""
    print("\n📊 Creating matplotlib visualizations...")
    
    # 1. Full network (spring layout)
    plt.figure(figsize=(20, 20))
    
    # Color nodes by in-degree (foundational courses)
    in_degree = dict(G.in_degree())
    node_colors = [in_degree.get(node, 0) for node in G.nodes()]
    
    # Size nodes by PageRank
    pagerank = nx.pagerank(G)
    node_sizes = [pagerank.get(node, 0) * 10000 for node in G.nodes()]
    
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        cmap='YlOrRd',
        alpha=0.7
    )
    
    nx.draw_networkx_edges(
        G, pos,
        alpha=0.2,
        arrows=True,
        arrowsize=10,
        width=0.5
    )
    
    # Label only top nodes
    top_nodes = [node for node, _ in results['top_foundational'][:15]]
    labels = {node: node for node in top_nodes if node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title("Hamilton Course Prerequisite Network\n(Node color = # of dependent courses, Size = importance)", 
              fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/network_full.png", dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {output_dir}/network_full.png")
    plt.close()
    
    # 2. Degree distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    
    ax1.hist(in_degrees, bins=30, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('In-Degree (# of dependent courses)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('In-Degree Distribution\n(Foundational courses have high in-degree)')
    ax1.grid(alpha=0.3)
    
    ax2.hist(out_degrees, bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax2.set_xlabel('Out-Degree (# of prerequisites)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Out-Degree Distribution\n(Advanced courses have high out-degree)')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/degree_distribution.png", dpi=150)
    print(f"   ✅ Saved: {output_dir}/degree_distribution.png")
    plt.close()
    
    # 3. Subject network (summarized)
    print("\n   Creating subject network...")
    subject_graph = nx.DiGraph()
    
    # Add edges between subjects based on prerequisites
    for course_id, info in course_dict.items():
        course_subject = info['subject']
        for prereq_id in info['prereqs']:
            if prereq_id in course_dict:
                prereq_subject = course_dict[prereq_id]['subject']
                if course_subject != prereq_subject:  # Cross-subject prerequisites
                    if subject_graph.has_edge(course_subject, prereq_subject):
                        subject_graph[course_subject][prereq_subject]['weight'] += 1
                    else:
                        subject_graph.add_edge(course_subject, prereq_subject, weight=1)
    
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(subject_graph, k=2, iterations=50)
    
    # Draw edges with width based on weight
    edges = subject_graph.edges()
    weights = [subject_graph[u][v]['weight'] for u, v in edges]
    
    nx.draw_networkx_edges(
        subject_graph, pos,
        width=[w/2 for w in weights],
        alpha=0.5,
        arrows=True,
        arrowsize=20
    )
    
    nx.draw_networkx_nodes(
        subject_graph, pos,
        node_size=2000,
        node_color='lightblue',
        alpha=0.8
    )
    
    nx.draw_networkx_labels(subject_graph, pos, font_size=10, font_weight='bold')
    
    plt.title("Subject Prerequisite Network\n(Edge width = # of cross-subject prerequisites)", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/subject_network.png", dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {output_dir}/subject_network.png")
    plt.close()

def visualize_network_plotly(G, course_dict, communities, output_dir):
    """Create interactive network visualization with Plotly."""
    if not PLOTLY_AVAILABLE:
        print("\n⚠️  Skipping Plotly visualizations")
        return
    
    print("\n🌐 Creating interactive Plotly visualization...")
    
    # Create layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Extract edge coordinates
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Extract node coordinates
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    in_degree = dict(G.in_degree())
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        title = course_dict.get(node, {}).get('title', 'Unknown')
        prereqs = course_dict.get(node, {}).get('prereqs', [])
        
        text = f"{node}<br>{title}<br>In-degree: {in_degree[node]}<br>Prerequisites: {len(prereqs)}"
        node_text.append(text)
        
        # Color by community if available
        if communities:
            node_colors.append(communities.get(node, 0))
        else:
            node_colors.append(in_degree[node])
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            color=node_colors,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Community' if communities else 'In-Degree',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Interactive Hamilton Course Network',
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0, l=0, r=0, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    output_file = f"{output_dir}/network_interactive.html"
    fig.write_html(output_file)
    print(f"   ✅ Saved: {output_file}")

# =============================================
# SAVE RESULTS
# =============================================
def save_results(G, course_dict, results, communities, output_dir):
    """Save analysis results to files."""
    print("\n💾 Saving results...")
    
    # 1. Network statistics
    stats = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_in_degree': np.mean([d for _, d in G.in_degree()]),
        'avg_out_degree': np.mean([d for _, d in G.out_degree()]),
    }
    
    with open(f"{output_dir}/network_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"   ✅ Saved: network_stats.json")
    
    # 2. Top courses CSV
    data = []
    for course_id in G.nodes():
        if course_id not in course_dict:
            continue
        
        info = course_dict[course_id]
        in_deg = G.in_degree(course_id)
        out_deg = G.out_degree(course_id)
        
        data.append({
            'course_id': course_id,
            'title': info['title'],
            'subject': info['subject'],
            'in_degree': in_deg,
            'out_degree': out_deg,
            'community': communities.get(course_id, -1) if communities else -1
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('in_degree', ascending=False)
    df.to_csv(f"{output_dir}/courses_ranked.csv", index=False)
    print(f"   ✅ Saved: courses_ranked.csv")
    
    # 3. Top foundational courses
    with open(f"{output_dir}/top_foundational.txt", 'w') as f:
        f.write("TOP 50 FOUNDATIONAL COURSES\n")
        f.write("(Most courses depend on these)\n\n")
        for course_id, count in results['top_foundational'][:50]:
            title = course_dict.get(course_id, {}).get('title', 'Unknown')
            f.write(f"{course_id:15} ({count:2} courses) - {title}\n")
    print(f"   ✅ Saved: top_foundational.txt")

# =============================================
# MAIN
# =============================================
def main():
    print("="*70)
    print("🕸️  COURSE PREREQUISITE NETWORK ANALYSIS")
    print("="*70)
    
    # Load catalog
    courses = load_catalog(CONFIG['catalog_json'])
    
    # Parse courses
    course_dict = parse_courses(courses)
    
    # Build network
    G = build_network(course_dict)
    
    # Analyze network
    results = analyze_network(G, course_dict)
    
    # Detect communities
    communities = detect_communities(G, course_dict)
    
    # Create output directory
    create_output_dir(CONFIG['output_dir'])
    
    # Visualizations
    visualize_network_matplotlib(G, course_dict, results, CONFIG['output_dir'])
    visualize_network_plotly(G, course_dict, communities, CONFIG['output_dir'])
    
    # Save results
    save_results(G, course_dict, results, communities, CONFIG['output_dir'])
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📁 Results saved to: {CONFIG['output_dir']}")
    print("\nKey outputs:")
    print(f"  • network_full.png - Full network visualization")
    print(f"  • subject_network.png - Subject-level network")
    print(f"  • degree_distribution.png - Degree distributions")
    print(f"  • network_interactive.html - Interactive exploration")
    print(f"  • courses_ranked.csv - All courses with metrics")
    print(f"  • top_foundational.txt - Most important prerequisite courses")

if __name__ == "__main__":
    main()