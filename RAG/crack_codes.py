import json
from collections import Counter, defaultdict

def analyze_requirement_codes(json_file):
    """Extract all unique requirement codes from the data"""
    print(f"Loading {json_file}...")
    with open(json_file, 'r') as f:
        courses_data = json.load(f)
    
    codes = []
    course_examples = {}
    
    for term, courses in courses_data.items():
        for course in courses:
            req_code = course.get('RequirementCode')
            if req_code:
                codes.append(req_code)
                if req_code not in course_examples:
                    course_examples[req_code] = {
                        'course': course.get('CourseName'),
                        'title': course.get('CourseTitle', ''),
                        'term': term
                    }
    
    print(f"\n{'='*70}")
    print(f"REQUIREMENT CODES ANALYSIS")
    print(f"{'='*70}")
    print(f"Total courses with requirement codes: {len(codes)}")
    print(f"Unique requirement codes: {len(set(codes))}")
    print(f"Courses without requirement codes: {sum(len(courses) for courses in courses_data.values()) - len(codes)}\n")
    
    # Show most common codes with examples
    print("Most common requirement codes:\n")
    print(f"{'Code':<8} {'Count':<7} {'Example Course':<15} {'Title'}")
    print(f"{'-'*70}")
    
    for code, count in Counter(codes).most_common(30):
        example = course_examples[code]
        title = example['title'][:40] + '...' if len(example['title']) > 40 else example['title']
        print(f"{code:<8} {count:<7} {example['course']:<15} {title}")
    
    # Save full list
    analysis_output = {
        'summary': {
            'total_courses_with_codes': len(codes),
            'unique_codes': len(set(codes)),
            'total_courses': sum(len(courses) for courses in courses_data.values())
        },
        'unique_codes': sorted(set(codes)),
        'code_examples': course_examples,
        'code_counts': dict(Counter(codes))
    }
    
    with open('requirement_codes_analysis.json', 'w') as f:
        json.dump(analysis_output, f, indent=2)
    
    print(f"\n✓ Saved detailed analysis to requirement_codes_analysis.json")
    return analysis_output

def find_code_patterns(json_file):
    """Look for patterns in requirement codes by department"""
    print(f"\nLoading {json_file}...")
    with open(json_file, 'r') as f:
        courses_data = json.load(f)
    
    # Group by department
    dept_codes = defaultdict(list)
    dept_courses = defaultdict(list)
    
    for term, courses in courses_data.items():
        for course in courses:
            req_code = course.get('RequirementCode')
            course_name = course.get('CourseName', '')
            
            if course_name:
                dept = course_name.split('-')[0] if '-' in course_name else 'UNKNOWN'
                
                if req_code:
                    dept_codes[dept].append(req_code)
                    dept_courses[dept].append({
                        'course': course_name,
                        'code': req_code,
                        'title': course.get('CourseTitle', '')
                    })
    
    # Analyze patterns
    print(f"\n{'='*70}")
    print(f"REQUIREMENT CODES BY DEPARTMENT")
    print(f"{'='*70}\n")
    print(f"{'Department':<12} {'Courses':<10} {'With Codes':<12} {'Unique Codes':<15} {'Code Range'}")
    print(f"{'-'*70}")
    
    dept_analysis = {}
    
    for dept in sorted(dept_codes.keys()):
        codes = [int(c) for c in dept_codes[dept] if str(c).isdigit()]
        unique_codes = set(dept_codes[dept])
        total_dept_courses = sum(1 for term, courses in courses_data.items() 
                                for course in courses 
                                if course.get('CourseName', '').startswith(dept + '-'))
        
        if codes:
            code_range = f"{min(codes)} - {max(codes)}"
        else:
            code_range = "N/A"
        
        print(f"{dept:<12} {total_dept_courses:<10} {len(dept_codes[dept]):<12} {len(unique_codes):<15} {code_range}")
        
        dept_analysis[dept] = {
            'total_courses': total_dept_courses,
            'courses_with_codes': len(dept_codes[dept]),
            'unique_codes': sorted(list(unique_codes)),
            'code_range': code_range,
            'examples': dept_courses[dept][:5]  # First 5 examples
        }
    
    # Save department analysis
    with open('requirement_codes_by_department.json', 'w') as f:
        json.dump(dept_analysis, f, indent=2)
    
    print(f"\n✓ Saved department analysis to requirement_codes_by_department.json")
    return dept_analysis

def create_placeholder_prerequisites(json_file, output_file):
    """Add human-readable placeholder for prerequisites"""
    print(f"\nLoading {json_file}...")
    with open(json_file, 'r') as f:
        courses_data = json.load(f)
    
    courses_modified = 0
    
    for term, courses in courses_data.items():
        for course in courses:
            req_code = course.get('RequirementCode')
            if req_code:
                # Placeholder message
                course['PrerequisiteDisplay'] = f"Prerequisites required (Code: {req_code})"
                course['HasPrerequisites'] = True
                courses_modified += 1
            else:
                course['PrerequisiteDisplay'] = "No prerequisites"
                course['HasPrerequisites'] = False
    
    with open(output_file, 'w') as f:
        json.dump(courses_data, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ Added prerequisite flags to {courses_modified} courses")
    print(f"✓ Saved to {output_file}")
    print(f"{'='*70}")

def generate_code_list_for_email(json_file):
    """Generate a clean list of codes to include in email"""
    with open(json_file, 'r') as f:
        courses_data = json.load(f)
    
    codes = set()
    for term, courses in courses_data.items():
        for course in courses:
            req_code = course.get('RequirementCode')
            if req_code:
                codes.add(req_code)
    
    sorted_codes = sorted([int(c) for c in codes if str(c).isdigit()])
    
    print(f"\n{'='*70}")
    print("SAMPLE REQUIREMENT CODES FOR EMAIL")
    print(f"{'='*70}")
    print(f"\nWe have {len(codes)} unique requirement codes in our dataset.")
    print(f"Examples include: {', '.join(map(str, sorted_codes[:20]))}")
    if len(sorted_codes) > 20:
        print(f"... and {len(sorted_codes) - 20} more")
    print(f"\nCode range: {min(sorted_codes)} - {max(sorted_codes)}")
    print(f"{'='*70}\n")

def full_analysis(json_file):
    """Run all analysis functions"""
    print(f"\n{'#'*70}")
    print(f"# HAMILTON COURSE REQUIREMENT CODES - FULL ANALYSIS")
    print(f"{'#'*70}\n")
    
    # Run all analyses
    analyze_requirement_codes(json_file)
    find_code_patterns(json_file)
    generate_code_list_for_email(json_file)
    
    # Create placeholder version
    create_placeholder_prerequisites(json_file, 'courses_with_prereq_flags.json')
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print("\nGenerated files:")
    print("  1. requirement_codes_analysis.json - Full code breakdown")
    print("  2. requirement_codes_by_department.json - Department patterns")
    print("  3. courses_with_prereq_flags.json - Original data + prereq flags")
    print("\nNext steps:")
    print("  - Review the analysis files above")
    print("  - Wait for Ms. Farrell's response with the official code mapping")
    print("  - Use courses_with_prereq_flags.json in HamPlan with placeholder text")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    import sys
    
    json_file = 'courses_all_terms.json'
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    
    # Check if file exists
    try:
        with open(json_file, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"Error: Could not find {json_file}")
        print("Usage: python analyze_requirements.py [json_file]")
        sys.exit(1)
    
    # Run full analysis
    full_analysis(json_file)