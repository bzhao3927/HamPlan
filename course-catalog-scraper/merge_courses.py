#!/usr/bin/env python3
"""Merge courses_fall_2025.json with prerequisites.json into a single file."""

import json

def main():
    # Load courses
    with open('courses_fall_2025.json', 'r') as f:
        courses = json.load(f)
    
    # Load prerequisites
    with open('prerequisites.json', 'r') as f:
        prereq_data = json.load(f)
    
    # Build a mapping from section_id to prerequisites
    prereq_map = {}
    for item in prereq_data['prerequisites']:
        section_id = item['section_id']
        prereq_map[section_id] = item['prerequisites']
    
    # Merge prerequisites into courses
    courses_with_prereqs = 0
    for course in courses:
        section_id = course.get('Id')
        if section_id and section_id in prereq_map:
            course['Prerequisites'] = prereq_map[section_id]
            if prereq_map[section_id]:
                courses_with_prereqs += 1
        else:
            course['Prerequisites'] = []
    
    # Save merged file
    with open('courses_with_prerequisites.json', 'w') as f:
        json.dump(courses, f, indent=2)
    
    print(f"Merged {len(courses)} courses")
    print(f"Courses with prerequisites: {courses_with_prereqs}")
    print(f"Saved to courses_with_prerequisites.json")

if __name__ == '__main__':
    main()

