#!/usr/bin/env python3
"""
Scrape prerequisites for all courses from Hamilton College's course catalog.
Extracts credentials from a HAR file and fetches prerequisites for each section.
"""

import json
import requests
import time
from pathlib import Path


def load_credentials_from_har(har_path: str) -> dict:
    """Extract cookies and headers from HAR file."""
    with open(har_path, "r") as f:
        har_data = json.load(f)

    # Find the SectionDetails request in the HAR file
    for entry in har_data["log"]["entries"]:
        if "SectionDetails" in entry["request"]["url"]:
            # Extract cookies as a dict
            cookies = {}
            for cookie in entry["request"]["cookies"]:
                cookies[cookie["name"]] = cookie["value"]

            # Extract relevant headers
            headers = {}
            for header in entry["request"]["headers"]:
                name = header["name"]
                # Only include necessary headers
                if name in [
                    "Content-Type",
                    "Accept",
                    "X-Requested-With",
                    "__IsGuestUser",
                    "__RequestVerificationToken",
                ]:
                    headers[name] = header["value"]

            return {"cookies": cookies, "headers": headers}

    raise ValueError("No SectionDetails request found in HAR file")


def fetch_section_details(section_id: str, cookies: dict, headers: dict) -> dict:
    """Fetch section details for a given section ID."""
    url = "https://collss-prod.hamilton.edu/Student/Student/Courses/SectionDetails"

    payload = {"sectionId": str(section_id), "studentId": None}

    response = requests.post(url, json=payload, cookies=cookies, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"HTTP {response.status_code}", "section_id": section_id}


def load_courses(courses_path: str) -> list:
    """Load courses from JSON file."""
    with open(courses_path, "r") as f:
        return json.load(f)


def extract_prerequisites(
    courses_path: str, har_path: str, output_path: str, delay: float = 0.5
):
    """
    Extract prerequisites for all courses.

    Args:
        courses_path: Path to courses JSON file
        har_path: Path to HAR file with credentials
        output_path: Path to save prerequisites JSON
        delay: Delay between requests in seconds
    """
    # Load credentials
    print("Loading credentials from HAR file...")
    creds = load_credentials_from_har(har_path)

    # Load courses
    print("Loading courses...")
    courses = load_courses(courses_path)

    # Extract unique section IDs with course info
    sections = {}
    for course in courses:
        section_id = course.get("Id")
        if section_id:
            sections[section_id] = {
                "section_id": section_id,
                "course_name": course.get("CourseName", ""),
                "section_name": course.get("SectionNameDisplay", ""),
                "title": course.get("Title", ""),
                "course_id": course.get("CourseId", ""),
            }

    print(f"Found {len(sections)} unique sections to fetch")

    # Fetch prerequisites for each section
    prerequisites_data = []
    failed_sections = []

    for i, (section_id, section_info) in enumerate(sections.items(), 1):
        print(
            f"[{i}/{len(sections)}] Fetching {section_info['section_name']}...", end=" "
        )

        try:
            details = fetch_section_details(
                section_id, creds["cookies"], creds["headers"]
            )

            if "error" in details:
                print(f"ERROR: {details['error']}")
                failed_sections.append(section_info)
                continue

            requisites = details.get("RequisiteItems", [])

            prereq_entry = {
                "section_id": section_id,
                "course_name": section_info["course_name"],
                "section_name": section_info["section_name"],
                "title": section_info["title"],
                "course_id": section_info["course_id"],
                "prerequisites": requisites,
            }

            prerequisites_data.append(prereq_entry)

            prereq_count = len(requisites)
            print(f"OK ({prereq_count} prerequisite{'s' if prereq_count != 1 else ''})")

        except Exception as e:
            print(f"ERROR: {str(e)}")
            failed_sections.append(section_info)

        # Rate limiting
        time.sleep(delay)

    # Save results
    print(f"\nSaving {len(prerequisites_data)} courses to {output_path}...")

    output = {
        "metadata": {
            "total_courses": len(prerequisites_data),
            "courses_with_prerequisites": sum(
                1 for p in prerequisites_data if p["prerequisites"]
            ),
            "failed_sections": len(failed_sections),
            "source_file": courses_path,
            "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "prerequisites": prerequisites_data,
        "failed_sections": failed_sections,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone!")
    print(f"  Total courses: {len(prerequisites_data)}")
    print(
        f"  Courses with prerequisites: {output['metadata']['courses_with_prerequisites']}"
    )
    print(f"  Failed sections: {len(failed_sections)}")


if __name__ == "__main__":
    # Paths
    courses_path = "courses_fall_2025.json"
    har_path = "collss-prod.hamilton.edu copy.har"
    output_path = "prerequisites.json"

    extract_prerequisites(courses_path, har_path, output_path)
