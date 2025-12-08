import requests
import json
import time
import re

# API endpoint
API_URL = "https://collss-prod.hamilton.edu/Student/Student/Courses/PostSearchCriteria"

# NEW COOKIES - Just updated!
COOKIES = {
    '_mkto_trk': 'id:533-KFT-589&token:_mch-hamilton.edu-5582b0636953fe0c6cf7b2ec8cb4da3a',
    'nmstat': '5745dba2-8997-ba10-a835-2f9388c6cef8',
    '_ga_DXDZSJ3KHT': 'GS1.2.1746022989.1.0.1746022989.0.0.0',
    'ADMISSIONINTEREST': '1',
    '_gcl_au': '1.1.1316622451.1756750505',
    '_ga_7JY7T788PK': 'GS2.1.s1757792308$o4$g1$t1757792497$j60$l0$h0',
    'sc_is_visitor_unique': 'rx5933137.1763216678.B07CFBFC450C489886C80DB5C7F180B5.1.1.1.1.1.1.1.1.1',
    'HAMSAML': '887F75FF-B3CC-4792-B55B2E611B402029',
    'HAMPORTAL': 'DB6AC77A-174E-45DC-B592F107DF10A002',
    '_gid': 'GA1.2.1642332481.1763392747',
    '.ColleagueSelfServiceAntiforgery': 'CfDJ8B0GSlrVWN9Dn-qVRI-h8-2Sb925JyAfzbwWZMoZ4ee5YEK7WT3nswn36WmbUKTYFimU8I3NAl6QT0mE8aNn1fspXE4NZqUaJybHSW7vieLKy1hVrm4zGnh9m3FXUZsgU4-YjXiSAclImsS5wDQ9pdk',
    'student_selfservice_productionDN6_AuthMethod': 'SAML',
    '_ga': 'GA1.2.213407989.1745166879',
    '_ga_JSE9MJMZKJ': 'GS2.1.s1763660532$o72$g1$t1763661570$j60$l0$h0',
    '_ga_Y29PC3P5S9': 'GS2.1.s1763660532$o72$g1$t1763661570$j60$l0$h0',
    '_ga_R8W4VZ6D6E': 'GS2.1.s1763660532$o14$g1$t1763661570$j60$l0$h0',
    '_ga_V80MCZNM7H': 'GS2.1.s1763660533$o283$g1$t1763661570$j60$l0$h0',
    'student_selfservice_productionDN6': 'CfDJ8B0GSlrVWN9Dn-qVRI-h8-3UMCewbs-6iIVp6isOMbGbKSGM8WCbLf0fY7cAkD762xfxo604KEUBk2BD911mqsUs68fZlp5vZvo2n3pLzKG-qd9wb-70Ipk-h4Ytcm87E7wzgNZLOSBwXXa4omphSXPaiIFEqPSQJdJHeDsGPvPp1CTSqmmO7rgmYRhO1jC7k5XaebbD1Eh8R8oyAR2sORJ94ItE7Cwa6--flRFzUzjL9iYw5z-Yu2EfKxzUuHOVumgDno3mp4B8FHvH_Eak7VlF7RDhldv9RlKKGgJ6QgYnCDe14rMp7kg30M0E5N5POpuEVX35Y-5O-qY0wvO0Vk9PNCCSZ_v9oRwX-pwduearaV7pJIomoM-UFe1ir0AmKPLowU_GcTD5ttztdTQHEFe3VL2aR1TnHtgzLJXLjZ8bmzG6WZyfkRWWhhsj8Czs_wghX-8h-veRQ1W33OuQ4FWcOXe_YcvFa2Tlorho-2XgQPOixdm7AqhoYnNx9oQcs5C8DNGO99THuNgQWRLhuJLn69jMnnIzImAWPb4yi61QC-PNV4pAP524ai1QV2AT5q8mJh_XkddmPr06v-cGFoFwhBuyx_32tu_efyl6wFrjLuvaPfLLlyMAIVGsiy4jSmDEtj_2-eL23PxC8Y048LRXtiXWpr0h3hJMFW7eNVlgTWPCnbrw9JQFAdu7Yy3a0lL0Gx1jGEbZj_fMdSy4AwxIUAgO9SGh8lkP3tgOCBsZB8Hc6EjzSJphupjWuPRuPxxk8uS4-x8OzOPu76efnLe7Xf_Cl7WxihZArqHumwAo5V2m3YvINrpiwVCPuj59fuikgorJyXwgv9FigdePzKAbaa4_voEN_deGI028reeqyVetAt-Tg82KdBec2eQty2vTkQRGZzFkiNttQWisqzWbmwYrSF2QElLq3231p9-j0RGWFuzQKhrZKSumU1tckseCfolZZQr0PPGcyBvT3zVtZV1FEfLlTx0G5dYLBsx89SBCWdsqz7AQ8XXBFfLeYKMiA7OsD06dw23lqCLgIAJPmKhx4myxWKNQKLauHm6IjWmN_-Gf0yY6B71rIGrdDMePm-AvhEYCTc-36IVT3wu41bwHQ-i_wp9D5EH5QnNlfXygEHTUL5e70L4t3enn6h077gA3QIu8T-kcPlUtICz1kQo5olWsmPAt6_FRIWSwmO1pRZPBzIy9_XsAAYvMqO2A8yClSBLcq9Hq_LOEFydAOfQfFhW2j89TjyT4x3b4uldx1351k5fTmHaAd36g4mUf9XKEqB1wE7ZVqNT3UT7sRAStufOTKr7tQBVQ_Bmye36SNCC7kya49mOvp0qjFc5IPZwgPg8f-XK3oj4zVIeApYWnhScrZg72MTxXo2cpuwsT-ttg45qrBpOrzPtviGDJggberxgeKzigdIxW_G9kJRnG2axdpUXLwr7oO54cRbCpLuAqS2P7ef2SFq5SUC3XFjLnqM_-PJfC_jdsUAa66W2oX0fQTsORYsHT7IQgOi9JMMbY0dCarVm3eQVq49dPdqwcOkQzsmaUwdPPfvc',
    'student_selfservice_productionDN6_sfid': 'N47J81757G41',
    'student_selfservice_productionDN6_Sso': 'CfDJ8B0GSlrVWN9Dn%2BqVRI%2Bh8%2B3%2FPyPlRs2sEYEl93imMDJ396%2FPaD2bd17p1mBTHBA9UX9mBor%2Fary0XSiaaCWEw17Y5Ro5qYsQHJb%2F8TdO6ZSDdGdRNS8zUCyHIB6m3nU6%2FJ%2FTepn5laUBGSWqxaU5tp0ZN7TkDSd9SgFE18WRJ1KQ',
    'student_selfservice_productionDN6_ls': 'CfDJ8B0GSlrVWN9Dn-qVRI-h8-1c_MyqUQm5BcMZuN1qxwg5GXKxewfH70LDO3qv6rPplbIPRC1UITPz7CBbuW_YUN33mIzsk2mzes06_gxLHNPCxF5-za-zmEO1rStnxDwSMwW5XsQlyvefAcBycYNBezM',
    'student_selfservice_productionDN6_0': 'CfDJ8B0GSlrVWN9Dn-qVRI-h8-0Sm52Ri680xDzmB20bjButM6dCzzW3b2fizvAdKCO6PSQP2cd-S_mPfgQEso1GHOMtwY-pUJ5xwN0d2HYlonn07kJSFh1JX4iKHzkjRQY_gDbACFneuclfXiykhe11nKhBoHoxUxFdZL7PJMpREbIdNzoHtJEBEDUJRo9-Lf7Rp0uJCOFyfVfLeCV_GLjNmKwU4teTrtAzzdGEOMSYznGcZ-fU-c0jXM6-2BtdeI30oRv3UQqlc1heMtabxOtzPYdkgGHoGtAzCDJrfUnBjkfKjiP8YV8lmLHOAKY9jnEIxjelETqOTRw1u9mPmwvwM_osrvu_UOmQRfm4pC-EiDVW9aTIsKCx4sDZp20p-zmt-4ZQfdyohIgWw_mCG4Dq7opwnx7S-TwBmJQZ0zR-4Zh4Wv0KkxJmdXiqHxOnYgTPpPT-bmM7llhMn0RrdHc4vHDqrsUupS8XdtaPCtnfHh7D6MvTCCgwPfFmT-FjDBZp0e4bxapv_t5YupA0_yv0KhMef3G1jL7HoxSpKT2wPOrVniVzuUKmWDAmWzvStpxA3H4WHUtKzXTBn4spXrB01AvQc_hsLRMkThXuqRExTY5uoQKUiKtu8h1iPs1qZyPPzoTVP5CMWzuoNEtcFq8cshaVI1-iHaKu32jhkt0BWZ-e6u-3tsMawTB0-uZNH5Nzaoda9GaQph-GeQgtAmOvVoE',
}

# NEW HEADER TOKEN - Just updated!
HEADERS = {
    'Content-Type': 'application/json',
    '__requestverificationtoken': 'CfDJ8B0GSlrVWN9Dn-qVRI-h8-0tRuTMYpqd9zZn3ANrUsR0LKVrzf81V-G5aD0O7rtMnGFJ15IMHFaMpYIhHhoAUsrQZHGICZHfEL_9Rfau1p7HehTDq3xszv17ToZj8f79lLOMJbs49i0X0Of-Fumz77dK2adNpmH-IZZLQ4J5wQLN-wAo6n2DTmFEY3AV7Azeog',
    '__isguestuser': 'false',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
    'Referer': 'https://collss-prod.hamilton.edu/Student/Student/Courses/Search?keyword=',
    'Origin': 'https://collss-prod.hamilton.edu',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'X-Requested-With': 'XMLHttpRequest',
}

# Rest of your code stays the same...
AVAILABLE_TERMS = [
    "25/SP",
    "25/FA", 
    "26/SP",
]

REQUEST_BODY_TEMPLATE = {
    "keyword": None,
    "terms": [],
    "requirement": None,
    "subrequirement": None,
    "courseIds": None,
    "sectionIds": None,
    "requirementText": None,
    "subrequirementText": "",
    "group": None,
    "startTime": None,
    "endTime": None,
    "openSections": None,
    "subjects": [],
    "academicLevels": [],
    "courseLevels": [],
    "synonyms": [],
    "courseTypes": [],
    "topicCodes": [],
    "days": [],
    "locations": [],
    "faculty": [],
    "onlineCategories": None,
    "keywordComponents": [],
    "startDate": None,
    "endDate": None,
    "startsAtTime": None,
    "endsByTime": None,
    "pageNumber": 1,
    "sortOn": "SectionName",
    "sortDirection": "Ascending",
    "subjectsBadge": [],
    "locationsBadge": [],
    "termFiltersBadge": [],
    "daysBadge": [],
    "facultyBadge": [],
    "academicLevelsBadge": [],
    "courseLevelsBadge": [],
    "courseTypesBadge": [],
    "topicCodesBadge": [],
    "onlineCategoriesBadge": [],
    "openSectionsBadge": "",
    "openAndWaitlistedSectionsBadge": "",
    "subRequirementText": None,
    "quantityPerPage": 30,
    "openAndWaitlistedSections": None,
    "searchResultsView": "SectionListing",
}

def extract_course_codes_from_text(text):
    """Extract course codes from plain text like PHIL-115, AFRST-220"""
    if not text:
        return []
    pattern = r'\b([A-Z]{4,6}-\d{3})\b'
    courses = re.findall(pattern, text)
    return list(set(courses))


def extract_prerequisite_info(section):
    """Extract prerequisite information from a section"""
    prereq_info = {
        "has_prerequisites": False,
        "requirement_codes": [],  # CHANGED: Now an array
        "prerequisite_text": None,
        "prerequisite_courses": [],
        "display_text": "No prerequisites"
    }
    
    course = section.get("Course", {})
    
    # FIXED: Check Requisites array properly!
    requisites = course.get("Requisites", [])
    for req in requisites:
        # Look for prerequisites (CompletionOrder = "Previous")
        if req.get("IsRequired") and req.get("CompletionOrder") == "Previous":
            req_code = req.get("RequirementCode")
            if req_code:
                prereq_info["requirement_codes"].append(req_code)
                prereq_info["has_prerequisites"] = True
    
    # Search for prerequisite text in descriptions
    possible_fields = [
        ("Description", course.get("Description")),
        ("LongDescription", course.get("LongDescription")),
        ("Comments", course.get("Comments")),
        ("SectionComments", section.get("Comments")),
        ("PrerequisiteText", course.get("PrerequisiteText")),
        ("AdditionalInformation", course.get("AdditionalInformation")),
    ]
    
    for field_name, field_value in possible_fields:
        if not field_value or not isinstance(field_value, str):
            continue
            
        lower_value = field_value.lower()
        if "prerequisite" in lower_value or "prereq" in lower_value:
            prereq_info["prerequisite_text"] = field_value
            prereq_info["has_prerequisites"] = True
            
            # Extract course codes from text
            courses = extract_course_codes_from_text(field_value)
            if courses:
                prereq_info["prerequisite_courses"] = courses
            
            # Create display text
            display = field_value.strip()
            if len(display) > 200:
                display = display[:200] + "..."
            prereq_info["display_text"] = display
            break
    
    # Create placeholder display text if we have codes but no text
    if prereq_info["has_prerequisites"] and not prereq_info["prerequisite_text"]:
        if prereq_info["prerequisite_courses"]:
            prereq_info["display_text"] = f"Prerequisites: {', '.join(prereq_info['prerequisite_courses'])}"
        elif prereq_info["requirement_codes"]:
            codes = ", ".join(prereq_info["requirement_codes"])
            prereq_info["display_text"] = f"Prerequisites required (Codes: {codes})"
    
    return prereq_info


def fetch_page(term, page_number):
    """Fetch a single page"""
    request_body = REQUEST_BODY_TEMPLATE.copy()
    request_body["terms"] = [term]
    request_body["pageNumber"] = page_number

    print(f"  Fetching {term} page {page_number}...")

    try:
        response = requests.post(
            API_URL, headers=HEADERS, cookies=COOKIES, json=request_body, timeout=30
        )
        response.raise_for_status()
        data = response.json()

        sections = data.get("Sections", [])
        print(f"    Retrieved {len(sections)} courses")

        for section in sections:
            prereq_info = extract_prerequisite_info(section)
            section["PrerequisiteInfo"] = prereq_info

        return sections

    except requests.exceptions.RequestException as e:
        print(f"    Error: {e}")
        return None


def fetch_term_courses(term):
    """Fetch all courses for a term"""
    print(f"\nFetching courses for {term}...")
    term_courses = []
    page_number = 1

    while True:
        sections = fetch_page(term, page_number)

        if sections is None:
            break

        if len(sections) == 0:
            print(f"  No more courses for {term}.")
            break

        term_courses.extend(sections)

        if len(sections) < REQUEST_BODY_TEMPLATE["quantityPerPage"]:
            break

        page_number += 1
        time.sleep(0.5)

    return term_courses


def fetch_all_courses():
    """Fetch all terms"""
    all_courses = {}

    for term in AVAILABLE_TERMS:
        courses = fetch_term_courses(term)
        if courses:
            all_courses[term] = courses
        time.sleep(1)

    return all_courses


def print_prerequisite_summary(courses_by_term):
    """Print summary"""
    print("\n" + "="*80)
    print("PREREQUISITE EXTRACTION SUMMARY")
    print("="*80)
    
    total = 0
    with_codes = 0
    with_text = 0
    with_parsed = 0
    examples = []
    
    for term, courses in courses_by_term.items():
        for course in courses:
            total += 1
            prereq_info = course.get("PrerequisiteInfo", {})
            
            if prereq_info.get("has_prerequisites"):
                with_codes += 1
                
                # Collect examples
                if len(examples) < 10:
                    course_obj = course.get("Course", {})
                    examples.append({
                        "course": f"{course_obj.get('SubjectCode', 'UNK')}-{course_obj.get('Number', '000')}",
                        "title": course_obj.get("Title", "Unknown")[:40],
                        "codes": prereq_info.get("requirement_codes", []),
                        "text": prereq_info.get("prerequisite_text", "")[:80] if prereq_info.get("prerequisite_text") else None
                    })
            
            if prereq_info.get("prerequisite_text"):
                with_text += 1
            
            if prereq_info.get("prerequisite_courses"):
                with_parsed += 1
    
    print(f"Total courses: {total}")
    print(f"With prerequisites: {with_codes} ({100*with_codes/total:.1f}%)")
    print(f"With prerequisite TEXT: {with_text} ({100*with_text/total:.1f}%)")
    print(f"With parsed course codes: {with_parsed} ({100*with_parsed/total:.1f}%)")
    
    if examples:
        print("\n" + "-"*80)
        print("SAMPLE COURSES WITH PREREQUISITES:")
        print("-"*80)
        for ex in examples:
            print(f"\n{ex['course']}: {ex['title']}")
            if ex['codes']:
                print(f"  → Requirement Codes: {', '.join(ex['codes'])}")
            if ex['text']:
                print(f"  → Text: {ex['text']}...")
    
    print("="*80 + "\n")


def save_to_json(courses_by_term, filename="courses_with_prerequisites.json"):
    """Save to JSON"""
    total = sum(len(courses) for courses in courses_by_term.values())
    print(f"Saving {total} courses to {filename}...")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(courses_by_term, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {filename}\n")


def main():
    print("="*80)
    print("HAMILTON COURSE SCRAPER WITH PREREQUISITE EXTRACTION")
    print("="*80)
    print(f"Terms: {', '.join(AVAILABLE_TERMS)}\n")

    courses_by_term = fetch_all_courses()

    if courses_by_term:
        print("\n" + "="*80)
        print("COMPLETE")
        print("="*80)
        for term, courses in courses_by_term.items():
            print(f"  {term}: {len(courses)} courses")
        
        print_prerequisite_summary(courses_by_term)
        save_to_json(courses_by_term)
        
        print("✓ Done!")
    else:
        print("\n❌ No courses fetched - cookies expired")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()