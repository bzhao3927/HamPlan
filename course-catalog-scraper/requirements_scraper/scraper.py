import requests
from bs4 import BeautifulSoup
import json


def scrape_minor_or_major(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    program_title = soup.select_one(".programTables h1").get_text(strip=True)

    output = {"program_title": program_title, "requirements": []}

    req_container = soup.select_one("#degreeRequirements")

    # All headings like: "Take 1 course from", "Complete", etc.
    sections = req_container.find_all(["h2", "h3"])

    for sec in sections:
        section_name = sec.get_text(strip=True)

        # Find the next table following the <h3>
        next_table = sec.find_next_sibling("table")

        section_data = {"section": section_name, "courses": []}

        if next_table:
            rows = next_table.find_all("tr")[1:]  # skip header row
            for tr in rows:
                tds = tr.find_all("td")

                if len(tds) < 3:
                    continue

                # Course number can contain cross-listings like "AFRST-224 / WMGST-224"
                course_number = tds[0].get_text(" ", strip=True)
                course_title = tds[1].get_text(" ", strip=True)
                credits = tds[2].get_text(strip=True)

                section_data["courses"].append(
                    {
                        "course_number": course_number,
                        "title": course_title,
                        "credits": credits,
                    }
                )

        output["requirements"].append(section_data)

    return output


# -------------------------
# Example usage (just plug in a Hamilton URL)
# -------------------------

if __name__ == "__main__":
    url = "https://www.hamilton.edu/academics/africana-studies/requirements"  # put your exact URL here

    data = scrape_minor_or_major(url)

    print(json.dumps(data, indent=2))
