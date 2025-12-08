import json
import time
from scraper import scrape_minor_or_major
from pathlib import Path

BASE_URL = "https://hamilton.smartcatalogiq.com/en"


def extract_concentration_and_minor_urls(data, urls=None):
    """Recursively extract all URLs that contain 'Concentration' or 'Minor' in their names."""
    if urls is None:
        urls = []

    # Check if current item has a Name field
    if isinstance(data, dict):
        name = data.get("Name", "")
        path = data.get("Path", "")

        # Check if this is a concentration or minor
        if path and ("Concentration" in name or "Minor" in name):
            # Convert path to proper URL format: lowercase and remove leading slash
            url = BASE_URL + path.lower()
            urls.append({"name": name, "path": path, "url": url})

        # Recursively check children
        if "Children" in data:
            for child in data["Children"]:
                extract_concentration_and_minor_urls(child, urls)

    elif isinstance(data, list):
        for item in data:
            extract_concentration_and_minor_urls(item, urls)

    return urls


def scrape_all_requirements(urls_json_path, output_path):
    """Scrape all concentration and minor requirements and save to JSON."""

    # Load the URLs JSON file
    print(f"Loading URLs from {urls_json_path}...")
    with open(urls_json_path, "r", encoding="utf-8") as f:
        urls_data = json.load(f)

    # Extract all concentration and minor URLs
    print("Extracting concentration and minor URLs...")
    target_urls = extract_concentration_and_minor_urls(urls_data)

    print(f"\nFound {len(target_urls)} concentrations and minors to scrape:")
    for item in target_urls:
        print(f"  - {item['name']}")

    # Scrape each URL
    all_requirements = []
    successful = 0
    failed = 0

    print(f"\n{'='*80}")
    print("Starting scraping process...")
    print(f"{'='*80}\n")

    for idx, item in enumerate(target_urls, 1):
        print(f"[{idx}/{len(target_urls)}] Scraping: {item['name']}")
        print(f"  URL: {item['url']}")

        try:
            # Scrape the requirements
            requirements_data = scrape_minor_or_major(item["url"])

            # Add metadata
            requirements_data["program_name"] = item["name"]
            requirements_data["program_path"] = item["path"]
            requirements_data["program_url"] = item["url"]

            all_requirements.append(requirements_data)
            successful += 1
            print(f"  ✓ Successfully scraped\n")

        except Exception as e:
            print(f"  ✗ Error scraping: {e}\n")
            failed += 1
            # Add a placeholder with error info
            all_requirements.append(
                {
                    "program_name": item["name"],
                    "program_path": item["path"],
                    "program_url": item["url"],
                    "error": str(e),
                    "requirements": [],
                }
            )

        # Be respectful with requests - add a delay
        if idx < len(target_urls):  # Don't delay after the last one
            time.sleep(1)

    # Save results to JSON
    print(f"\n{'='*80}")
    print(f"Scraping complete!")
    print(f"{'='*80}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(target_urls)}")
    print(f"\nSaving results to {output_path}...")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_requirements, f, indent=2, ensure_ascii=False)

    print(f"✓ Successfully saved {len(all_requirements)} programs to {output_path}")

    # Print summary statistics
    print(f"\n{'='*80}")
    print("Summary Statistics:")
    print(f"{'='*80}")

    total_courses = 0
    total_sections = 0

    for program in all_requirements:
        if "error" not in program:
            sections = len(program.get("requirements", []))
            total_sections += sections

            for section in program.get("requirements", []):
                total_courses += len(section.get("courses", []))

    print(f"Total programs scraped: {len(all_requirements)}")
    print(f"Total requirement sections: {total_sections}")
    print(f"Total course entries: {total_courses}")
    print(f"{'='*80}\n")


def main():
    # Set up paths
    script_dir = Path(__file__).parent
    urls_json_path = script_dir / "urls.json"
    output_path = script_dir.parent / "requirements_data.json"

    # Run the scraper
    scrape_all_requirements(urls_json_path, output_path)


if __name__ == "__main__":
    main()
