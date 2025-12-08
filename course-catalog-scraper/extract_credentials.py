import json
import sys


def extract_credentials_from_har(har_file_path):
    """Extract cookies and request verification token from HAR file."""

    print(f"Reading HAR file: {har_file_path}")

    with open(har_file_path, "r", encoding="utf-8") as f:
        har_data = json.load(f)

    # Find the PostSearchCriteria request
    entries = har_data.get("log", {}).get("entries", [])

    for entry in entries:
        request = entry.get("request", {})
        url = request.get("url", "")

        if "PostSearchCriteria" in url:
            print(f"Found PostSearchCriteria request!")

            # Extract cookies
            cookies = {}
            for cookie in request.get("cookies", []):
                name = cookie.get("name")
                value = cookie.get("value")
                if name and value:
                    cookies[name] = value

            # Extract __RequestVerificationToken from headers
            verification_token = None
            for header in request.get("headers", []):
                if header.get("name") == "__RequestVerificationToken":
                    verification_token = header.get("value")
                    break

            # Print results
            print("\n" + "=" * 80)
            print("COOKIES:")
            print("=" * 80)
            for name, value in cookies.items():
                print(f'    "{name}": "{value}",')

            print("\n" + "=" * 80)
            print("REQUEST VERIFICATION TOKEN:")
            print("=" * 80)
            if verification_token:
                print(f'    "{verification_token}"')
            else:
                print("    Token not found!")

            print("\n" + "=" * 80)
            print("INSTRUCTIONS:")
            print("=" * 80)
            print("1. Open scrape_courses.py")
            print("2. Replace the COOKIES dictionary with the values above")
            print(
                "3. Replace the __RequestVerificationToken in HEADERS with the token above"
            )
            print("4. Run: python scrape_courses.py")
            print("=" * 80)

            return cookies, verification_token

    print("\nERROR: Could not find PostSearchCriteria request in HAR file")
    return None, None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        har_file = "collss-prod.hamilton.edu.har"
        print(f"No HAR file specified, using default: {har_file}")
    else:
        har_file = sys.argv[1]

    extract_credentials_from_har(har_file)
