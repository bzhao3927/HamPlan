# Hamilton College Course Scraper

This Python script scrapes all courses from Hamilton College's course registration system for Fall 2025.

## Files

1. **`scrape_courses.py`** - Main script that fetches all courses using pagination
2. **`extract_credentials.py`** - Helper script to extract cookies and tokens from HAR files
3. **`courses_fall_2025.json`** - Output file containing all 824 courses (6.1MB)

## How It Works

The script:
- Uses the Hamilton College API endpoint for course search
- Authenticates using cookies and verification token from your HAR file
- Fetches courses page by page (30 courses per page)
- Automatically stops when all pages are retrieved
- Saves all courses to a JSON file

## Results

Successfully retrieved:
- **824 courses** from **28 pages**
- Saved to `courses_fall_2025.json`
- Includes complete course information (term, description, faculty, schedule, etc.)

## Usage

### To re-run the scraper with fresh credentials:

1. Export a new HAR file from your browser while viewing the course search page
2. Run the extraction script:
   ```bash
   python extract_credentials.py your-new-file.har
   ```
3. Copy the cookies and token to `scrape_courses.py`
4. Run the scraper:
   ```bash
   python scrape_courses.py
   ```

## Notes

- The script respects the server with a 0.5-second delay between requests
- Configured for Fall 2025 (`"terms": ["25/FA"]`) - don't change this
- Cookies and tokens expire, so you'll need to update them periodically
- The script handles pagination automatically until all courses are fetched

## Dependencies

```bash
pip install requests
```

## Data Structure

The JSON file contains an array of course objects, each with:
- Term information
- Course details (ID, subject, number, credits)
- Section information
- Faculty/instructor data
- Meeting times and locations
- Capacity and enrollment data
- And much more...

