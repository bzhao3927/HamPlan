#!/usr/bin/env python3
"""
Scrape department overview pages from Hamilton catalog AND main site
"""

import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time
import json

CATALOG_BASE = "https://hamilton.smartcatalogiq.com"
MAIN_SITE_BASE = "https://www.hamilton.edu"

# Catalog departments
CATALOG_DEPARTMENTS = {
    "africana-studies": "africana-studies-overview",
    "american-indian-and-indigenous-studies": "american-indian-and-indigenous-studies-overview",
    "american-studies": "american-studies-overview",
    "anthropology": "anthropology-overview",
    "art": "art-overview",
    "art-history": "art-history-overview",
    "asian-studies": "asian-studies-overview",
    "biochemistry-molecular-biology": "biochemistrymolecular-biology-overview",
    "biology": "biology-overview",
    "chemical-physics": "chemical-physics-overview",
    "chemistry": "chemistry-overview",
    "cinema-and-media-studies": "cinema-and-media-studies-overview",
    "classics": "classics-overview",
    "computer-science": "computer-science-overview",
    "dance-and-movement-studies": "dance-and-movement-studies-overview",
    "data-science": "data-science-data-science-overview",
    "east-asian-languages-and-literatures": "east-asian-languages-and-literatures-overview",
    "economics": "economics-overview",
    "environmental-studies": "environmental-studies-overview",
    "french-and-francophone-studies": "french-and-francophone-studies-overview",
    "geoarchaeology": "geoarchaeology-overview",
    "geosciences": "geosciences-overview",
    "german-russian-italian-and-arabic": "german-russian-italian-and-arabic-overview",
    "government": "government-overview",
    "hispanic-studies": "hispanic-studies-overview",
    "history": "history-overview",
    "jurisprudence-law-and-justice-studies": "jurisprudence-law-and-justice-studies-overview",
    "linguistics": "linguistics-overview",
    "literature-and-creative-writing": "literature-and-creative-writing-overview",
    "mathematics-and-statistics": "mathematics-and-statistics-overview",
    "middle-east-and-islamicate-worlds-studies": "middle-east-and-islamicate-worlds-studies-overview",
    "music": "music-overview",
    "neuroscience": "neuroscience-overview",
    "philosophy": "philosophy-overview",
    "physics": "physics-overview",
    "psychology": "psychology-overview",
    "public-policy": "public-policy-overview",
    "religious-studies": "religious-studies-overview",
    "sociology": "sociology-overview",
    "theatre": "theatre-overview",
    "women-s-and-gender-studies": "womens-and-gender-studies-overview"
}

# Main site departments (additional or supplementary info)
MAIN_SITE_DEPARTMENTS = {
    "biochemistry-molecular-biology": "/academics/departments/biochemistry-molecular-biology",
    "africana-studies": "/academics/departments/africana-studies",
    "anthropology": "/academics/departments/anthropology",
    "art": "/academics/departments/art",
    "art-history": "/academics/departments/art-history",
    "biology": "/academics/departments/biology",
    "chemistry": "/academics/departments/chemistry",
    "classics": "/academics/departments/classics",
    "computer-science": "/academics/departments/computer-science",
    "economics": "/academics/departments/economics",
    "environmental-studies": "/academics/departments/environmental-studies",
    "geosciences": "/academics/departments/geosciences",
    "government": "/academics/departments/government",
    "history": "/academics/departments/history",
    "mathematics": "/academics/departments/mathematics",
    "music": "/academics/departments/music",
    "neuroscience": "/academics/departments/neuroscience",
    "philosophy": "/academics/departments/philosophy",
    "physics": "/academics/departments/physics",
    "psychology": "/academics/departments/psychology",
    "sociology": "/academics/departments/sociology",
    "theatre": "/academics/departments/theatre"
}

def scrape_catalog_overview(dept_name, overview_slug):
    """Scrape the catalog overview page for a department."""
    url = f"{CATALOG_BASE}/current/college-catalogue/academicprograms/{dept_name}/{overview_slug}"
    
    print(f"📖 Catalog: {dept_name}")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the main content
        content = soup.find('div', {'id': 'main'})
        if not content:
            content = soup.find('div', {'class': 'content'})
        if not content:
            content = soup.find('main')
        if not content:
            content = soup.find('article')
        
        if content:
            # Remove navigation, script, and style elements
            for element in content(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
            
            # Remove side navigation
            for nav in content.find_all(['div', 'ul'], class_=lambda x: x and ('nav' in x.lower() or 'menu' in x.lower() or 'sidebar' in x.lower())):
                nav.decompose()
            
            text = content.get_text(separator='\n', strip=True)
            
            # Clean up extra whitespace
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
            # Skip if too short
            if len(text) < 300:
                print(f"  ⚠️  Content too short ({len(text)} chars)")
                return None
            
            return {
                'department': dept_name,
                'url': url,
                'source': 'catalog',
                'text': text,
                'char_count': len(text)
            }
        else:
            print(f"  ⚠️  No content found")
            return None
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def scrape_main_site_overview(dept_name, dept_path):
    """Scrape the main Hamilton site for a department."""
    url = f"{MAIN_SITE_BASE}{dept_path}"
    
    print(f"🌐 Main site: {dept_name}")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find main content area
        content = soup.find('div', {'class': 'main-content'})
        if not content:
            content = soup.find('main')
        if not content:
            content = soup.find('article')
        if not content:
            content = soup.find('div', {'id': 'content'})
        
        if content:
            # Remove navigation, script, and style elements
            for element in content(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            text = content.get_text(separator='\n', strip=True)
            
            # Clean up
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
            if len(text) < 200:
                print(f"  ⚠️  Content too short ({len(text)} chars)")
                return None
            
            return {
                'department': dept_name,
                'url': url,
                'source': 'main_site',
                'text': text,
                'char_count': len(text)
            }
        else:
            print(f"  ⚠️  No content found")
            return None
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def main():
    print("="*70)
    print("📚 HAMILTON DEPARTMENT OVERVIEW SCRAPER")
    print("   (Catalog + Main Site)")
    print("="*70 + "\n")
    
    # Create output directories
    output_dir = Path("department_overviews")
    output_dir.mkdir(exist_ok=True)
    
    txt_dir = output_dir / "txt"
    txt_dir.mkdir(exist_ok=True)
    
    results = []
    successful = 0
    failed = 0
    
    # Scrape catalog
    print("\n📖 SCRAPING CATALOG OVERVIEWS\n")
    for dept_name, overview_slug in CATALOG_DEPARTMENTS.items():
        result = scrape_catalog_overview(dept_name, overview_slug)
        
        if result:
            results.append(result)
            successful += 1
            
            # Save individual text file
            txt_file = txt_dir / f"{dept_name}-catalog.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(f"Department: {dept_name}\n")
                f.write(f"Source: Catalog\n")
                f.write(f"URL: {result['url']}\n")
                f.write("="*70 + "\n\n")
                f.write(result['text'])
            
            print(f"  ✅ Saved {result['char_count']:,} chars to {txt_file.name}")
        else:
            failed += 1
        
        time.sleep(1)
    
    # Scrape main site
    print("\n🌐 SCRAPING MAIN SITE DEPARTMENTS\n")
    for dept_name, dept_path in MAIN_SITE_DEPARTMENTS.items():
        result = scrape_main_site_overview(dept_name, dept_path)
        
        if result:
            results.append(result)
            successful += 1
            
            # Save individual text file
            txt_file = txt_dir / f"{dept_name}-mainsite.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(f"Department: {dept_name}\n")
                f.write(f"Source: Main Site\n")
                f.write(f"URL: {result['url']}\n")
                f.write("="*70 + "\n\n")
                f.write(result['text'])
            
            print(f"  ✅ Saved {result['char_count']:,} chars to {txt_file.name}")
        else:
            failed += 1
        
        time.sleep(1)
    
    # Save combined JSON
    json_file = output_dir / "all_departments.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print(f"✅ Successfully scraped: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Saved to: {output_dir}")
    print(f"   - Individual files: {txt_dir}")
    print(f"   - Combined JSON: {json_file}")
    print("="*70)

if __name__ == "__main__":
    main()