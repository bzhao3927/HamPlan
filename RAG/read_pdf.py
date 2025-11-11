import PyPDF2

pdf_path = "/Users/CS/Documents/Deep Learning/Final Project/CPSCI-366_Syllabus.pdf"

# Open PDF
with open(pdf_path, "rb") as f:
    reader = PyPDF2.PdfReader(f)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

print(text[:1000])  # preview first 1000 characters
