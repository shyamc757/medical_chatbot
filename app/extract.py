import os
import re
from datetime import datetime
from typing import List, Dict

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def is_text_extracted_meaningful(text: str, threshold: int = 500, ratio: float = 0.25) -> bool:
    """
    Determine if extracted text is meaningful based on length and ratio of alphanumeric content.

    Args:
        text (str): Extracted text to evaluate.
        threshold (int): Minimum length to be considered meaningful.
        ratio (float): Minimum proportion of alphanumeric characters.

    Returns:
        bool: True if the text is meaningful, False if likely garbled.
    """
    alnum_chars = sum(c.isalnum() for c in text)
    return len(text.strip()) > threshold and (alnum_chars / len(text)) > ratio


def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """
    Enhance image quality for OCR via grayscale, contrast, and filtering.

    Args:
        img (Image.Image): PIL image.

    Returns:
        Image.Image: Enhanced image.
    """
    img = img.convert("L")
    img = img.filter(ImageFilter.MedianFilter())
    img = ImageEnhance.Contrast(img).enhance(2)
    return img


def extract_text_with_ocr_from_image(image_path: str) -> str:
    """
    Perform OCR on a medical image file using Tesseract.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Extracted text.
    """
    try:
        img = Image.open(image_path)
        img = preprocess_image_for_ocr(img)
        return pytesseract.image_to_string(img, config="--psm 6")
    except Exception as e:
        print(f"[ERROR] OCR failed on image {image_path}: {e}")
        return ""


def extract_text_with_ocr_from_pdf(pdf_path: str) -> str:
    """
    Perform OCR on a PDF file by converting pages to images and extracting text.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text via OCR.
    """
    try:
        images = convert_from_path(pdf_path, dpi=300)
        ocr_text = ""
        for img in images:
            processed = preprocess_image_for_ocr(img)
            ocr_text += pytesseract.image_to_string(processed, config="--psm 6")
        return ocr_text
    except Exception as e:
        print(f"[ERROR] OCR failed on PDF {pdf_path}: {e}")
        return ""


def extract_text_from_pdf(pdf_path: str, try_table_extraction: bool = False) -> str:
    """
    Extract text from a PDF using pdfplumber. Fallback to OCR if content is garbled.

    Args:
        pdf_path (str): Path to PDF file.
        try_table_extraction (bool): Whether to extract tables as structured rows.

    Returns:
        str: Cleaned extracted text.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            texts = []
            for page in pdf.pages:
                if try_table_extraction and page.extract_tables():
                    for table in page.extract_tables():
                        table_text = "\n".join([" | ".join([cell or "" for cell in row]) for row in table])
                        texts.append(table_text)
                else:
                    text = page.extract_text()
                    if text:
                        texts.append(text)
            text = "\n".join(texts)
    except Exception as e:
        print(f"[ERROR] PDFPlumber failed for {pdf_path}: {e}")
        text = ""

    if not is_text_extracted_meaningful(text):
        print(f"[OCR Fallback] Using OCR for: {os.path.basename(pdf_path)}")
        text = extract_text_with_ocr_from_pdf(pdf_path)

    return text


def safe_extract_value(line: str) -> str | None:
    """
    Extract a confident numeric value from a line, skipping known patterns
    like reference ranges (e.g., 3.5 - 7.2) and inequalities (<1.0, >2.5).
    """
    # Clean edge case patterns
    if re.search(r"[<>]=?\s*\d", line):
        return None

    # Discard line if it ONLY contains a reference range
    range_match = re.search(r"\b(\d{1,3}\.\d{1,2})\s*[-–—]\s*(\d{1,3}\.\d{1,2})\b", line)
    if range_match and len(re.findall(r"\d{1,3}\.\d{1,2}", line)) <= 2:
        return None

    # Extract the first confident float
    match = re.search(r"\b(\d{1,3}\.\d{1,2})\b", line)
    return match.group(1) if match else None


def parse_medical_fields(text: str) -> dict:
    """
    Parse medical values like Hemoglobin, WBC, etc. using keywords and safe regex.

    Args:
        text (str): Extracted raw text.

    Returns:
        dict: Extracted medical values.
    """
    results = {}
    lines = text.splitlines()

    fields = {
        "hemoglobin": ["hemoglobin"],
        "wbc_count": ["wbc"],
        "rbc_count": ["rbc"],
        "platelet_count": ["platelet"],
        "esr": ["esr"],
        "glucose": ["glucose"],
        "creatinine": ["creatinine"],
        "uric_acid": ["uric acid"],
        "vitamin_d": ["vitamin d"]
    }

    for line in lines:
        line_lower = line.lower()
        for key, keywords in fields.items():
            if any(k in line_lower for k in keywords):
                value = safe_extract_value(line_lower)
                if value and key not in results:
                    results[key] = value
    return results


def process_files(input_dir: str) -> List[Dict]:
    """
    Process PDFs and image files in a directory, extract relevant info, and save to JSON.

    Args:
        input_dir (str): Directory containing medical report files.

    Returns:
        List[Dict]: List of metadata and extracted content.
    """
    processed_docs = []

    for filename in os.listdir(input_dir):
        ext = os.path.splitext(filename)[-1].lower()
        file_path = os.path.join(input_dir, filename)

        if ext == ".pdf":
            print(f"\nProcessing PDF: {filename}")
            raw_text = extract_text_from_pdf(file_path, try_table_extraction=True)
        elif ext in SUPPORTED_IMAGE_EXTENSIONS:
            print(f"\nProcessing Image: {filename}")
            raw_text = extract_text_with_ocr_from_image(file_path)
        else:
            print(f"[SKIP] Unsupported file type: {filename}")
            continue

        parsed_fields = parse_medical_fields(raw_text)

        doc = {
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "raw_text": raw_text,
            "parsed_fields": parsed_fields,
        }

        processed_docs.append(doc)

    return processed_docs
