import re
import fitz  # PyMuPDF
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np

from table_parser import TableAwareResumeParser

# ----------------- Contact extraction regex -----------------
EMAIL_REGEX = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
PHONE_REGEX = r"(?:(?:\+?\d{1,3}[-.\s]?)?(?:\d{3}|\(\d{3}\))[-.\s]?\d{3}[-.\s]?\d{4})"

def extract_email(text: str):
    emails = re.findall(EMAIL_REGEX, text)
    return emails[0] if emails else None

def extract_phone(text: str):
    phones = re.findall(PHONE_REGEX, text)
    return phones[0] if phones else None


# ========== LOAD AI MODELS ==========
_EMBED_MODEL = None

def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


# ========= SKILL DICTIONARY =========
SKILL_KEYWORDS = [
    "python", "pandas", "numpy", "sql", "tensorflow",
    "pytorch", "scikit-learn", "power bi", "excel",
    "machine learning", "deep learning", "computer vision"
]

_table_parser = TableAwareResumeParser()

# ========== TEXT EXTRACTION (SMART) ==========
def extract_text_from_pdf(pdf_path: str) -> str:

    try:
        doc = fitz.open(pdf_path)
        text_parts = []

        for page in doc:
            content = page.get_text()
            if content:
                text_parts.append(content)

        text = "\n".join(text_parts).lower()

        # 🔥 Fallback trigger
        if len(text.strip()) < 100:
            raise Exception("Weak extraction")

        return text

    except Exception:
        print("Using Table-Aware OCR Parser...")
        ocr_text = _table_parser.parse(pdf_path)
        return ocr_text.lower()


# ========= SKILL EXTRACTION ==========
def extract_skills_keyword(text: str):
    found = []
    for skill in SKILL_KEYWORDS:
        if skill in text:
            found.append(skill)
    return found


# ========= SEMANTIC MATCH ==========
def get_semantic_score(text, jd):
    model = _get_embed_model()
    emb = model.encode([text, jd], convert_to_numpy=True)
    score = cosine_similarity([emb[0]], [emb[1]])[0][0]
    return round(float(score) * 100, 2)


# ========= MAIN PARSER ==========
def ai_parse_and_match(pdf_path: str, jd: str):

    jd_lower = jd.lower()
    text = extract_text_from_pdf(pdf_path)

    resume_skills = extract_skills_keyword(text)
    jd_skills = extract_skills_keyword(jd_lower)

    matched_skills = list(set(resume_skills).intersection(set(jd_skills)))

    skill_match_percent = (
        round(100 * len(matched_skills) / len(jd_skills), 2)
        if jd_skills else 0.0
    )

    semantic_score = get_semantic_score(text, jd_lower)

    return {
        "resume_path": pdf_path,
        "semantic_score": semantic_score,
        "skill_match_percent": skill_match_percent,
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "matched_skills": matched_skills,
        "email": extract_email(text),
        "phone": extract_phone(text),
        "raw_text": text
    }
