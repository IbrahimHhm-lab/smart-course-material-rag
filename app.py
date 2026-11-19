
import os
import re
import fitz
import faiss
import gradio as gr
import numpy as np

from functools import lru_cache
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


# ============================================================
# 1. إعدادات المشروع
# ============================================================

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

GROQ_MODEL = "openai/gpt-oss-120b"

MAX_PAGES = 100
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_QUESTIONS = 5

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250

TOP_K = 8
SIMILARITY_THRESHOLD = 0.35

# التصحيح الرئيسي: زيادة حد التوليد
MAX_OUTPUT_TOKENS = 1500

ABSTAIN = (
    "لا تتوفر معلومات كافية في المادة "
    "المسترجعة للإجابة بدقة."
)


# ============================================================
# 2. تحميل نموذج Embeddings مرة واحدة
# ============================================================

@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(MODEL_NAME)


# ============================================================
# 3. تنظيف النصوص المستخرجة من PDF
# ============================================================

def clean_text(text):
    if not text:
        return ""

    # إزالة المحارف غير المرئية الشائعة
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    # توحيد المسافات
    text = re.sub(r"[ \t]+", " ", text)

    # تقليل الأسطر الفارغة المتكررة
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# 4. استخراج النصوص من ملف PDF
# ============================================================

def extract_pdf_pages(file_path):
    pages = []

    with fitz.open(file_path) as pdf:

        if len(pdf) > MAX_PAGES:
            raise ValueError(
                "الحد الأقصى المسموح هو 100 صفحة."
            )

        for page_number, page in enumerate(pdf, 1):

            # محاولة ترتيب النص حسب موضعه على الصفحة
            text = page.get_text(
                "text",
                sort=True
            )
