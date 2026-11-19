
import os
import re
import pymupdf as fitz
import faiss
import gradio as gr
import numpy as np
from pathlib import Path
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

def add_code_warning(answer):
    code_blocks = re.findall(
        r"```[^\n]*\n(.*?)```",
        answer,
        flags=re.DOTALL
    )

    arabic_pattern = re.compile(r"[\u0600-\u06FF]")

    has_arabic_code = any(
        arabic_pattern.search(code)
        for code in code_blocks
    )

    warning = (
        "\n\n⚠️ **تنبيه:** يحتوي أحد الأمثلة البرمجية "
        "على نص عربي قد يتأثر باتجاه الكتابة عند "
        "استخراجه من PDF. يُرجى مطابقته بالمصدر "
        "الأصلي قبل التنفيذ."
    )

    if has_arabic_code and warning not in answer:
        return answer + warning
