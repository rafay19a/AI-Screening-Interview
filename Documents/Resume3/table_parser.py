import cv2
import numpy as np
import os
import streamlit as st

# --------------------------------------------------
# CLOUD-SAFE OCR GUARD (CRITICAL)
# --------------------------------------------------
OCR_AVAILABLE = True

try:
    import pytesseract
    from pdf2image import convert_from_path
except Exception:
    OCR_AVAILABLE = False


# ✅ Optional Windows safety (local only)
if OCR_AVAILABLE and os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class TableAwareResumeParser:

    def detect_tables(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

        table_mask = detect_horizontal + detect_vertical

        contours, _ = cv2.findContours(
            table_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        tables = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 200 and h > 200:
                tables.append((x, y, w, h))

        return tables

    def mask_tables(self, image, tables):

        masked = image.copy()

        for (x, y, w, h) in tables:
            cv2.rectangle(masked, (x, y), (x+w, y+h), (255, 255, 255), -1)

        return masked

    def parse(self, pdf_path):

        # --------------------------------------------------
        # 🔥 CLOUD SAFETY EXIT (CRITICAL)
        # --------------------------------------------------
        if not OCR_AVAILABLE:
            st.warning("OCR engine not available in this environment.")
            return ""

        try:
            pages = convert_from_path(pdf_path)
        except Exception as e:
            st.warning("Poppler not available. OCR disabled.")
            return ""

        all_text = ""

        for page in pages:

            image = np.array(page)

            tables = self.detect_tables(image)
            masked_image = self.mask_tables(image, tables)

            gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            _, thresh = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            text = pytesseract.image_to_string(
                thresh,
                config="--oem 3 --psm 6"
            )

            all_text += "\n\n" + text

        return all_text
