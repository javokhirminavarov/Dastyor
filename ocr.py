import os
import streamlit as st
import json
from PIL import Image, ImageStat
import io
import fitz  # PyMuPDF for PDF handling
from typing import List
import re
import base64
import pandas as pd
import requests
from mistralai import Mistral

# Mistral API Token (keep default fallback for convenience and backward compatibility)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "v1kUkHNjeKVXTd1kmegHHj2NZhwxUoWC")
MISTRAL_TEXT_MODEL = "ministral-3b-latest"

# Page configuration
st.set_page_config(
    page_title="OCR Text Extraction",
    page_icon="📄",
    layout="wide"
)

# Default prompts for different document types (hidden from user)
DEFAULT_PROMPTS = {
    "Invoice": """Analyze this invoice document and extract the following key fields:
- Invoice Number
- Invoice Date
- Due Date
- Vendor/Seller Name
- Vendor Address
- Buyer/Customer Name
- Buyer Address
- Line Items (Product/Service, Quantity, Unit Price, Total)
- Subtotal
- Tax Amount
- Tax Rate
- Total Amount
- Payment Terms
- Currency

Return the extracted information in a structured JSON format. If any field is not found, mark it as "Not Found".""",

    "Certificate of Origin": """Analyze this Certificate of Origin document and extract the following key fields:
- Certificate Number
- Issue Date
- Exporter Name
- Exporter Address
- Consignee Name
- Consignee Address
- Country of Origin
- Country of Destination
- Invoice Number
- Invoice Date
- Certification Statement
- Issuing Authority
- Stamp/Seal Information
- Commodities: return an array named "commodities" (always include at least one item). Each
  commodity should include Description of Goods, HS Code/Tariff Number, Quantity, Weight
  (Gross/Net), Number of Packages, and Marks and Numbers.

Return the extracted information in a structured JSON format with a top-level "commodities"
array whenever goods are present so commodity-level data is never lost. If any field is not
found, mark it as "Not Found".""",

    "Custom": ""
}

def process_pdf_to_images(pdf_bytes: bytes) -> List[dict]:
    """Convert PDF pages to images and split obvious two-up scans."""
    images: list[dict] = []
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        # Render page to image at higher resolution
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        aspect_ratio = pix.width / float(pix.height)
        midpoint = pix.width // 2

        # Look for a bright, low-text gutter around the center to avoid
        # splitting ordinary landscape pages that lack a clear divider.
        gutter_width = max(10, int(pix.width * 0.06))
        gray_page = image.convert("L")
        center_band = gray_page.crop(
            (
                midpoint - gutter_width // 2,
                0,
                midpoint + gutter_width // 2,
                pix.height,
            )
        )
        page_stats = ImageStat.Stat(gray_page)
        center_stats = ImageStat.Stat(center_band)
        center_blankish = (
            center_stats.mean[0] > page_stats.mean[0] + 5
            and center_stats.stddev[0] < 25
        )

        should_split = (
            pix.width > pix.height
            and aspect_ratio > 1.3
            and center_blankish
        )

        if should_split:
            left_img = image.crop((0, 0, midpoint, pix.height))
            right_img = image.crop((midpoint, 0, pix.width, pix.height))
            images.append({"image": left_img, "page_number": page_num + 1, "part": 1})
            images.append({"image": right_img, "page_number": page_num + 1, "part": 2})
        else:
            images.append({"image": image, "page_number": page_num + 1, "part": None})

    pdf_document.close()
    return images

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def structure_text_with_mistral_llm(text: str, prompt: str, api_key: str) -> str:
    """Use Mistral's chat completions to structure OCR text with a JSON-focused reply."""

    if not api_key:
        raise ValueError("Mistral API key is required for text structuring.")

    client = Mistral(api_key=api_key)

    full_prompt = (
        f"{prompt}\n\nExtracted Text:\n{text}\n\n"
        "Respond with ONLY valid JSON for the structured output."
    )

    try:
        response = client.chat.complete(
            model=MISTRAL_TEXT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that turns OCR text into concise JSON. Always keep "
                        "line-item or commodity tables as an array named 'commodities' with one "
                        "object per item so that granular data is preserved."
                    ),
                },
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.1,
            max_tokens=512,
        )

        if not response.choices:
            raise ValueError("No completion returned by Mistral.")

        content = response.choices[0].message.content or ""
        return content.strip() or text

    except Exception as mistral_error:
        st.warning(
            "Mistral structuring failed. Showing raw OCR text instead. "
            f"({mistral_error})"
        )
        return text


def extract_with_mistral(image: Image.Image, mistral_api_key: str) -> str:
    """Use Mistral's official OCR model."""
    if not mistral_api_key:
        raise ValueError("Mistral API key is required for OCR.")

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    headers = {
        "Authorization": f"Bearer {mistral_api_key}",
        "Content-Type": "application/json",
    }

    # The OCR API expects a structured document object. Sending only a string
    # (e.g., "data:image/png;base64,...") produces a 422 validation error
    # similar to "DocumentURLChunk" complaints. Providing the type and
    # payload field keeps the request compatible with the latest API version.
    payload = {
        "model": "mistral-ocr-latest",
        "document": {
            "type": "image_url",
            "image_url": f"data:image/png;base64,{img_b64}",
        },
        "include_image_base64": False,
    }

    response = requests.post(
        "https://api.mistral.ai/v1/ocr",
        headers=headers,
        json=payload,
        timeout=60,
    )

    if response.status_code != 200:
        raise Exception(f"Mistral API error {response.status_code}: {response.text}")

    result = response.json()

    # Support both canonical {"text": "..."} and chat-style formats for forward compatibility
    if isinstance(result, dict):
        if "text" in result:
            return str(result.get("text", "")).strip()
        if "content" in result:
            return str(result.get("content", "")).strip()
        if "message" in result and isinstance(result["message"], dict):
            content = result["message"].get("content", "")
            return content if isinstance(content, str) else str(content)

    return str(result).strip()


def extract_json_from_text(text: str) -> dict:
    """Try to extract JSON from text response"""
    try:
        # Try to find JSON in code blocks
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            json_str = text.split("```")[1].split("```")[0].strip()
        else:
            # Try to find JSON object
            json_match = re.search(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = text

        return json.loads(json_str)
    except:
        return None


def split_structured_output(json_data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split structured JSON into general info and commodity tables."""

    general_rows: list[dict] = []
    commodity_rows: list[dict] = []

    def _is_commodity_list(value) -> bool:
        return isinstance(value, list) and value and all(isinstance(item, dict) for item in value)

    def _flatten_general(value, parent_key=""):
        if isinstance(value, dict):
            for key, val in value.items():
                new_key = f"{parent_key}.{key}" if parent_key else key
                if _is_commodity_list(val):
                    for idx, item in enumerate(val, start=1):
                        row = {"Item": idx}
                        for k, v in item.items():
                            row[k] = v
                        commodity_rows.append(row)
                else:
                    _flatten_general(val, new_key)
        elif isinstance(value, list):
            # Non-commodity lists are stored as comma-separated values
            joined = ", ".join([str(v) for v in value]) if value else ""
            general_rows.append({"Field": parent_key or "Value", "Value": joined})
        else:
            general_rows.append({"Field": parent_key or "Value", "Value": value})

    _flatten_general(json_data)

    general_df = pd.DataFrame(general_rows) if general_rows else pd.DataFrame(columns=["Field", "Value"])
    commodity_df = pd.DataFrame(commodity_rows) if commodity_rows else pd.DataFrame()
    return general_df, commodity_df


def make_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe safe for Streamlit/Arrow serialization."""

    if df is None or df.empty:
        return df.copy()

    safe_df = df.copy()

    def _stringify(value):
        if pd.isna(value):
            return ""
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        return str(value)

    return safe_df.applymap(_stringify)


def extract_text_with_api(
    file_bytes: bytes,
    file_type: str,
    prompt: str,
):
    """Extract text using Mistral OCR and structure it with the ministral text model."""
    try:
        # Convert file to image(s)
        if file_type == "application/pdf":
            images = process_pdf_to_images(file_bytes)
        else:
            image = Image.open(io.BytesIO(file_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            images = [{"image": image, "page_number": 1, "part": None}]

        all_structured_outputs = []
        all_raw_outputs = []
        page_contents = []

        # Process each page
        for i, page_info in enumerate(images):
            img = page_info["image"]
            page_label = f"Page {page_info['page_number']}"
            if page_info.get("part"):
                page_label += f" (part {page_info['part']})"

            raw = extract_with_mistral(img, MISTRAL_API_KEY)
            structured = raw
            if prompt.strip():
                structured = structure_text_with_mistral_llm(
                    raw,
                    prompt,
                    MISTRAL_API_KEY,
                )

            all_structured_outputs.append(f"--- {page_label} ---\n{structured}\n")
            all_raw_outputs.append(f"--- {page_label} ---\n{raw}\n")
            page_contents.append({
                "page_number": page_info["page_number"],
                "part": page_info.get("part"),
                "structured": structured,
                "raw": raw
            })

        # Combine all pages
        combined_structured = "\n".join(all_structured_outputs)
        combined_raw = "\n".join(all_raw_outputs)

        return combined_structured, combined_raw, page_contents

    except Exception as e:
        return f"Error: {str(e)}", None, None

def main():
    st.title("📄 OCR Text Extraction")
    st.markdown("Extract structured data from documents using **AI Models**")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Document")

        # Document type selection
        doc_type = st.selectbox(
            "Document Type",
            ["Invoice", "Certificate of Origin"],
            help="Select the type of document to extract data from"
        )

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF or Image file",
            type=["pdf", "png", "jpg", "jpeg"],
            help="Upload a document to extract text from"
        )

        # Extract button
        extract_button = st.button("🚀 Extract Text", type="primary", use_container_width=True)
        
        if uploaded_file:
            file_type = uploaded_file.type

            # Display uploaded file
            if file_type == "application/pdf":
                st.info(f"📄 PDF uploaded: {uploaded_file.name}")
                # Show first page preview
                pdf_bytes = uploaded_file.read()
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                if len(pdf_document) > 0:
                    page = pdf_document[0]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_bytes = pix.tobytes("png")
                    st.image(img_bytes, caption="First Page Preview", use_column_width=True)
                uploaded_file.seek(0)  # Reset file pointer
            else:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)


    with col2:
        st.header("📊 Extraction Results")

        # Results section
        if extract_button:
            if not uploaded_file:
                st.error("⚠️ Please upload a document first")
                return
            if not MISTRAL_API_KEY:
                st.error(
                    "A Mistral API key is required for OCR. Set the MISTRAL_API_KEY environment "
                    "variable before running extraction."
                )
                return

            with st.spinner("🔄 Running an OCR model and structuring the output..."):
                try:
                    # Read file bytes
                    uploaded_file.seek(0)
                    file_bytes = uploaded_file.read()
                    file_type = uploaded_file.type

                    # Get the appropriate prompt based on document type
                    extraction_prompt = DEFAULT_PROMPTS[doc_type]

                    # Process with cloud API
                    extracted_data, raw_text, page_contents = extract_text_with_api(
                        file_bytes,
                        file_type,
                        extraction_prompt,
                    )

                    if raw_text and not extracted_data.startswith("Error:"):
                        st.markdown("**Extracted Structured Data:**")

                        if extraction_prompt.strip():
                            json_result = extract_json_from_text(extracted_data)

                            if json_result:
                                general_df, commodity_df = split_structured_output(json_result)
                                general_df_safe = make_arrow_compatible(general_df)
                                commodity_df_safe = make_arrow_compatible(commodity_df)

                                st.markdown("**General Information:**")
                                st.data_editor(
                                    general_df_safe,
                                    use_container_width=True,
                                    key="general_info_editor",
                                )

                                st.markdown("**Commodity Details:**")
                                if not commodity_df.empty:
                                    st.data_editor(
                                        commodity_df_safe,
                                        use_container_width=True,
                                        num_rows="dynamic",
                                        key="commodity_editor",
                                    )
                                else:
                                    st.info("No commodity-level data was detected in the structured output.")
                            else:
                                st.error("Unable to parse structured data into tables.")
                        else:
                            st.error("No extraction prompt provided to structure the data.")

                        st.success("✅ Extraction completed successfully!")
                    else:
                        st.error(extracted_data)

                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
