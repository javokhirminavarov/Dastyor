# B5ixCHcvfnjtRtinPvVBEgXsCwf6fuvy

import streamlit as st
import base64
from mistralai import Mistral
import json
from PIL import Image
import fitz  # PyMuPDF for PDF handling
import pandas as pd
import google.generativeai as genai

# Replace the placeholders below with your actual API keys before running locally.
MISTRAL_OCR_API_KEY = "YOUR_MISTRAL_OCR_API_KEY"
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
GEMINI_MODEL = "gemini-1.5-flash"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Page configuration
st.set_page_config(
    page_title="OCR Text Extraction System",
    page_icon="📄",
    layout="wide"
)

# Default prompts for different document types
DEFAULT_PROMPTS = {
    "Invoice": """Extract the following key fields from this invoice document:
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

    "Certificate of Origin": """Extract the following key fields from this Certificate of Origin document:
- Certificate Number
- Issue Date
- Exporter Name
- Exporter Address
- Consignee Name
- Consignee Address
- Country of Origin
- Country of Destination
- Description of Goods
- HS Code/Tariff Number
- Quantity
- Weight (Gross/Net)
- Number of Packages
- Marks and Numbers
- Invoice Number
- Invoice Date
- Certification Statement
- Issuing Authority
- Stamp/Seal Information

Return the extracted information in a structured JSON format. If any field is not found, mark it as "Not Found".""",

    "Custom": ""
}

def extract_text_with_mistral_ocr(file_bytes, file_type):
    """Call Mistral OCR API to extract text from PDF or image."""
    try:
        if not MISTRAL_OCR_API_KEY:
            raise ValueError("Mistral OCR API key is not configured. Update MISTRAL_OCR_API_KEY at the top of the file.")

        client = Mistral(api_key=MISTRAL_OCR_API_KEY)
        
        # Encode file to base64
        base64_content = base64.b64encode(file_bytes).decode('utf-8')
        
        # Determine the MIME type and create data URI
        if file_type == "application/pdf":
            data_uri = f"data:application/pdf;base64,{base64_content}"
            doc_type = "document_url"
        elif file_type == "image/jpeg" or file_type == "image/jpg":
            data_uri = f"data:image/jpeg;base64,{base64_content}"
            doc_type = "image_url"
        elif file_type == "image/png":
            data_uri = f"data:image/png;base64,{base64_content}"
            doc_type = "image_url"
        else:
            # Default to image/jpeg for other image types
            data_uri = f"data:image/jpeg;base64,{base64_content}"
            doc_type = "image_url"
        
        # Call Mistral OCR API
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": doc_type,
                doc_type: data_uri
            },
            include_image_base64=True
        )
        
        # Extract markdown content from all pages
        full_text = ""
        page_contents = []
        
        for page in ocr_response.pages:
            page_text = f"--- Page {page.index + 1} ---\n{page.markdown}\n"
            full_text += page_text
            page_contents.append({
                "page_number": page.index + 1,
                "markdown": page.markdown,
                "images": page.images if hasattr(page, 'images') else []
            })
        
        return full_text, page_contents
    except Exception as e:
        return f"Error: {str(e)}", None


def format_text_with_gemini(prompt: str, raw_text: str):
    """Use Gemini to convert raw OCR text into structured JSON."""
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key is not configured. Update GEMINI_API_KEY at the top of the file.")

    system_prompt = (
        "You are a meticulous data extraction assistant. "
        "Return only valid JSON that captures the information requested. "
        "If a field is missing, set its value to 'Not Found'."
    )

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content([
        system_prompt,
        f"Extraction instructions:\n{prompt.strip()}\n\nDocument Content:\n{raw_text}",
    ])

    if not response.text:
        raise RuntimeError("Gemini did not return any content.")

    return response.text.strip()


def display_structured_data_table(data):
    """Render structured data as a table for easier review."""
    if isinstance(data, dict):
        rows = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            rows.append({"Field": key, "Value": value})
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
    elif isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame({"Value": data})
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Structured data could not be converted into a table.")

def main():
    st.title("📄 OCR Text Extraction System")
    st.markdown("Extract structured data from invoices, certificates of origin, and other documents using Mistral OCR and Gemini")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.info("**OCR Model**: mistral-ocr-latest")
        st.info(f"**Formatter Model**: {GEMINI_MODEL}")
        st.caption("API keys are configured directly in the source code.")
        
        st.divider()
        
        # Document type selection
        doc_type = st.selectbox(
            "Document Type",
            ["Invoice", "Certificate of Origin", "Custom"],
            help="Select the type of document to extract data from"
        )
        
        st.divider()
        st.markdown("### 📝 Extraction Prompt")
        st.markdown("Customize the prompt to control what data is extracted")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Document")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF or Image file",
            type=["pdf", "png", "jpg", "jpeg"],
            help="Upload a document to extract text from"
        )
        
        if uploaded_file:
            file_type = uploaded_file.type
            
            # Display uploaded file
            if file_type == "application/pdf":
                st.info(f"📄 PDF uploaded: {uploaded_file.name}")
                # Show first page preview using PyMuPDF
                pdf_bytes = uploaded_file.read()
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                if len(pdf_document) > 0:
                    page = pdf_document[0]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_bytes = pix.tobytes("png")
                    st.image(img_bytes, caption="First Page Preview", use_container_width=True)
                uploaded_file.seek(0)  # Reset file pointer
            else:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.header("✏️ Customizable Prompt")
        
        # Prompt text area with default based on document type
        extraction_prompt = st.text_area(
            "Edit the extraction prompt below:",
            value=DEFAULT_PROMPTS[doc_type],
            height=400,
            help="Modify this prompt to change what fields are extracted. Leave empty to see raw OCR output."
        )
        
        # Extract button
        extract_button = st.button("🚀 Extract Text", type="primary", use_container_width=True)
    
    # Results section
    if extract_button:
        if not uploaded_file:
            st.error("⚠️ Please upload a document first")
            return
        
        st.divider()
        st.header("📊 Extraction Results")
        
        with st.spinner("🔄 Processing document with Mistral OCR..."):
            try:
                # Read file bytes
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                file_type = uploaded_file.type
                
                # Process with Mistral OCR
                raw_ocr, page_contents = extract_text_with_mistral_ocr(
                    file_bytes,
                    file_type,
                )

                if isinstance(raw_ocr, str) and raw_ocr.startswith("Error:"):
                    st.error(raw_ocr)
                    return

                structured_output = None
                if extraction_prompt.strip():
                    structured_output = format_text_with_gemini(extraction_prompt, raw_ocr)

                tab1, tab2 = st.tabs(["📋 Structured Data", "📄 Raw OCR Text"])

                with tab1:
                    st.markdown("**Extracted Structured Data:**")

                    if extraction_prompt.strip() and structured_output:
                        try:
                            if "```json" in structured_output:
                                json_str = structured_output.split("```json")[1].split("```")[0].strip()
                            elif "```" in structured_output:
                                json_str = structured_output.split("```")[1].split("```")[0].strip()
                            else:
                                json_str = structured_output

                            json_result = json.loads(json_str)
                            st.json(json_result)
                            display_structured_data_table(json_result)

                            st.download_button(
                                label="⬇️ Download JSON",
                                data=json.dumps(json_result, indent=2),
                                file_name="extracted_data.json",
                                mime="application/json"
                            )
                        except Exception:
                            st.code(structured_output, language="text")
                            st.download_button(
                                label="⬇️ Download Text",
                                data=structured_output,
                                file_name="extracted_data.txt",
                                mime="text/plain"
                            )
                    else:
                        st.info("No extraction prompt provided. Showing raw OCR output.")
                        st.markdown(raw_ocr)

                with tab2:
                    st.markdown("**Raw OCR Output (Markdown):**")

                    if page_contents:
                        for page_info in page_contents:
                            with st.expander(f"📄 Page {page_info['page_number']}", expanded=True):
                                st.markdown(page_info['markdown'])

                                if page_info['images']:
                                    st.markdown(f"**Images found: {len(page_info['images'])}**")
                    else:
                        st.text_area(
                            "OCR Text",
                            value=raw_ocr,
                            height=400,
                            disabled=True
                        )

                    st.download_button(
                        label="⬇️ Download Raw OCR",
                        data=raw_ocr,
                        file_name="ocr_output.md",
                        mime="text/markdown"
                    )

                st.success("✅ Extraction completed successfully!")
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    st.divider()
    st.header("📥 Load Excel Data")
    excel_file = st.file_uploader(
        "Upload an Excel file to preview",
        type=["xlsx", "xls"],
        key="excel_loader",
        help="Use this to quickly inspect spreadsheets that contain structured information."
    )
    if st.button("📂 Load Excel", use_container_width=True):
        if excel_file:
            excel_file.seek(0)
            df = pd.read_excel(excel_file)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("Please upload an Excel file first.")

    # Footer with instructions
    st.divider()
    with st.expander("ℹ️ How to use this application"):
        st.markdown("""
        ### Steps:
        1. **Configure API Keys**: Edit `ocr.py` to add your Mistral OCR and Gemini API keys
        2. **Select Document Type**: Choose between Invoice, Certificate of Origin, or Custom
        3. **Upload Document**: Upload a PDF or image file (PNG, JPG, JPEG)
        4. **Customize Prompt**: Edit the extraction prompt to specify what fields you want to extract
           - Leave the prompt empty to see raw OCR output only
        5. **Extract**: Click the "Extract Text" button to process the document

        ### How it works:
        - **Step 1**: Document is processed using `mistral-ocr-latest` to extract all text in markdown format
        - **Step 2**: Extracted text is sent to Gemini with your custom prompt to structure the data
        - **Result**: You get both structured data (JSON) and raw OCR text
        
        ### Features:
        - ✅ Support for PDF and image files (PNG, JPG, JPEG)
        - ✅ Multi-page PDF processing
        - ✅ Customizable extraction prompts
        - ✅ Pre-configured templates for common document types
        - ✅ Both structured (JSON) and raw (Markdown) outputs
        - ✅ Download extracted data
        - ✅ High accuracy OCR powered by Mistral AI
        - ✅ Image extraction from documents
        
        ### Tips:
        - For better results, ensure your document is clear and well-lit
        - Customize the prompt to match your specific document format
        - Leave the prompt empty if you just want raw OCR text extraction
        - The OCR model supports multiple languages and complex layouts
        
        ### Pricing:
        - Mistral OCR: $1 per 1,000 pages (~$0.001 per page)
        - More affordable with batch processing
        
        ### Example Usage:
        The app uses Mistral's official API format:
        ```python
        client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",  # or "document_url" for PDFs
                "image_url": "data:image/jpeg;base64,{base64_string}"
            },
            include_image_base64=True
        )
        ```
        """)

if __name__ == "__main__":
    main()