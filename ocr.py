# B5ixCHcvfnjtRtinPvVBEgXsCwf6fuvy

import streamlit as st
import base64
from mistralai import Mistral
import json
from PIL import Image
import io
import fitz  # PyMuPDF for PDF handling

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

def extract_text_with_mistral_ocr(api_key, file_bytes, file_type, prompt):
    """Call Mistral OCR API to extract text from PDF or image"""
    try:
        client = Mistral(api_key=api_key)
        
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
        
        # If prompt is provided, use chat completion to extract structured data
        if prompt.strip():
            chat_response = client.chat.complete(
                model="mistral-large-latest",
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nDocument Content:\n{full_text}"
                    }
                ]
            )
            structured_data = chat_response.choices[0].message.content
            return structured_data, full_text, page_contents
        else:
            return full_text, full_text, page_contents
    
    except Exception as e:
        return f"Error: {str(e)}", None, None

def main():
    st.title("📄 OCR Text Extraction System")
    st.markdown("Extract structured data from invoices, certificates of origin, and other documents using Mistral's OCR API")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Mistral API Key",
            type="password",
            help="Enter your Mistral API key"
        )
        
        # Model display
        st.info("**OCR Model**: mistral-ocr-latest")
        st.info("**Chat Model**: mistral-large-latest")
        
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
        if not api_key:
            st.error("⚠️ Please enter your Mistral API key in the sidebar")
            return
        
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
                extracted_data, raw_ocr, page_contents = extract_text_with_mistral_ocr(
                    api_key, 
                    file_bytes, 
                    file_type, 
                    extraction_prompt
                )
                
                if raw_ocr and not extracted_data.startswith("Error:"):
                    # Create tabs for different views
                    tab1, tab2 = st.tabs(["📋 Structured Data", "📄 Raw OCR Text"])
                    
                    with tab1:
                        st.markdown("**Extracted Structured Data:**")
                        
                        if extraction_prompt.strip():
                            # Try to parse as JSON
                            try:
                                # Look for JSON in the response
                                if "```json" in extracted_data:
                                    json_str = extracted_data.split("```json")[1].split("```")[0].strip()
                                elif "```" in extracted_data:
                                    json_str = extracted_data.split("```")[1].split("```")[0].strip()
                                else:
                                    json_str = extracted_data
                                
                                json_result = json.loads(json_str)
                                st.json(json_result)
                                
                                # Download button for JSON
                                st.download_button(
                                    label="⬇️ Download JSON",
                                    data=json.dumps(json_result, indent=2),
                                    file_name="extracted_data.json",
                                    mime="application/json"
                                )
                            except:
                                st.code(extracted_data, language="text")
                                
                                # Download button for text
                                st.download_button(
                                    label="⬇️ Download Text",
                                    data=extracted_data,
                                    file_name="extracted_data.txt",
                                    mime="text/plain"
                                )
                        else:
                            st.info("No extraction prompt provided. Showing raw OCR output.")
                            st.markdown(extracted_data)
                    
                    with tab2:
                        st.markdown("**Raw OCR Output (Markdown):**")
                        
                        # Display page by page
                        if page_contents:
                            for page_info in page_contents:
                                with st.expander(f"📄 Page {page_info['page_number']}", expanded=True):
                                    st.markdown(page_info['markdown'])
                                    
                                    # Show extracted images if any
                                    if page_info['images']:
                                        st.markdown(f"**Images found: {len(page_info['images'])}**")
                        else:
                            st.text_area(
                                "OCR Text",
                                value=raw_ocr,
                                height=400,
                                disabled=True
                            )
                        
                        # Download button for raw OCR
                        st.download_button(
                            label="⬇️ Download Raw OCR",
                            data=raw_ocr,
                            file_name="ocr_output.md",
                            mime="text/markdown"
                        )
                    
                    st.success("✅ Extraction completed successfully!")
                else:
                    st.error(extracted_data)
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Footer with instructions
    st.divider()
    with st.expander("ℹ️ How to use this application"):
        st.markdown("""
        ### Steps:
        1. **Enter API Key**: Add your Mistral API key in the sidebar
        2. **Select Document Type**: Choose between Invoice, Certificate of Origin, or Custom
        3. **Upload Document**: Upload a PDF or image file (PNG, JPG, JPEG)
        4. **Customize Prompt**: Edit the extraction prompt to specify what fields you want to extract
           - Leave the prompt empty to see raw OCR output only
        5. **Extract**: Click the "Extract Text" button to process the document
        
        ### How it works:
        - **Step 1**: Document is processed using `mistral-ocr-latest` to extract all text in markdown format
        - **Step 2**: Extracted text is sent to `mistral-large-latest` with your custom prompt to structure the data
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