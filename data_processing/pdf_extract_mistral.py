from mistralai import Mistral
from pathlib import Path
from mistralai import DocumentURLChunk
from mistralai.models import OCRResponse
from dotenv import load_dotenv
import os
from data_processing.s3_utils import upload_file_to_s3
import logging
from uuid import uuid4

# load environment variables
load_dotenv()

api_key = os.getenv("MISTRAL_KEY")
client = Mistral(api_key)


def process_pdf_mistral(file_content: bytes, file_name: str) -> dict:
    """
    Process a PDF file using Mistral OCR, extract markdown content and images,
    and upload them to S3 with a structured naming format.

    Args:
        file_content (bytes): The content of the uploaded PDF file.
        file_name (str): The original name of the uploaded file.

    Returns:
        dict: A dictionary with S3 URLs for the markdown file, extracted images, and status information.
    """
    logging.basicConfig(level=logging.DEBUG)

    try:
        logging.debug("Starting the PDF processing function using Mistral.")

        # Step 1: Validate the PDF file
        if not file_content.startswith(b"%PDF"):
            logging.error("The uploaded file is not a valid PDF.")
            raise ValueError("The provided file is not a valid PDF.")
        logging.debug("PDF file content validated.")

        # Step 2: Create a cleaned-up filename for S3 storage
        pdf_filename = Path(file_name).stem.replace(" ", "_").upper()
        pdf_filename = pdf_filename.split('_')[0].upper()
        logging.debug(f"Structured S3 filename: {pdf_filename}")

        # Step 3: Upload the PDF to Mistral for OCR processing
        temp_pdf_path = Path(f"temp_{uuid4().hex[:8]}.pdf")
        with open(temp_pdf_path, "wb") as temp_file:
            temp_file.write(file_content)
        logging.debug(f"Temporary PDF saved to {temp_pdf_path}.")

        uploaded_file = client.files.upload(
            file={
                "file_name": temp_pdf_path.name,
                "content": temp_pdf_path.read_bytes(),
            },
            purpose="ocr",
        )
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        logging.debug(f"PDF uploaded to Mistral. Signed URL obtained.")

        # Step 4: Process PDF with OCR, including embedded images
        pdf_response = client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=True,
        )
        logging.debug("OCR processing completed successfully.")

        # Step 5: Extract and upload images to S3
        image_s3_urls = []
        image_dir = Path(f"{pdf_filename}_images")
        os.makedirs(image_dir, exist_ok=True)
        
        picture_counter = 0
        for page in pdf_response.pages:
            for img in page.images:
                picture_counter += 1
                temp_image_path = image_dir / f"{pdf_filename}-image-{picture_counter}.png"
                with open(temp_image_path, "wb") as fp:
                    fp.write(img.image_base64.encode())
                logging.debug(f"Image saved temporarily: {temp_image_path}")

                # Upload image to S3 under {pdf_filename}/images/
                image_s3_url = upload_file_to_s3(
                    file_path=str(temp_image_path),
                    source=f"{pdf_filename}/images",
                    metadata={
                        "file_type": "image",
                        "original_filename": file_name,
                    },
                )
                image_s3_urls.append(image_s3_url)
                logging.debug(f"Image uploaded to S3: {image_s3_url}")

                # Clean up temporary image files
                os.remove(temp_image_path)
                logging.debug(f"Temporary image file deleted: {temp_image_path}")

        # Step 6: Save and upload Markdown content to S3
        def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
            for img_name, base64_str in images_dict.items():
                markdown_str = markdown_str.replace(
                    f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})"
                )
            return markdown_str

        def get_combined_markdown(ocr_response: OCRResponse) -> str:
            markdowns = []
            for page in ocr_response.pages:
                image_data = {img.id: img.image_base64 for img in page.images}
                markdowns.append(replace_images_in_markdown(page.markdown, image_data))
            return "\n\n".join(markdowns)

        temp_markdown_path = Path(f"{pdf_filename}_with_images.md")
        with open(temp_markdown_path, "w") as markdown_file:
            markdown_file.write(get_combined_markdown(pdf_response))
        logging.debug(f"Markdown content saved temporarily: {temp_markdown_path}")

        # Upload Markdown to S3 under {pdf_filename}/markdown/
        markdown_s3_url = upload_file_to_s3(
            file_path=str(temp_markdown_path),
            source=f"{pdf_filename}/markdown",
            metadata={
                "file_type": "markdown",
                "original_filename": file_name,
            },
        )
        logging.debug(f"Markdown uploaded to S3: {markdown_s3_url}")

        # Clean up temporary files
        if temp_pdf_path.exists():
            logging.debug(f"Deleting temporary PDF file: {temp_pdf_path}")
            os.remove(temp_pdf_path)
            logging.debug(f"Temporary PDF file deleted: {temp_pdf_path}")
        if temp_markdown_path.exists():
            logging.debug(f"Deleting temporary Markdown file: {temp_markdown_path}")
            os.remove(temp_markdown_path)
            logging.debug(f"Temporary Markdown file deleted: {temp_markdown_path}")
        if image_dir.exists() and len(list(image_dir.iterdir())) == 0:
            logging.debug(f"Deleting empty image directory: {image_dir}")
            os.rmdir(image_dir)
            logging.debug(f"Empty image directory deleted: {image_dir}")

        logging.debug("Temporary files cleaned up.")

        # Step 7: Return success response
        logging.debug("PDF processing completed successfully.")
        
        return {
            "markdown_s3_url": markdown_s3_url,
            "image_s3_urls": image_s3_urls,
            "pdf_filename": pdf_filename,
            "status": "success",
            "message": "PDF processed and uploaded to S3 successfully",
        }

    except Exception as e:
        logging.error(f"Error processing PDF using Mistral: {e}", exc_info=True)
        raise RuntimeError(f"Error processing PDF using Mistral: {str(e)}")

