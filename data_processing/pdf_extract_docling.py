import os
from uuid import uuid4
import logging
from pathlib import Path
from docling_core.types.doc import ImageRefMode, PictureItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from data_processing.s3_utils import upload_file_to_s3



IMAGE_RESOLUTION_SCALE = 2.0

def process_pdf(file_content: bytes, file_name: str) -> dict:
    """
    Process a PDF file to extract markdown content and images, and upload them to S3 with a structured naming format.

    Args:
        file_content (bytes): The content of the uploaded PDF file.
        file_name (str): The original name of the uploaded file.

    Returns:
        dict: A dictionary with S3 URLs for the markdown file, extracted images, and status information.
    """
    logging.basicConfig(level=logging.DEBUG)

    try:
        logging.debug("Starting the PDF processing function.")

        # Step 1: Validate the PDF file
        if not file_content.startswith(b"%PDF"):
            logging.error("The uploaded file is not a valid PDF.")
            raise ValueError("The provided file is not a valid PDF.")
        logging.debug("PDF file content validated.")

        # Step 2: Create a cleaned-up filename for S3 storage
        pdf_filename = Path(file_name).stem.replace(" ", "_").upper()
        pdf_filename = pdf_filename.split('_')[0].upper()
        logging.debug(f"Structured S3 filename: {pdf_filename}")

        # Step 3: Write the PDF content to a temporary file
        temp_pdf_path = Path(f"temp_{uuid4().hex[:8]}.pdf")
        with open(temp_pdf_path, "wb") as temp_file:
            temp_file.write(file_content)
        logging.debug(f"Temporary PDF saved to {temp_pdf_path}.")

        # Step 4: Configure pipeline options for image extraction
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.do_table_structure = True
        logging.debug(f"Pipeline options configured: {pipeline_options}")

        # Step 5: Initialize DocumentConverter and convert the PDF
        logging.debug("Initializing DocumentConverter...")
        doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        conv_res = doc_converter.convert(Path(temp_pdf_path))
        logging.debug("PDF conversion completed successfully.")

        # Step 6: Extract and upload images to S3
        logging.debug("Extracting images from PDF...")
        image_s3_urls = []
        picture_counter = 0
        for element, _level in conv_res.document.iterate_items():
            if isinstance(element, PictureItem):
                picture_counter += 1
                temp_image_path = f"{Path(temp_pdf_path).stem}-image-{picture_counter}.png"
                with open(temp_image_path, "wb") as fp:
                    element.get_image(conv_res.document).save(fp, "PNG")
                logging.debug(f"Image saved temporarily: {temp_image_path}")

                # Upload the image to S3 under {pdf_filename}/images/
                image_s3_url = upload_file_to_s3(
                    file_path=temp_image_path,
                    source=pdf_filename,  # Store under PDF name
                    metadata={
                        "file_type": "image",
                        "original_filename": file_name
                    }
                )
                image_s3_urls.append(image_s3_url)
                logging.debug(f"Image uploaded to S3: {image_s3_url}")

                # Clean up the temporary image file
                os.remove(temp_image_path)
                logging.debug(f"Temporary image file deleted: {temp_image_path}")

        # Step 7: Save and upload Markdown content to S3
        logging.debug("Saving Markdown content...")
        temp_markdown_path = Path(f"{pdf_filename}_with_images.md")
        conv_res.document.save_as_markdown(temp_markdown_path, image_mode=ImageRefMode.REFERENCED)
        logging.debug(f"Markdown content saved temporarily: {temp_markdown_path}")

        # Upload Markdown to S3 under {pdf_filename}/markdown/
        markdown_s3_url = upload_file_to_s3(
            file_path=temp_markdown_path,
            source=pdf_filename,  # Store under PDF name
            metadata={
                "file_type": "markdown",
                "original_filename": file_name
            }
        )
        logging.debug(f"Markdown uploaded to S3: {markdown_s3_url}")

        # Step 8: Clean up temporary files
        os.remove(temp_pdf_path)
        os.remove(temp_markdown_path)
        logging.debug("Temporary files cleaned up.")

        # Ensure the images artifact folder is deleted last
        artifact_folder = Path(f"{pdf_filename}_with_images_artifacts")  # This is likely the folder being left behind

        try:
            if artifact_folder.exists() and artifact_folder.is_dir():
                for file in artifact_folder.glob("*"):  # Delete all files first
                    os.remove(file)
                os.rmdir(artifact_folder)  # Then delete the folder itself
                logging.debug(f"Deleted artifact folder: {artifact_folder}")
        except Exception as e:
            logging.error(f"Failed to delete artifact folder {artifact_folder}: {e}")

        # Step 9: Return success response
        logging.debug("PDF processing completed successfully.")
        return {
            "markdown_s3_url": markdown_s3_url,
            "image_s3_urls": image_s3_urls,
            "pdf_filename": pdf_filename,
            "status": "success",
            "message": "PDF processed and uploaded to S3 successfully"
        }

    except Exception as e:
        logging.error(f"Error processing PDF: {e}", exc_info=True)
        raise RuntimeError(f"Error processing PDF: {str(e)}")