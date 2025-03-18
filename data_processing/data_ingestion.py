import os
import re
import time
import boto3
import requests
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Load environment variables from .env
load_dotenv()

# AWS S3 Configuration
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Initialize S3 Client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

# Function to clean filenames and remove unnecessary text
def clean_filename(text, year):
    text = re.sub(r"\(opens in new window\)", "", text, flags=re.IGNORECASE)
    text = re.sub(rf"Q{year}", "", text)
    text = re.sub(r"[<>:\"/\\|?*]", "", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("_")
    return text

# Function to upload a file to S3
def upload_file_to_s3(local_path, s3_path):
    try:
        s3_client.upload_file(local_path, S3_BUCKET_NAME, s3_path)
        print(f"Uploaded to S3: s3://{S3_BUCKET_NAME}/{s3_path}")
        return True
    except Exception as e:
        print(f"Error uploading {local_path}: {e}")
        return False

# Function to delete a local file after uploading to S3
def delete_local_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted local file: {file_path}")

# Function to delete temp_downloads folder at the end**
def delete_temp_folder(folder_path):
    """Deletes the temp folder only after all processing is done."""
    if os.path.exists(folder_path) and not os.listdir(folder_path):  # Ensure folder is empty
        os.rmdir(folder_path)
        print(f"Deleted temp folder: {folder_path}")

# NVIDIA Quarterly Results Page URL
url = "https://investor.nvidia.com/financial-info/quarterly-results/default.aspx"

# Temporary download directory inside the current working directory
TEMP_DOWNLOAD_DIR = os.path.join(os.getcwd(), "temp_downloads")
os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)

# Initialize Selenium WebDriver
driver = webdriver.Chrome()
driver.get(url)

# Define a wait instance
wait = WebDriverWait(driver, 10)

# Define the years to check
years_to_check = ["2020", "2021", "2022", "2023", "2024", "2025"]

results = []

# Function to download a PDF file locally
def download_pdf(pdf_url, local_path):
    try:
        print(f"Downloading: {pdf_url}")
        response = requests.get(pdf_url, stream=True)

        if response.status_code == 200:
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Saved locally: {local_path}")
            return True
        else:
            print(f"Failed to download {pdf_url}, Status Code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {pdf_url}: {e}")
        return False

# Loop through each desired year
for year in years_to_check:
    try:
        year_dropdown = wait.until(EC.presence_of_element_located((By.ID, "_ctrl0_ctl75_selectEvergreenFinancialAccordionYear")))
        year_select = Select(year_dropdown)
        year_select.select_by_value(year)
        print(f"Processing year: {year}")
    except Exception as e:
        print(f"Could not select year {year}: {e}")
        continue

    time.sleep(2)

    try:
        accordion_container = driver.find_element(By.ID, "_ctrl0_ctl75_divFinancialAccordionItemsContainer")
    except Exception as e:
        print(f"Accordion container not found for year {year}: {e}")
        continue

    quarter_headers = accordion_container.find_elements(By.XPATH, ".//button")
    for header in quarter_headers:
        try:
            if header.get_attribute("aria-expanded") != "true":
                driver.execute_script("arguments[0].scrollIntoView(true);", header)
                header.click()
                time.sleep(1)
        except Exception as e:
            print(f"Error clicking quarter header '{header.text.strip()}' for year {year}: {e}")

    time.sleep(2)

    pdf_links = accordion_container.find_elements(By.XPATH, ".//a[contains(@href, '.pdf')]")
    for link in pdf_links:
        try:
            raw_text = link.text.strip()
            pdf_url = link.get_attribute("href")

            link_text = clean_filename(raw_text, year)

            if re.search(r"(10-K|10-Q|Form 10-Q)", link_text, re.IGNORECASE):
                try:
                    quarter = link.find_element(By.XPATH, "preceding::button[1]").text.strip()
                except Exception:
                    quarter = "Unknown"

                quarter = re.sub(r"Quarter (\d+)", r"Q\1", quarter).replace("Fourth", "Q4").replace("Third", "Q3").replace("Second", "Q2").replace("First", "Q1")
                quarter = re.sub(rf"\s*Q{year}\s*", "", quarter).strip()
                quarter = f"{year}{quarter}"

                filename = f"{quarter}_{link_text}.pdf"
                filename = filename.replace(f"_{year}Q", "_")
                local_path = os.path.join(TEMP_DOWNLOAD_DIR, filename)

                s3_path = f"nvidia-filings/{quarter}/{filename}"

                # print(f"DEBUG: Year={year}, Quarter={quarter}, Filename={filename}, S3 Path={s3_path}")

                if download_pdf(pdf_url, local_path):
                    if upload_file_to_s3(local_path, s3_path):
                        delete_local_file(local_path)

                results.append({
                    "year": year,
                    "quarter": quarter,
                    "name": link_text,
                    "url": pdf_url,
                    "s3_path": f"s3://{S3_BUCKET_NAME}/{s3_path}"
                })
        except Exception as e:
            print(f"Error processing PDF for year {year}: {e}")

driver.quit()

# print("Found filings:")
# for item in results:
#     print(f"Year: {item['year']} | Quarter: {item['quarter']} | Name: {item['name']} | URL: {item['url']} | S3: {item['s3_path']}")

delete_temp_folder(TEMP_DOWNLOAD_DIR)