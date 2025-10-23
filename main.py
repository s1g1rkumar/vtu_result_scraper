import os
import warnings
import time
import random
import io
import json
import sys
import concurrent.futures 
from typing import Any, Dict, List, Tuple

# NEW IMPORT for serving frontend
from flask import Flask, jsonify, request, send_file, Response, render_template 
from flask_cors import CORS 
from dotenv import load_dotenv

# Web Scraping Libraries
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Gemini API Libraries (for Advanced CAPTCHA Solving)
from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- Configuration & Initialization ---

# Suppress warnings for insecure requests
warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables from .env file
load_dotenv()

# --- MODIFIED KEY LOADING BLOCK (Using only one key) ---
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("="*60)
    print("❌ CRITICAL FAILURE: 'GEMINI_API_KEY' not found in environment variables.")
    print("Please define 'GEMINI_API_KEY' in your .env file or environment.")
    print("The application will not start.")
    print("="*60)
    sys.exit(1) # Stop execution and return a non-zero error code
# --- END MODIFIED KEY LOADING BLOCK ---

MODEL_NAME = 'gemini-2.5-flash'
MAX_WORKERS = 5 # Maximum number of concurrent requests to speed up bulk processing

# Flask App Setup
app = Flask(__name__)
CORS(app) 

# Global in-memory storage for temporary Excel files
TEMP_EXCEL_STORAGE: Dict[str, io.BytesIO] = {}

# Define a set of current default URLs
DEFAULT_INDEX_URL = 'https://results.vtu.ac.in/JJEcbcs25/index.php'
DEFAULT_RESULT_URL = 'https://results.vtu.ac.in/JJEcbcs25/resultpage.php'

# --- CAPTCHA Solving (Gemini Single Key Method with 429 Handling) ---

def solve_captcha_gemini(image_content: bytes) -> str | None:
    """
    Sends raw image bytes to the Gemini model for CAPTCHA extraction.
    Implements a specific retry mechanism for 429 RESOURCE_EXHAUSTED errors.
    """
    global GEMINI_API_KEY
    
    if not GEMINI_API_KEY:
        app.logger.error("No API key is available.")
        return None

    # Max attempts for the CAPTCHA API call itself (not the fetch_result loop)
    MAX_API_ATTEMPTS = 5
    
    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as e:
            app.logger.error(f"Client Initialization Error: {e}")
            return None

        try:
            # --- API Call Logic (Unchanged) ---
            captcha_schema = types.Schema(
                type=types.Type.OBJECT,
                properties={"captcha_code": types.Schema(type=types.Type.STRING, description="The exact 6-character alphanumeric code visible in the CAPTCHA image.")},
                required=["captcha_code"]
            )
            prompt_text = (
                "Analyze the following CAPTCHA image. The image is distorted, noisy, "
                "and may have overlapping lines or color variations. First, mentally **clean "
                "and sharpen the image** to isolate the characters. The final code is always "
                "a 6-character alphanumeric string (A-Z, 0-9). Extract ONLY this 6-character "
                "code. Return the result in a JSON object that adheres to the provided schema. "
                "Do not include any explanation or surrounding text."
            )
            
            contents = [
                types.Part.from_text(text=prompt_text),
                types.Part.from_bytes(data=image_content, mime_type='image/png')
            ]
            
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=captcha_schema,
                temperature=0.0
            )

            app.logger.info(f"-> Submitting CAPTCHA to {MODEL_NAME} (Attempt {attempt})...")
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=config
            )
            # --- End API Call Logic ---

            # Extract and Parse the JSON Result
            json_data = json.loads(response.text)
            captcha_code = json_data.get("captcha_code", "").strip()
            
            if len(captcha_code) == 6 and captcha_code.isalnum():
                app.logger.info(f"-> AI Solution: '{captcha_code}'")
                return captcha_code
            else:
                app.logger.warning(f"-> AI Result Invalid: {captcha_code}. Full JSON: {response.text}")
                # Treat invalid output as a failure and let the outer loop retry the whole fetch.
                return None

        except APIError as e:
            error_message = str(e)
            
            # --- START 429 ERROR HANDLING ---
            if 'RESOURCE_EXHAUSTED' in error_message or '429' in error_message:
                app.logger.warning("-> Quota limit hit (429 RESOURCE_EXHAUSTED). Implementing mandatory wait.")
                
                # Try to extract the retryDelay from the JSON body
                retry_delay_seconds = 0
                try:
                    # The Gemini API error body is often complex, parse it for the RetryInfo
                    error_json = json.loads(error_message.split(":", 1)[1].strip())['error']
                    for detail in error_json.get('details', []):
                        if '@type' in detail and 'RetryInfo' in detail['@type']:
                            # The delay is usually a string like '45s'
                            delay_str = detail['retryDelay'].replace('s', '')
                            retry_delay_seconds = float(delay_str)
                            break
                except Exception:
                    # Fallback to a safe, long delay if parsing fails (e.g., 50 seconds)
                    retry_delay_seconds = 50.0

                if retry_delay_seconds > 0:
                    app.logger.info(f"--> Waiting for required {retry_delay_seconds:.2f} seconds before retrying API call...")
                    time.sleep(retry_delay_seconds + random.uniform(0.5, 1.5)) # Add small buffer
                    
                    # Continue the inner loop to retry the API call with the same CAPTCHA image
                    continue 
                
                # If we get a 429 but the retry delay is zero or cannot be parsed, fail the API call
                app.logger.error("-> Failed to parse retry delay for 429 error. Aborting CAPTCHA attempt.")
                return None
            # --- END 429 ERROR HANDLING ---
            
            # For other API errors (e.g., Auth, invalid model), log and fail the AI step
            app.logger.error(f"-> GEMINI API Error (Attempt {attempt}): {e}")
            return None 
            
        except json.JSONDecodeError:
            app.logger.error(f"-> JSON Parsing Error (Attempt {attempt}).")
            return None
            
        except Exception as e:
            app.logger.error(f"-> Unexpected error during AI solving (Attempt {attempt}): {e}")
            return None

    app.logger.error(f"-> Failed to solve CAPTCHA after {MAX_API_ATTEMPTS} API attempts.")
    return None # Return None to trigger retry in the outer fetch_result loop


# --- Core Scraper Logic (fetch_result remains mostly unchanged) ---

def fetch_result(usn: str, index_url: str, result_url: str) -> dict | None:
    """Fetch VTU result for a given USN with automatic retry on CAPTCHA failure."""
    
    global GEMINI_API_KEY
    if not GEMINI_API_KEY:
        return None

    # NOTE: Requests.Session is NOT thread-safe for reuse across threads. 
    session = requests.Session() 
    
    # Configure retry mechanism
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    # Headers to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        'Referer': index_url # Dynamic Referer
    }
    
    MAX_ATTEMPTS = 3 # Max attempts for the full fetch (network + CAPTCHA solve)
    
    # --- START RETRY LOOP (AI Attempts Only) ---
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt == 1:
             app.logger.info(f"[Attempt {attempt}] Starting fetch for {usn} from {index_url}...")
        
        try:
            # Step 1 & 2: Get Token and CAPTCHA
            session.cookies.clear()
            r = session.get(index_url, headers=headers, verify=False, timeout=10)
            if r.status_code != 200:
                time.sleep(random.uniform(0.5, 1))
                continue
            
            soup = BeautifulSoup(r.text, 'html.parser')
            
            token_tag = soup.find('input', {'name': 'Token'})
            if not token_tag:
                app.logger.error(f"Error for {usn}: No Token found at {index_url}. Aborting.")
                return None
            token = token_tag['value']
            
            captcha_img = soup.find('img', alt='CAPTCHA') or soup.find('img', src=lambda s: s and 'captcha' in s.lower())
            if not captcha_img:
                app.logger.error(f"Error for {usn}: No CAPTCHA image found. Aborting.")
                return None
            
            captcha_src = urljoin(index_url, captcha_img['src'])
            captcha_r = session.get(captcha_src, headers=headers, verify=False, timeout=10)
            if captcha_r.status_code != 200:
                time.sleep(random.uniform(0.5, 1))
                continue
            
            image_content = captcha_r.content
            
            # Step 3: Solve CAPTCHA using Gemini (Now handles 429 internally)
            captcha_code = solve_captcha_gemini(image_content)
            
            if not captcha_code:
                # If Gemini fails (after its internal retries), retry the network step
                time.sleep(random.uniform(0.5, 1.5)) 
                continue 
            
            # Step 4 & 5: Submit Result and Check for Success (Unchanged)
            data = {'Token': token, 'lns': usn, 'captchacode': captcha_code}
            
            post_r = session.post(result_url, data=data, headers=headers, verify=False, timeout=10)
            
            if post_r.status_code != 200:
                time.sleep(random.uniform(0.5, 1))
                continue
            
            text_lower = post_r.text.lower()
            if 'invalid captcha code' in text_lower or 'wrong' in text_lower or 'error' in text_lower:
                time.sleep(random.uniform(0.5, 1.5))
                continue
            
            if 'student name' not in text_lower and 'university seat number' not in text_lower:
                app.logger.warning(f"Error for {usn}: No result found (Invalid USN/expired link).")
                return None
            
            # --- SUCCESS PATH: Parsing logic (Unchanged) ---
            result_soup = BeautifulSoup(post_r.text, 'html.parser')
            
            name = "Unknown"
            name_parts = result_soup.find_all('td')
            for i, td in enumerate(name_parts):
                if 'Student Name' in td.get_text():
                    if i+1 < len(name_parts):
                        name_text = name_parts[i+1].get_text(strip=True)
                        name = name_text.split(':', 1)[1].strip() if ':' in name_text else name_text
            
            semester = "Unknown"
            for div in result_soup.find_all('div', string=lambda t: t and 'Semester' in t):
                sem_text = div.get_text(strip=True)
                if ':' in sem_text:
                    semester = sem_text.split(':', 1)[1].strip()
                    break
            
            subjects: List[Dict[str, str]] = []
            table_body = result_soup.find('div', {'class': 'divTableBody'})
            if table_body:
                rows = table_body.find_all('div', {'class': 'divTableRow'})
                for row in rows[1:]:
                    cells = row.find_all('div', {'class': 'divTableCell'})
                    if len(cells) >= 7:
                        subjects.append({
                            'code': cells[0].get_text(strip=True),
                            'name': cells[1].get_text(strip=True),
                            'internal': cells[2].get_text(strip=True),
                            'external': cells[3].get_text(strip=True),
                            'total': cells[4].get_text(strip=True),
                            'result': cells[5].get_text(strip=True),
                            'announced': cells[6].get_text(strip=True)
                        })
            
            app.logger.info(f"✓ Success: Result fetched for {usn} ({name}).")
            return {'usn': usn, 'name': name, 'semester': semester, 'subjects': subjects}

        except Exception as e:
            app.logger.error(f"Error processing {usn} on attempt {attempt}: {e}")
            time.sleep(random.uniform(0.5, 1.5))
            continue
    # --- END RETRY LOOP ---
    
    app.logger.error(f"❌ Failed to retrieve result for {usn} after {MAX_ATTEMPTS} attempts.")
    return None

# --- Bulk Processing Function (Unchanged, relies on fetch_result) ---

def get_bulk_results(usn_list: List[str], index_url: str, result_url: str, subject_code: str = '') -> Tuple[List[Dict], List[Dict]]:
    """
    Processes a list of USNs concurrently, fetching results for each.
    Returns (successful_results, failed_usns).
    """
    successful_results: List[Dict] = []
    failed_usns: List[Dict] = []
    
    usn_list = [u.strip().upper() for u in usn_list if u.strip()]

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_usn = {
            executor.submit(fetch_result, usn, index_url, result_url): usn 
            for usn in usn_list
        }
        
        for future in concurrent.futures.as_completed(future_to_usn):
            usn = future_to_usn[future]
            try:
                raw_result = future.result()
                
                if raw_result:
                    if subject_code:
                        filtered_subjects = [
                            sub for sub in raw_result['subjects'] 
                            if sub['code'].lower() == subject_code.lower()
                        ]
                        raw_result['subjects'] = filtered_subjects

                    successful_results.append(raw_result)
                else:
                    failed_usns.append({"usn": usn, "error": "Failed to retrieve result after multiple CAPTCHA attempts or USN is invalid/not found."})
            
            except Exception as exc:
                app.logger.error(f'USN {usn} generated an exception: {exc}')
                failed_usns.append({"usn": usn, "error": f"Internal exception: {exc}"})
            
            # Add a small, quick, random delay between processing of completed tasks
            time.sleep(random.uniform(0.1, 0.3)) 

    return successful_results, failed_usns

# --- Excel Generation, Flask Routes, and Main Block (Unchanged) ---

def generate_bulk_excel_file(results_data: List[dict]) -> tuple[str, io.BytesIO]:
    if not results_data:
        raise ValueError("No data provided for Excel generation.")

    consolidated_rows = []

    for result in results_data:
        usn = result['usn']
        name = result.get('name', 'N/A')
        semester = result.get('semester', 'N/A')
        
        student_base_data = {
            'USN': usn,
            'Name': name,
            'Semester': semester,
        }
        
        for subject in result['subjects']:
            row = student_base_data.copy()
            row.update({
                'Subject Code': subject.get('code', ''),
                'Subject Name': subject.get('name', ''),
                'Internal Marks': subject.get('internal', ''),
                'External Marks': subject.get('external', ''),
                'Total Marks': subject.get('total', ''),
                'Result': subject.get('result', ''),
                'Announced Date': subject.get('announced', '')
            })
            consolidated_rows.append(row)

    consolidated_df = pd.DataFrame(consolidated_rows)
    
    COLUMNS_ORDER = [
        'USN', 'Name', 'Semester', 'Subject Code', 'Subject Name', 
        'Internal Marks', 'External Marks', 'Total Marks', 'Result', 'Announced Date'
    ]
    final_df = consolidated_df.reindex(columns=COLUMNS_ORDER, fill_value='')

    output = io.BytesIO()
    timestamp = int(time.time())
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        final_df.to_excel(writer, sheet_name='Consolidated Results', index=False)
    
    output.seek(0)
    
    base_usn = results_data[0].get('usn', 'Bulk')
    filename = f"VTU_Results_Consolidated_{base_usn}_{timestamp}.xlsx"
    
    return filename, output

@app.route('/', methods=['GET'])
def index() -> Response:
    return render_template('index.html', 
        default_index_url=DEFAULT_INDEX_URL,
        default_result_url=DEFAULT_RESULT_URL
    )

@app.route('/api/vtu/download/<filename>', methods=['GET'])
def download_excel(filename: str) -> Response:
    excel_stream = TEMP_EXCEL_STORAGE.pop(filename, None)
    
    if excel_stream is None:
        return jsonify({"error": "File not found or link has expired. Please fetch the result again."}), 404
    
    app.logger.info(f"Serving and removing temporary file: {filename}")
    
    return send_file(
        excel_stream,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=filename
    )


@app.route('/api/vtu/results', methods=['POST'])
def get_bulk_vtu_results() -> Response:
    global GEMINI_API_KEY
    
    try:
        request_data: Any = request.get_json(silent=True)
        if not request_data or not isinstance(request_data, dict):
            return jsonify({"error": "Invalid or missing JSON body."}), 400

        usn_list_raw = request_data.get('usns')
        subject_code = str(request_data.get('subject_code', '')).strip()
        
        index_url = str(request_data.get('index_url', DEFAULT_INDEX_URL)).strip()
        result_url = str(request_data.get('result_url', DEFAULT_RESULT_URL)).strip()
        
        if not index_url.startswith('http') or not result_url.startswith('http'):
             return jsonify({"error": "Invalid 'index_url' or 'result_url'. Must be a complete URL starting with http/https."}), 400
        
        if not isinstance(usn_list_raw, list) or not usn_list_raw:
            return jsonify({"error": "Missing or invalid 'usns' list in the request body."}), 400
        
        usn_list = [str(u).strip() for u in usn_list_raw if str(u).strip()]
        
    except Exception as e:
        app.logger.error(f"Error parsing request body: {e}")
        return jsonify({"error": "Failed to parse request data."}), 400

    app.logger.info(f"API Request received for {len(usn_list)} USNs. Source URL: {index_url}")
    
    if not GEMINI_API_KEY:
         return jsonify({"error": "No GEMINI API Key is available. Cannot proceed with AI CAPTCHA solving."}), 500

    successful_results, failed_usns = get_bulk_results(usn_list, index_url, result_url, subject_code)
    
    download_url = "No Excel file generated (No successful results)."
    
    if successful_results:
        try:
            filename, excel_stream = generate_bulk_excel_file(successful_results)
            TEMP_EXCEL_STORAGE[filename] = excel_stream
            download_url = f"{request.url_root.rstrip('/')}/api/vtu/download/{filename}"
        except Exception as e:
            app.logger.error(f"Error generating Bulk Excel file: {e}")
            download_url = f"Error generating Excel file: {str(e)}"

    response_data = {
        "status": "partial_success" if successful_results and failed_usns else ("success" if successful_results else "failure"),
        "total_requested": len(usn_list),
        "total_successful": len(successful_results),
        "total_failed": len(failed_usns),
        "download_url": download_url,
        "current_vtu_index_url": index_url,
        "current_vtu_result_url": result_url,
        "successful_results": successful_results,
        "failed_usns": failed_usns
    }

    return jsonify(response_data), 200

if __name__ == '__main__':
    import logging
    
    app.logger.setLevel(logging.INFO)
    app.logger.info(f"Successfully configured application for single API key. Concurrency set to {MAX_WORKERS} workers. Starting Flask server.")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
