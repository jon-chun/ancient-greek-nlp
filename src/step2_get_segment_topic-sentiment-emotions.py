#!/usr/bin/env python3
"""
bible_sentiment_analysis.py - Analyzes segmented Bible text using Gemini API for topic, sentiment, 
and emotion analysis.
"""

import os
import re
import json
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

# --- Configuration ---

# Load environment variables (especially GOOGLE_API_KEY)
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Define base data directory relative to the script location
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DATA_DIR = SCRIPT_DIR.parent / 'data'  # Assumes script is in src directory

# Define input and output directories for Bible text
CORPUS_NAME = "bible_ancient-greek"
INPUT_SUBDIR = Path('bible_ancient-greek/segments')
OUTPUT_SUBDIR = Path('bible_ancient-greek/analysis_json')

# Complete paths
INPUT_DIR = BASE_DATA_DIR / INPUT_SUBDIR
OUTPUT_DIR = BASE_DATA_DIR / OUTPUT_SUBDIR
OUTPUT_REPORT_DIR = BASE_DATA_DIR / 'bible_ancient-greek/reports'

# --- Model and Retry Configuration ---
DEFAULT_MODEL_NAME = "gemini-2.0-flash"
MAX_API_RETRIES = 3  # Total number of attempts (1 initial + 2 retries)
RETRY_DELAY = 5  # Seconds to wait between retries

# Gemini Configuration
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
GENERATION_CONFIG = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Delay between distinct file API calls
API_CALL_DELAY = 2

# Marker for JSON extraction
JSON_EXTRACTION_MARKER = "Final Results for JSON Extraction:"

# --- Logging Setup ---

def setup_logging(log_dir):
    """Sets up logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"bible_analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized.")
    return log_filename

# --- Prompt Template --- 
# Modified for Ancient Greek Bible text analysis
PROMPT_TEMPLATE = """
###TEXT_CONTENT (Ancient Greek Bible segment):

{text_content}

###INSTRUCTIONS:

You are analyzing an Ancient Greek text segment from the Bible. Carefully think step by step to analyze this text segment.

First, identify what Bible section and events/teachings this segment likely contains. Use the following format:

### Text Identification:
[Briefly identify what part of the Bible this appears to be (Old Testament, New Testament, specific book if recognizable)]

### Text Summary:
[Provide a concise summary (approx. 100-150 words) of the main religious themes, narratives, or teachings in this text segment]

---
Now, your main task is to identify all the distinct topics, themes, theological concepts, or narrative elements present in this text segment. For each identified topic, conduct a sentiment analysis and an emotional analysis based on Plutchik's eight basic emotions. Provide the following detailed assessment for each topic:

Sentiment Polarity Analysis:
Reasoning: Give a concise rationale for the assigned Polarity Score and Confidence values. Consider the religious/spiritual context. Explain *why* the score and confidence were chosen.
Polarity Score: Assign a float value -1.0 to 1.0 (use 99.9 if unknown).
Confidence: Provide an integer percentage 0-100 (use 0 if unknown).

Emotional Weights Analysis:
Reasoning: Give a concise rationale for the assigned emotional weight and confidence values for *each* emotion. Consider the religious context. Explain *why* the weight and confidence were chosen for each emotion.
Emotion Dictionary: For each of Plutchik's eight emotions (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation), provide:
Weight: A float value 0.0 to 1.0 (use 99.9 if unknown).
Confidence: An integer percentage 0-100 (use 0 if unknown).

Consider theological context, spiritual tone, religious perspective. Reflect uncertainty in confidence scores. Include all topics even if info is insufficient (use 99.9 score/weight, 0% confidence).

After completing the analysis for all topics, include a section titled "Final Results for JSON Extraction:". Immediately following this title, provide ONLY a valid JSON list. Each element in the list should be a JSON object representing a single topic. Each topic object must contain keys: "topic", "sentiment_polarity", and "emotions".

- The value for "topic" should be a string.
- The value for "sentiment_polarity" should be an object with keys "score" (float or 99.9), "confidence" (integer), and "reasoning" (string).
- The value for "emotions" should be an object where keys are the eight emotion names (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation) and the value for each emotion key is an object with keys "weight" (float[0.0,1.0] or 99.9), "confidence" (int[0,100]), and "reasoning" (string).

Example of the required overall JSON list structure (ONLY the list goes under the Final Results heading):
[
  topic_object_1,
  topic_object_2,
  ...
]

Example structure for ONE topic object inside the list (use double quotes for all keys and string values in the final JSON):
{{  # Start topic object
    "topic": "Example Topic Name",
    "sentiment_polarity": {{ "score": 0.7, "confidence": 90, "reasoning": "<insert concise 1-2 sentence explanation for values>" }},
    "emotions": {{ # Start emotions object
        "Joy": {{ "weight": 0.8, "confidence": 85, "reasoning": "<insert concise 1-2 sentence explanation for values>" }},
        "Trust": {{ "weight": 0.4, "confidence": 70, "reasoning": "<insert concise 1-2 sentence explanation for values>" }},
        "Fear": {{ "weight": 0.1, "confidence": 95, "reasoning": "<insert concise 1-2 sentence explanation for values>" }},
        "Surprise": {{ "weight": 0.2, "confidence": 80, "reasoning": "<insert concise 1-2 sentence explanation for values>" }},
        "Sadness": {{ "weight": 0.1, "confidence": 90, "reasoning": "<insert concise 1-2 sentence explanation for values>" }},
        "Disgust": {{ "weight": 0.0, "confidence": 98, "reasoning": "<insert concise 1-2 sentence explanation for values>" }},
        "Anger": {{ "weight": 0.1, "confidence": 88, "reasoning": "<insert concise 1-2 sentence explanation for values>" }},
        "Anticipation": {{ "weight": 0.5, "confidence": 75, "reasoning": "<insert concise 1-2 sentence explanation for values>" }}
    }} # End emotions object
}} # End topic object

Ensure the final output under the marker is ONLY the JSON list `[...]` and is syntactically correct JSON. Do not include ```json ``` markdown around the JSON list.
"""

# --- Gemini API Call Function ---
def call_gemini_api(model, text_content, attempt_num):
    """Calls the Gemini API with the given text and prompt template."""
    prompt = PROMPT_TEMPLATE.format(text_content=text_content)
    logging.info(f"Attempt {attempt_num}/{MAX_API_RETRIES}: Sending request to Gemini API...")
    try:
        response = model.generate_content(
            prompt,
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS
            )

        if response.prompt_feedback and response.prompt_feedback.block_reason:
             logging.error(f"Attempt {attempt_num}: API request blocked: {response.prompt_feedback.block_reason}")
             raise ValueError(f"API request blocked: {response.prompt_feedback.block_reason}")
        if not response.candidates:
            logging.error(f"Attempt {attempt_num}: API response has no candidates.")
            if hasattr(response, 'prompt_feedback'): logging.error(f"Attempt {attempt_num}: Prompt Feedback: {response.prompt_feedback}")
            raise ValueError("API response was empty or malformed (no candidates).")

        api_result_text = ""
        if response.candidates[0].content and response.candidates[0].content.parts:
            api_result_text = "".join(part.text for part in response.candidates[0].content.parts)
        elif response.candidates[0].content:
             logging.warning(f"Attempt {attempt_num}: API response candidate content has no parts.")
        else:
             logging.warning(f"Attempt {attempt_num}: API response candidate has no content.")

        if not api_result_text.strip():
             logging.warning(f"Attempt {attempt_num}: API returned an empty text response.")
             raise ValueError("API returned empty text response.")

        logging.info(f"Attempt {attempt_num}: Received response from Gemini API (Length: {len(api_result_text)}).")
        return api_result_text

    except Exception as e:
        logging.error(f"Attempt {attempt_num}: Error calling Gemini API: {e}")
        raise ValueError(f"Gemini API call failed: {e}") from e


def parse_api_response(api_response_text, attempt_num):
    """
    Parses the API response text to extract metadata (identification, summary)
    and the topic analysis JSON list.
    
    Returns:
        tuple: (metadata_dict, topic_list)
    Raises:
        ValueError: If essential parsing or validation fails.
    """
    logging.debug(f"Attempt {attempt_num}: Parsing API response (length {len(api_response_text)})...")
    metadata_dict = {'text_identification': None, 'text_summary': None}
    topic_list = None

    # --- Extract Metadata ---
    try:
        # Extract text identification
        id_marker = "### Text Identification:"
        id_match = re.search(rf"{id_marker}\s*(.*?)(?=###|\n\n|$)", api_response_text, re.DOTALL)
        if id_match:
            metadata_dict['text_identification'] = id_match.group(1).strip()

        # Extract text summary
        summary_marker = "### Text Summary:"
        summary_match = re.search(rf"{summary_marker}\s*(.*?)(?=---|\n\n|$)", api_response_text, re.DOTALL)
        if summary_match:
            metadata_dict['text_summary'] = summary_match.group(1).strip()

        logging.info(f"Attempt {attempt_num}: Metadata parsing completed. Identification and summary extracted.")
    except Exception as e:
        logging.exception(f"Attempt {attempt_num}: Error during metadata extraction: {e}")
        metadata_dict = {'text_identification': None, 'text_summary': None}

    # --- Extract Topic Analysis JSON List ---
    try:
        json_marker_index = api_response_text.find(JSON_EXTRACTION_MARKER)

        if json_marker_index == -1:
            logging.warning(f"Attempt {attempt_num}: Marker '{JSON_EXTRACTION_MARKER}' not found. Attempting fallback.")
            list_start_index = api_response_text.rfind('[')
            if list_start_index == -1:
                raise ValueError("JSON Marker not found and fallback failed to find '[' for topic list.")
            json_string = api_response_text[list_start_index:]
        else:
            logging.info(f"Attempt {attempt_num}: JSON Marker found at index {json_marker_index}.")
            list_start_index = api_response_text.find('[', json_marker_index + len(JSON_EXTRACTION_MARKER))
            if list_start_index == -1:
                raise ValueError("JSON list start '[' not found after JSON marker.")
            json_string = api_response_text[list_start_index:]

        # Find the end of the JSON list
        open_brackets = 0
        list_end_index = -1
        for i, char in enumerate(json_string):
            if char == '[':
                open_brackets += 1
            elif char == ']':
                open_brackets -= 1
                if open_brackets == 0 and list_end_index == -1:
                    list_end_index = i + 1  # Include the closing bracket
                    break

        if list_end_index == -1:
            raise ValueError("Matching JSON list end ']' not found.")

        json_string_cleaned = json_string[:list_end_index].strip()
        if json_string_cleaned.startswith("```json"):
            json_string_cleaned = json_string_cleaned[7:].strip()
        if json_string_cleaned.endswith("```"):
            json_string_cleaned = json_string_cleaned[:-3].strip()
            
        if not json_string_cleaned:
            raise ValueError("Cleaned JSON string for topic list is empty.")

        topic_list = json.loads(json_string_cleaned)
        
        # Validate JSON structure
        if not isinstance(topic_list, list):
            raise ValueError(f"Parsed JSON is not a list, but type {type(topic_list)}.")
            
        min_topics_required = 2
        if len(topic_list) < min_topics_required:
            logging.warning(f"Found only {len(topic_list)} topics, fewer than the expected minimum {min_topics_required}.")
        
        # Check structure of first topic
        if topic_list and isinstance(topic_list[0], dict):
            required_keys = {"topic", "sentiment_polarity", "emotions"}
            if not required_keys.issubset(set(topic_list[0].keys())):
                missing = required_keys - set(topic_list[0].keys())
                raise ValueError(f"First topic missing required keys: {missing}")
                
            # Check sentiment structure
            sentiment = topic_list[0].get("sentiment_polarity", {})
            if not isinstance(sentiment, dict) or not {"score", "confidence", "reasoning"}.issubset(set(sentiment.keys())):
                raise ValueError("Invalid sentiment_polarity structure")
                
            # Check emotions structure
            emotions = topic_list[0].get("emotions", {})
            if not isinstance(emotions, dict):
                raise ValueError("Invalid emotions structure")
                
            required_emotions = {"Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation"}
            if not required_emotions.issubset(set(emotions.keys())):
                missing = required_emotions - set(emotions.keys())
                raise ValueError(f"Missing required emotions: {missing}")
                
            # Check structure of one emotion
            first_emotion = emotions.get("Joy", {})
            if not isinstance(first_emotion, dict) or not {"weight", "confidence", "reasoning"}.issubset(set(first_emotion.keys())):
                raise ValueError("Invalid emotion value structure")
        
        logging.info(f"Attempt {attempt_num}: Successfully extracted and validated topic list with {len(topic_list)} topics.")
        
    except json.JSONDecodeError as e:
        logging.error(f"Attempt {attempt_num}: JSON decode error: {e}")
        raise ValueError(f"Failed to decode JSON topic list: {e}")
    except ValueError as e:
        logging.error(f"Attempt {attempt_num}: Error during JSON processing: {e}")
        raise
    except Exception as e:
        logging.exception(f"Attempt {attempt_num}: Unexpected error during JSON processing: {e}")
        raise ValueError(f"Unexpected JSON processing error: {e}")

    return metadata_dict, topic_list


def process_bible_segments(input_dir, output_dir, model, force_overwrite=False):
    """Process all segmented Bible files and generate analysis for each."""
    processing_summary = []
    total_files = 0
    success_count = 0
    skipped_count = 0
    error_count = 0
    api_call_counter = 0

    if not input_dir.is_dir():
        logging.error(f"Input directory not found: {input_dir}")
        return None

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Processing Bible segments from: {input_dir}")
    logging.info(f"Saving analysis results to: {output_dir}")

    # Find all segment files (format: N_bible_ancient-greek.txt)
    segment_files = sorted(list(input_dir.glob('*_bible_ancient-greek.txt')))
    if not segment_files:
        logging.warning(f"No segment files found in {input_dir}")
        return None

    total_files = len(segment_files)
    logging.info(f"Found {total_files} Bible segment files to process")

    for segment_file in segment_files:
        status = "pending"
        final_error_msg = None
        result_data = None
        processed_successfully = False
        final_output_filepath = None
        
        segment_number = segment_file.stem.split('_')[0]  # Extract the segment number
        output_filename = f"{segment_number}_bible_ancient-greek_analysis.json"
        output_filepath = output_dir / output_filename
        
        logging.info(f"--- Processing segment {segment_number}: {segment_file.name} ---")
        
        # Check if output already exists
        if output_filepath.exists() and not force_overwrite:
            logging.info(f"Output file already exists: {output_filepath.name}, skipping.")
            skipped_count += 1
            status = "skipped_exists"
            processing_summary.append({
                "input_file": str(segment_file.relative_to(BASE_DATA_DIR)),
                "output_file": str(output_filepath.relative_to(BASE_DATA_DIR)),
                "status": status,
                "error": None
            })
            continue
            
        # Process the file with retries
        for attempt in range(1, MAX_API_RETRIES + 1):
            try:
                if attempt > 1:
                    logging.info(f"Waiting {RETRY_DELAY} seconds before retry {attempt}...")
                    time.sleep(RETRY_DELAY)
                    
                # Read the segment file
                with open(segment_file, 'r', encoding='utf-8') as f:
                    segment_text = f.read()
                    
                if not segment_text.strip():
                    logging.error(f"Segment file {segment_file.name} is empty. Skipping.")
                    final_error_msg = "Input file is empty."
                    status = "error_empty_input"
                    break
                    
                # Call Gemini API
                api_response = call_gemini_api(model, segment_text, attempt)
                
                # Parse the response
                metadata, topic_list = parse_api_response(api_response, attempt)
                
                if not isinstance(topic_list, list):
                    raise ValueError("Parsing failed to return a valid topic list.")
                    
                # Success for this attempt
                api_call_counter += 1
                logging.info(f"Successfully processed segment {segment_number} on attempt {attempt} (API Call #{api_call_counter}).")
                
                # Construct result data
                result_data = {
                    "segment_metadata": {
                        "segment_number": segment_number,
                        "segment_filename": segment_file.name,
                        "text_identification": metadata.get('text_identification'),
                        "text_summary": metadata.get('text_summary')
                    },
                    "topic_analysis": topic_list
                }
                
                final_output_filepath = output_filepath
                processed_successfully = True
                status = "success"
                final_error_msg = None
                break  # Exit retry loop
                
            except ValueError as ve:
                logging.warning(f"Attempt {attempt}/{MAX_API_RETRIES} failed for segment {segment_number}: {ve}")
                final_error_msg = f"Attempt {attempt}: {ve}"
                if attempt == MAX_API_RETRIES:
                    logging.error(f"Failed to process segment {segment_number} after {MAX_API_RETRIES} attempts.")
                    status = "error_retries_failed"
                    
            except Exception as e:
                logging.exception(f"Unexpected error processing segment {segment_number} on attempt {attempt}: {e}")
                final_error_msg = f"Unexpected error: {e}"
                status = "error_unexpected"
                break
                
        # After retry loop
        if processed_successfully and result_data is not None and final_output_filepath is not None:
            try:
                with open(final_output_filepath, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
                logging.info(f"Successfully saved result: {final_output_filepath.name}")
                success_count += 1
            except Exception as e:
                logging.exception(f"Error writing JSON output {final_output_filepath.name}: {e}")
                status = "error_writing_output"
                final_error_msg = f"Failed to write JSON file: {e}"
                processed_successfully = False
                error_count += 1
        elif not processed_successfully:
            error_count += 1
            
        # Append to summary
        processing_summary.append({
            "input_file": str(segment_file.relative_to(BASE_DATA_DIR)),
            "output_file": str(final_output_filepath.relative_to(BASE_DATA_DIR)) if final_output_filepath else None,
            "status": status,
            "error": final_error_msg
        })
        
        # Delay between files
        if status != "skipped_exists":
            logging.debug(f"Waiting {API_CALL_DELAY}s before processing next segment...")
            time.sleep(API_CALL_DELAY)
            
    logging.info("--- Processing Complete ---")
    logging.info(f"Total segment files: {total_files}")
    logging.info(f"Successfully processed and written: {success_count}")
    logging.info(f"Skipped (output existed): {skipped_count}")
    logging.info(f"Errors (final): {error_count}")
    
    return processing_summary, total_files, success_count, skipped_count, error_count


def generate_reports(summary_data, report_dir, totals):
    """Generates human-readable and machine-readable reports."""
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_base_name = f"bible_analysis_report_{timestamp}"
    txt_report_path = report_dir / f"{report_base_name}.txt"
    json_report_path = report_dir / f"{report_base_name}.json"

    total_files, success_count, skipped_count, error_count = totals

    # Generate Text Report
    try:
        with open(txt_report_path, 'w', encoding='utf-8') as f:
            f.write("--- Ancient Greek Bible Text Analysis Report ---\n")
            f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Using Model: {DEFAULT_MODEL_NAME}\n")
            f.write(f"Max Attempts per File: {MAX_API_RETRIES}\n")
            f.write("\n--- Summary ---\n")
            f.write(f"Total segment files: {total_files}\n")
            f.write(f"Successfully processed & written: {success_count}\n")
            f.write(f"Skipped (output existed): {skipped_count}\n")
            f.write(f"Errors encountered (final): {error_count}\n")
            f.write("\n--- Detailed Log ---\n")
            
            if summary_data:
                for item in summary_data:
                    in_file = item.get("input_file", "N/A")
                    status = item.get("status", "N/A")
                    out_file = item.get("output_file", "N/A")
                    error = item.get("error", "")
                    
                    f.write(f"Input: {in_file} | Status: {status} | Output: {out_file}")
                    if error:
                        f.write(f" | Error: {error}\n")
                    else:
                        f.write("\n")
            else:
                f.write("No files were processed or summary data is unavailable.\n")

        logging.info(f"Text report generated: {txt_report_path}")
    except Exception as e:
        logging.error(f"Failed to generate text report: {e}")

    # Generate JSON Report
    try:
        report_json_data = {
            "report_metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "model_used": DEFAULT_MODEL_NAME,
                "max_attempts": MAX_API_RETRIES,
                "total_files_scanned": total_files,
                "success_count": success_count,
                "skipped_count": skipped_count,
                "error_count_final": error_count
            },
            "processing_details": summary_data if summary_data else []
        }
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(report_json_data, f, indent=2, ensure_ascii=False)
        logging.info(f"JSON report generated: {json_report_path}")
    except Exception as e:
        logging.error(f"Failed to generate JSON report: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Ancient Greek Bible segments using Gemini for topic/sentiment/emotion analysis.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing JSON output files.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help=f"Name of the Gemini model to use (default: {DEFAULT_MODEL_NAME}).")
    parser.add_argument("--delay", type=float, default=API_CALL_DELAY, help=f"Delay in seconds between processing different files (default: {API_CALL_DELAY}).")
    parser.add_argument("--retry-delay", type=float, default=RETRY_DELAY, help=f"Delay in seconds between retries for the same file (default: {RETRY_DELAY}).")
    parser.add_argument("--max-retries", type=int, default=MAX_API_RETRIES, help=f"Total maximum attempts per file (default: {MAX_API_RETRIES}).")
    parser.add_argument('--debug', action='store_true', help='Enable debug logging.')
    args = parser.parse_args()

    # Update globals from args
    DEFAULT_MODEL_NAME = args.model
    API_CALL_DELAY = args.delay
    RETRY_DELAY = args.retry_delay
    MAX_API_RETRIES = args.max_retries

    # Setup logging
    log_file = setup_logging(OUTPUT_REPORT_DIR)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("DEBUG logging enabled.")

    logging.info(f"Script started. Log file: {log_file}")
    logging.info(f"Base data directory: {BASE_DATA_DIR}")
    logging.info(f"Using model: {args.model}")
    logging.info(f"Max attempts per file: {args.max_retries}")
    logging.info(f"Overwrite existing files: {args.force}")
    logging.info(f"Delay between files: {args.delay}s")
    logging.info(f"Delay between retries: {args.retry_delay}s")

    if not API_KEY:
        logging.error("GOOGLE_API_KEY not found. Please set it in your .env file.")
        exit(1)

    try:
        # Configure Gemini API
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(args.model)
        logging.info(f"Successfully configured Gemini model: {args.model}")
        
        # Process all Bible segments
        process_result = process_bible_segments(
            INPUT_DIR, 
            OUTPUT_DIR, 
            model, 
            force_overwrite=args.force
        )
        
        if process_result:
            summary, *counts = process_result
            generate_reports(summary, OUTPUT_REPORT_DIR, counts)
        else:
            logging.error("Processing function failed critically. No reports generated.")
            
    except Exception as e:
        logging.exception(f"Critical error in main execution: {e}")
        
    logging.info("Script finished.")