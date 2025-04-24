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
import re # Import regex

# --- Configuration ---

# Load environment variables (especially GOOGLE_API_KEY)
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Define base data directory relative to the script location
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DATA_DIR = SCRIPT_DIR.parent / 'data' # Assumes script is in 'code/sentiment-arabic-news'

NEWS_LANG = 'en' # select from ['ar','en']

if NEWS_LANG == 'ar':
    # Define input and output directories based on the structure
    INPUT_SUBDIR_REL_PATHS = [
        'bbc_arabic/article_download',
        'cnn_arabic/article_download'
    ]

    OUTPUT_SUBDIR_REL_PATHS = [
        'bbc_arabic/analysis_json',
        'cnn_arabic/analysis_json'
    ]
elif NEWS_LANG == 'en':
    # Define input and output directories based on the structure
    INPUT_SUBDIR_REL_PATHS = [
        # 'bbc_english/article_download',
        'cnn_english/article_download'
    ]

    OUTPUT_SUBDIR_REL_PATHS = [
        # 'bbc_english/analysis_json',
        'cnn_english/analysis_json'
    ]
else:
    raise ValueError(f"Unsupported news language: {NEWS_LANG}")

OUTPUT_REPORT_DIR = BASE_DATA_DIR / 'metaanalysis'

# --- Model and Retry Configuration ---
DEFAULT_MODEL_NAME = "gemini-2.0-flash"
MAX_API_RETRIES = 3 # Total number of attempts (1 initial + 2 retries)
RETRY_DELAY = 5 # Seconds to wait between retries

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
METADATA_TITLE_MARKER = "### English Title:"
METADATA_SUMMARY_MARKER = "### English Text Summary:"
METADATA_END_MARKER = "---" # Marker separating metadata from main analysis
JSON_EXTRACTION_MARKER = "Final Results for JSON Extraction:"


# --- Logging Setup ---

def setup_logging(log_dir):
    """Sets up logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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

# --- Prompt Template --- (Keep existing)
PROMPT_TEMPLATE = """
###NEWS_TEXT:

{news_text}

###INSTRUCTIONS:

Carefully think step by step to analyze the text of the provided article.

If the article is not in English, provide English translations for the article's title and a concise summary of its main points. Use the following format exactly:
### English Title:
[Provide the English translation of the original article's title here]

### English Text Summary:
[Provide a concise English summary (approx. 100-150 words) of the main points of the article text here]

---
Now, your main task is to identify all the distinct topics discussed in the original text, including but not limited to issues, people, policies, countries, and recent events. For each identified topic, conduct a sentiment analysis and an emotional analysis based on Plutchik's eight basic emotions. Provide the following detailed assessment for each topic:

Sentiment Polarity Analysis:
Reasoning: Give a concise rationale for the assigned Polarity Score and Confidence values. Include relevant fragments from the original text (especially if Arabic). Note linguistic subtleties. Explain *why* the score and confidence were chosen.
Polarity Score: Assign a float value -1.0 to 1.0 (use 99.9 if unknown).
Confidence: Provide an integer percentage 0-100 (use 0 if unknown).

Emotional Weights Analysis:
Reasoning: Give a concise rationale for the assigned emotional weight and confidence values for *each* emotion. Reference specific parts of the text (especially if Arabic). Explain *why* the weight and confidence were chosen for each emotion.
Emotion Dictionary: For each of Plutchik's eight emotions (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation), provide:
Weight: A float value 0.0 to 1.0 (use 99.9 if unknown).
Confidence: An integer percentage 0-100 (use 0 if unknown).

Consider context, tone, perspective, bias, ambiguity. Reflect uncertainty in confidence scores. Include all topics even if info is insufficient (use 99.9 score/weight, 0% confidence).

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

# --- Gemini API Call Function --- (Keep existing from previous version)
def call_gemini_api(model, text_content, attempt_num):
    """Calls the Gemini API with the given text and prompt template."""
    prompt = PROMPT_TEMPLATE.format(news_text=text_content)
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
             # Decide if empty response is an error - let's treat it as potentially retryable
             raise ValueError("API returned empty text response.")

        logging.info(f"Attempt {attempt_num}: Received response from Gemini API (Length: {len(api_result_text)}).")
        return api_result_text

    except Exception as e:
        logging.error(f"Attempt {attempt_num}: Error calling Gemini API: {e}")
        raise ValueError(f"Gemini API call failed: {e}") from e


def parse_api_response(api_response_text, attempt_num):
    """
    Parses the API response text to extract metadata (title, summary)
    and the topic analysis JSON list. Validates the structure of the
    topic analysis list according to predefined criteria.

    Returns:
        tuple: (metadata_dict, topic_list)
    Raises:
        ValueError: If essential parsing or validation fails.
    """
    logging.debug(f"Attempt {attempt_num}: Parsing API response (length {len(api_response_text)})...")
    metadata_dict = {'title_en': None, 'text_summary_en': None}
    topic_list = None
    log_snippet_length = 300

    # --- Extract Metadata ---
    try:
        # (Keep the metadata extraction logic from the previous version - it worked)
        title_marker_idx = api_response_text.find(METADATA_TITLE_MARKER)
        summary_marker_idx = api_response_text.find(METADATA_SUMMARY_MARKER)
        end_marker_idx = -1
        if summary_marker_idx != -1:
             end_marker_idx = api_response_text.find(METADATA_END_MARKER, summary_marker_idx + len(METADATA_SUMMARY_MARKER))
        json_marker_idx = api_response_text.find(JSON_EXTRACTION_MARKER)

        logging.debug(f"Marker indices - Title: {title_marker_idx}, Summary: {summary_marker_idx}, End after Summary: {end_marker_idx}, JSON: {json_marker_idx}")

        title_regex = rf"^{METADATA_TITLE_MARKER}\s*(.*?)(?=\s*(?:{METADATA_SUMMARY_MARKER}|{JSON_EXTRACTION_MARKER}|$))"
        title_match = re.search(title_regex, api_response_text, re.MULTILINE | re.DOTALL)

        if title_match:
            extracted_title = title_match.group(1).strip()
            if extracted_title: metadata_dict['title_en'] = extracted_title
            else: metadata_dict['title_en'] = "" # Explicitly empty if captured nothing

        summary_regex = rf"^{METADATA_SUMMARY_MARKER}\s*(.*?)(?=\s*(?:{METADATA_END_MARKER}|$))"
        summary_match = re.search(summary_regex, api_response_text, re.MULTILINE | re.DOTALL)

        if summary_match:
             extracted_summary = summary_match.group(1).strip()
             if extracted_summary: metadata_dict['text_summary_en'] = extracted_summary
             else: metadata_dict['text_summary_en'] = "" # Explicitly empty

        title_found = metadata_dict.get('title_en') is not None
        summary_found = metadata_dict.get('text_summary_en') is not None
        logging.info(f"Attempt {attempt_num}: Metadata parsing result -> Title Found: {title_found} (Value: '{str(metadata_dict.get('title_en'))[:50]}...'), Summary Found: {summary_found} (Value: '{str(metadata_dict.get('text_summary_en'))[:50]}...')")

    except Exception as e:
        logging.exception(f"Attempt {attempt_num}: Unexpected error during metadata extraction: {e}")
        metadata_dict = {'title_en': None, 'text_summary_en': None}


    # --- Extract Topic Analysis JSON List ---
    raw_response_for_debug = api_response_text
    json_string = None
    json_string_cleaned = None
    parsed_json_object = None # Variable to hold the intermediate parsed object

    try:
        # (Keep the JSON extraction logic - finding marker/fallback, parsing)
        logging.debug(f"Attempt {attempt_num}: Looking for JSON marker: '{JSON_EXTRACTION_MARKER}'")
        json_marker_index = raw_response_for_debug.find(JSON_EXTRACTION_MARKER)

        if json_marker_index == -1:
            logging.warning(f"Attempt {attempt_num}: Marker '{JSON_EXTRACTION_MARKER}' not found. Attempting fallback.")
            list_start_index = raw_response_for_debug.rfind('[')
            if list_start_index == -1: raise ValueError("JSON Marker not found and fallback failed to find '[' for topic list.")
            json_string = raw_response_for_debug[list_start_index:]
        else:
            logging.info(f"Attempt {attempt_num}: JSON Marker '{JSON_EXTRACTION_MARKER}' found at index {json_marker_index}.")
            list_start_index = raw_response_for_debug.find('[', json_marker_index + len(JSON_EXTRACTION_MARKER))
            if list_start_index == -1: raise ValueError("JSON list start '[' not found after JSON marker.")
            json_string = raw_response_for_debug[list_start_index:]

        open_brackets = 0; list_end_index = -1
        for i, char in enumerate(json_string):
            if char == '[': open_brackets += 1
            elif char == ']': open_brackets -= 1;
            if open_brackets == 0 and list_end_index == -1 : list_end_index = i + 1; # Set only once
        # Break early if needed, e.g. if open_brackets goes negative or i gets too large

        if list_end_index == -1: raise ValueError("Matching JSON list end ']' not found.")

        json_string_cleaned = json_string[:list_end_index].strip()
        if json_string_cleaned.startswith("```json"): json_string_cleaned = json_string_cleaned[7:].strip()
        if json_string_cleaned.endswith("```"): json_string_cleaned = json_string_cleaned[:-3].strip()
        if not json_string_cleaned: raise ValueError("Cleaned JSON string for topic list is empty.")

        logging.debug(f"Attempt {attempt_num}: Calling json.loads for topic list...")
        parsed_json_object = json.loads(json_string_cleaned) # Store intermediate result
        logging.info(f"Attempt {attempt_num}: Successfully parsed JSON object (type: {type(parsed_json_object)}). Validating structure...")

        # --- NEW: Detailed Validation Logic ---
        min_topics_required = 2 # As requested by user
        required_sentiment_keys = {"score", "confidence", "reasoning"}
        required_emotion_keys = {"Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation"}
        required_emotion_value_keys = {"weight", "confidence", "reasoning"}

        # 1. Check if it's a list
        if not isinstance(parsed_json_object, list):
            logging.error(f"Validation Error: Parsed JSON is not a list, but type {type(parsed_json_object)}. Content: {str(parsed_json_object)[:200]}")
            raise ValueError("Malformed topic_analysis structure: Parsed result is not a list.")

        # Assign to topic_list now that we know it's a list
        topic_list = parsed_json_object

        # 2. Check minimum number of topics
        if len(topic_list) < min_topics_required:
            logging.error(f"Validation Error: Found only {len(topic_list)} topics, required at least {min_topics_required}.")
            logging.error(f"Topics found: {topic_list}") # Log the insufficient list
            raise ValueError(f"Malformed topic_analysis structure: Less than {min_topics_required} topics found.")

        # 3. Check the structure of the first topic object in the list
        logging.debug(f"Validating structure of first topic object (out of {len(topic_list)})...")
        first_topic = topic_list[0] # Safe now because we checked length >= min_topics_required (which is >= 1)

        if not isinstance(first_topic, dict):
            logging.error(f"Validation Error: First item in topic_analysis list is not a dictionary (type: {type(first_topic)}). Item: {str(first_topic)[:200]}")
            raise ValueError("Malformed topic_analysis structure: Topic item is not a dictionary.")

        # Check main keys
        required_main_keys = {"topic", "sentiment_polarity", "emotions"}
        if not required_main_keys.issubset(first_topic.keys()):
            missing_keys = required_main_keys - first_topic.keys()
            logging.error(f"Validation Error: First topic object missing required keys: {missing_keys}. Found keys: {list(first_topic.keys())}")
            raise ValueError(f"Malformed topic_analysis structure: Missing keys {missing_keys} in topic object.")

        # Check sentiment_polarity structure
        sentiment_data = first_topic.get("sentiment_polarity")
        if not isinstance(sentiment_data, dict):
             logging.error(f"Validation Error: sentiment_polarity is not a dictionary (type: {type(sentiment_data)}).")
             raise ValueError("Malformed topic_analysis structure: sentiment_polarity is not a dictionary.")
        if not required_sentiment_keys.issubset(sentiment_data.keys()):
             missing_keys = required_sentiment_keys - sentiment_data.keys()
             logging.error(f"Validation Error: sentiment_polarity missing keys: {missing_keys}. Found keys: {list(sentiment_data.keys())}")
             raise ValueError(f"Malformed topic_analysis structure: Missing keys {missing_keys} in sentiment_polarity.")

        # Check emotions structure
        emotions_data = first_topic.get("emotions")
        if not isinstance(emotions_data, dict):
             logging.error(f"Validation Error: emotions is not a dictionary (type: {type(emotions_data)}).")
             raise ValueError("Malformed topic_analysis structure: emotions is not a dictionary.")
        if not required_emotion_keys.issubset(emotions_data.keys()):
             missing_keys = required_emotion_keys - emotions_data.keys()
             logging.error(f"Validation Error: emotions missing required emotion keys: {missing_keys}. Found keys: {list(emotions_data.keys())}")
             raise ValueError(f"Malformed topic_analysis structure: Missing emotion keys {missing_keys}.")

        # Check structure within one emotion (e.g., first required key)
        first_emotion_key = list(required_emotion_keys)[0]
        emotion_value_data = emotions_data.get(first_emotion_key)
        if not isinstance(emotion_value_data, dict):
             logging.error(f"Validation Error: Value for emotion '{first_emotion_key}' is not a dictionary (type: {type(emotion_value_data)}).")
             raise ValueError(f"Malformed topic_analysis structure: Emotion value for {first_emotion_key} is not a dictionary.")
        if not required_emotion_value_keys.issubset(emotion_value_data.keys()):
             missing_keys = required_emotion_value_keys - emotion_value_data.keys()
             logging.error(f"Validation Error: Emotion value for '{first_emotion_key}' missing keys: {missing_keys}. Found keys: {list(emotion_value_data.keys())}")
             raise ValueError(f"Malformed topic_analysis structure: Missing keys {missing_keys} in emotion value for {first_emotion_key}.")

        # If all checks passed for the first item
        logging.info(f"Attempt {attempt_num}: Successfully validated topic_analysis structure for the first topic object.")
        # --- Validation Passed ---

    except json.JSONDecodeError as e:
        logging.error(f"Attempt {attempt_num}: JSON Decode Error for topic list: {e}")
        logging.error(f"Problematic cleaned JSON string snippet: {(json_string_cleaned or '')[:500]}...")
        raise ValueError(f"Failed to decode JSON topic list: {e}") from e
    except ValueError as e: # Catch JSON extraction or validation errors
        logging.error(f"Attempt {attempt_num}: Error during JSON topic list processing/validation: {e}")
        raise # Re-raise to be caught by retry loop
    except Exception as e:
        logging.exception(f"Attempt {attempt_num}: Unexpected error during JSON topic list processing/validation: {e}")
        raise ValueError(f"Unexpected JSON processing/validation error: {e}") from e

    # --- Final Return ---
    # If we reach here, topic_list should be a validated list
    if isinstance(topic_list, list):
        logging.info(f"Attempt {attempt_num}: Returning parsed metadata and validated topic list.")
        final_metadata = {'title_en': metadata_dict.get('title_en'), 'text_summary_en': metadata_dict.get('text_summary_en')}
        return final_metadata, topic_list
    else:
        # Should be unreachable if validation logic is correct, but acts as safeguard
        logging.error(f"Attempt {attempt_num}: Code reached end of function but topic_list is not a list (Type: {type(topic_list)}). This indicates a logic error.")
        raise ValueError("Internal logic error: Failed to obtain valid topic list.")


# --- Main Processing Function ---

def process_files(input_dirs, output_dirs, model, force_overwrite=False):
    """Iterates through input directories, processes files with retries, and saves results."""
    processing_summary = []
    total_files = 0
    success_count = 0
    skipped_count = 0
    error_count = 0
    api_call_counter = 0 # Initialize API call sequence counter

    if len(input_dirs) != len(output_dirs):
        logging.error("Mismatch between input and output directory lists. Exiting.")
        return None

    for input_rel_path, output_rel_path in zip(input_dirs, output_dirs):
        input_dir = BASE_DATA_DIR / input_rel_path
        output_dir = BASE_DATA_DIR / output_rel_path

        if not input_dir.is_dir():
            logging.warning(f"Input directory not found, skipping: {input_dir}")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Processing files from: {input_dir}")
        logging.info(f"Saving results to: {output_dir}")

        article_files = sorted(list(input_dir.glob('*.txt')))
        if not article_files:
             logging.warning(f"No *.txt files found in {input_dir}")
             continue

        total_files += len(article_files)

        for input_filepath in article_files:
            status = "pending"
            final_error_msg = None
            result_data = None
            processed_successfully = False
            final_output_filepath = None # Initialize

            base_output_filename = input_filepath.stem + ".json"
            logging.info(f"--- Checking Input: {input_filepath.name} (Base Output: {base_output_filename}) ---")

            # --- Check for existing output file (Restartability) ---
            # Use glob to find any file matching *_{base_output_filename}
            existing_files = list(output_dir.glob(f"*_{base_output_filename}"))
            if existing_files and not force_overwrite:
                # Log the first existing file found
                logging.info(f"Output file already exists (e.g., {existing_files[0].name}), skipping.")
                skipped_count += 1
                status = "skipped_exists"
                processing_summary.append({
                    "input_file": str(input_filepath.relative_to(BASE_DATA_DIR)),
                    "output_file": str(existing_files[0].relative_to(BASE_DATA_DIR)), # Report existing file
                    "status": status,
                    "error": None
                })
                continue # Skip to next input file

            # --- Retry Loop ---
            for attempt in range(1, MAX_API_RETRIES + 1):
                try:
                    if attempt > 1:
                        logging.info(f"Waiting {RETRY_DELAY} seconds before retry attempt {attempt} for {input_filepath.name}...")
                        time.sleep(RETRY_DELAY)

                    logging.debug(f"Attempt {attempt}: Reading file {input_filepath.name}")
                    with open(input_filepath, 'r', encoding='utf-8') as f:
                        article_text = f.read()

                    if not article_text.strip():
                         logging.error(f"Input file {input_filepath.name} is empty. Skipping.")
                         final_error_msg = "Input file is empty."
                         status = "error_empty_input"; break

                    api_response = call_gemini_api(model, article_text, attempt)
                    metadata, topic_list = parse_api_response(api_response, attempt) # Use new parsing function

                    # Check if parsing was successful (topic_list must be a list)
                    if not isinstance(topic_list, list):
                        # Should have been caught by parse_api_response, but double-check
                        raise ValueError("Parsing failed to return a valid topic list.")

                    # --- Success for this attempt ---
                    api_call_counter += 1 # Increment SUCCESSFUL API call counter
                    logging.info(f"Successfully processed {input_filepath.name} on attempt {attempt} (API Call #{api_call_counter}).")

                    # Construct the final combined data structure
                    result_data = {
                        "article_metadata": {
                            "title_en": metadata.get('title_en'),
                            "text_summary_en": metadata.get('text_summary_en'),
                            "original_filename": input_filepath.name # Add original filename
                        },
                        "topic_analysis": topic_list
                    }

                    # Construct final output path using the counter
                    final_output_filename = f"{api_call_counter}_{base_output_filename}"
                    final_output_filepath = output_dir / final_output_filename

                    processed_successfully = True
                    status = "success"
                    final_error_msg = None
                    break # Exit retry loop

                except ValueError as ve: # Catch potentially retryable errors (API or parsing)
                    logging.warning(f"Attempt {attempt}/{MAX_API_RETRIES} failed for {input_filepath.name}: {ve}")
                    final_error_msg = f"Attempt {attempt}: {ve}"
                    if attempt == MAX_API_RETRIES:
                        logging.error(f"Failed to process {input_filepath.name} after {MAX_API_RETRIES} attempts.")
                        status = "error_retries_failed"

                except FileNotFoundError:
                    logging.error(f"Input file not found during processing attempt: {input_filepath}")
                    final_error_msg = "Input file not found during processing."; status = "error_file_not_found"; break
                except Exception as e:
                    logging.exception(f"Unexpected error processing {input_filepath.name} on attempt {attempt}: {e}")
                    final_error_msg = f"Unexpected error: {e}"; status = "error_unexpected"; break

            # --- After Retry Loop ---
            if processed_successfully and result_data is not None and final_output_filepath is not None:
                try:
                    with open(final_output_filepath, 'w', encoding='utf-8') as f:
                        json.dump(result_data, f, ensure_ascii=False, indent=2)
                    logging.info(f"Successfully saved result: {final_output_filepath.name}")
                    success_count += 1
                except Exception as e:
                    logging.exception(f"Error writing JSON output {final_output_filepath.name}: {e}")
                    status = "error_writing_output"; final_error_msg = f"Failed to write JSON file: {e}"
                    processed_successfully = False # Mark as overall failure
                    # Decrement counter since write failed? No, API call was successful. Leave counter.

            # Update counts if file ended in failure
            if not processed_successfully:
                 error_count += 1

            # Append result for this file to the summary
            processing_summary.append({
                "input_file": str(input_filepath.relative_to(BASE_DATA_DIR)),
                "output_file": str(final_output_filepath.relative_to(BASE_DATA_DIR)) if final_output_filepath else None,
                "status": status,
                "error": final_error_msg
            })

            # Delay between processing different files
            if status != "skipped_exists":
                 logging.debug(f"Waiting {API_CALL_DELAY}s before next file...")
                 time.sleep(API_CALL_DELAY)

    logging.info("--- Processing Complete ---")
    logging.info(f"Total files found: {total_files}")
    logging.info(f"Successfully processed and written: {success_count}")
    logging.info(f"Skipped (output existed): {skipped_count}")
    logging.info(f"Errors (final): {error_count}")

    return processing_summary, total_files, success_count, skipped_count, error_count

# --- Reporting Function --- (Keep existing, maybe add counter to metadata)
def generate_reports(summary_data, report_dir, totals):
    """Generates human-readable and machine-readable reports."""
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_base_name = f"topic_sa_report_{timestamp}"
    txt_report_path = report_dir / f"{report_base_name}.txt"
    json_report_path = report_dir / f"{report_base_name}.json"

    total_files, success_count, skipped_count, error_count = totals

    # Generate Text Report (minor update for clarity)
    try:
        with open(txt_report_path, 'w', encoding='utf-8') as f:
            f.write("--- Topic Sentiment Analysis Processing Report ---\n")
            f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Using Model: {DEFAULT_MODEL_NAME}\n")
            f.write(f"Max Attempts per File: {MAX_API_RETRIES}\n")
            f.write("\n--- Summary ---\n")
            f.write(f"Total files scanned: {total_files}\n")
            f.write(f"Successfully processed & written: {success_count}\n") # Clarify written
            f.write(f"Skipped (output existed): {skipped_count}\n")
            f.write(f"Errors encountered (final): {error_count}\n")
            f.write("\n--- Detailed Log ---\n")
            if summary_data:
                 max_in = max(len(item.get("input_file", "")) for item in summary_data) if summary_data else 0
                 max_status = max(len(item.get("status", "")) for item in summary_data) if summary_data else 0
                 max_out = max(len(Path(item.get("output_file", "")).name) for item in summary_data if item.get("output_file")) if any(item.get("output_file") for item in summary_data) else 0


                 for item in summary_data:
                     in_file = item.get("input_file", "N/A").ljust(max_in)
                     status = item.get("status", "N/A").ljust(max_status)
                     out_file_name = Path(item.get("output_file", "N/A")).name.ljust(max_out) # Use just name for alignment
                     error = item.get("error", "")
                     f.write(f"Input: {in_file} | Status: {status} | Output: {out_file_name}")
                     if error:
                         f.write(f" | Last Error: {error}\n")
                     else:
                         f.write("\n")
            else:
                 f.write("No files were processed or summary data is unavailable.\n")

        logging.info(f"Text report generated: {txt_report_path}")
    except Exception as e:
        logging.error(f"Failed to generate text report: {e}")

    # Generate JSON Report (minor update)
    try:
        report_json_data = {
            "report_metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "model_used": DEFAULT_MODEL_NAME,
                "max_attempts": MAX_API_RETRIES,
                "total_files_scanned": total_files,
                "success_count": success_count, # Files successfully processed AND written
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


# --- Main Execution --- (Keep existing, args are fine)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process news articles using Gemini for topic/sentiment/emotion analysis.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing JSON output files.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help=f"Name of the Gemini model to use (default: {DEFAULT_MODEL_NAME}).")
    parser.add_argument("--delay", type=float, default=API_CALL_DELAY, help=f"Delay in seconds between processing different files (default: {API_CALL_DELAY}).")
    parser.add_argument("--retry-delay", type=float, default=RETRY_DELAY, help=f"Delay in seconds between retries for the *same* file (default: {RETRY_DELAY}).")
    parser.add_argument("--max-retries", type=int, default=MAX_API_RETRIES, help=f"Total maximum attempts per file (1 initial + N retries) (default: {MAX_API_RETRIES}).")
    parser.add_argument('--debug', action='store_true', help='Enable debug logging.')
    args = parser.parse_args()

    # Update globals from args
    DEFAULT_MODEL_NAME = args.model
    API_CALL_DELAY = args.delay
    RETRY_DELAY = args.retry_delay
    MAX_API_RETRIES = args.max_retries

    log_file = setup_logging(OUTPUT_REPORT_DIR)
    if args.debug: logging.getLogger().setLevel(logging.DEBUG); logging.info("DEBUG logging enabled.")

    logging.info(f"Script started. Log file: {log_file}")
    logging.info(f"Base data directory: {BASE_DATA_DIR}")
    logging.info(f"Using model: {args.model}")
    logging.info(f"Max attempts per file: {args.max_retries}")
    logging.info(f"Overwrite existing files: {args.force}")
    logging.info(f"Delay between files: {args.delay}s")
    logging.info(f"Delay between retries: {args.retry_delay}s")

    if not API_KEY: logging.error("GOOGLE_API_KEY not found. Please set it."); exit(1)

    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(args.model)
        logging.info(f"Successfully configured Gemini model: {args.model}")
    except Exception as e:
        logging.error(f"Failed to configure Gemini API: {e}"); exit(1)

    process_result = process_files( INPUT_SUBDIR_REL_PATHS, OUTPUT_SUBDIR_REL_PATHS, model, force_overwrite=args.force)

    if process_result:
         summary, *counts = process_result
         generate_reports(summary, OUTPUT_REPORT_DIR, counts)
    else:
         logging.error("Processing function failed critically. No reports generated.")

    logging.info("Script finished.")