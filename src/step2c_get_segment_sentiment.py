#!/usr/bin/env python3
"""
step2a_get_segment_sentiment.py - Performs sentiment analysis on segmented Ancient Greek text.

This script:
1. Reads an Ancient Greek text file and segments it
2. Processes segments in batches for API efficiency
3. Performs sentiment analysis on each segment via Gemini API
4. Saves results to a CSV file
5. Supports restarting from where it left off
"""

import os
import re
import csv
import json
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dotenv import load_dotenv
import google.generativeai as genai

# --- Configuration ---

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Define script directory and paths
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DATA_DIR = SCRIPT_DIR.parent / 'data'
CORPUS_NAME = "bible_ancient-greek"

# Input and output file paths
INPUT_FILE = BASE_DATA_DIR / CORPUS_NAME / "document" / f"{CORPUS_NAME}.txt"
OUTPUT_FILE = BASE_DATA_DIR / CORPUS_NAME / f"{CORPUS_NAME}-sentiment.csv"

# Processing configuration
SEGMENT_SEP_DEFAULT = r"[\n]{2,}"  # Regex for segment separation
MIN_SEGMENT_CHAR = 110 # Ancient Greek Bible: round(17.4 words/sent * 5.4 char/word + 16 spaces)
BATCH_SIZE = 10  # Number of segments to process in each API call
MAX_API_RETRIES = 3  # Maximum number of API retry attempts
FLAG_FULL_RESTART = False  # Set to True to ignore existing output and start from scratch

# API configuration
DEFAULT_MODEL_NAME = "gemini-2.0-flash"
RETRY_DELAY = 5  # Seconds to wait between retries
API_CALL_DELAY = 2  # Seconds to wait between distinct API calls

# Gemini API Configuration
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
# Update GENERATION_CONFIG with more conservative settings
GENERATION_CONFIG = {
    "temperature": 0.1,  # Lower temperature for more consistent outputs
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",  # Request JSON response format
}

# CSV Column headers
CSV_HEADERS = ["segment_no", "text_original", "text_en", "polarity", "confidence_percent", "reasoning_en"]

# --- Logging Setup ---

def setup_logging():
    """Sets up logging to file and console."""
    log_dir = BASE_DATA_DIR / CORPUS_NAME / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"sentiment_analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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

PROMPT_TEMPLATE = """
###INSTRUCTIONS:

You are analyzing a batch of Ancient Greek text segments. For each segment, carefully translate it to English and analyze its sentiment.

Return your analysis in a valid JSON format with the following structure for EACH segment in the batch:

You are analyzing a batch of Ancient Greek text segments. For each segment, carefully translate it to English and analyze its sentiment.

Return your analysis in a valid JSON format with the following structure for EACH segment in the batch:

{{
"segment_no": segment_number,
"text_original": "original_greek_text",
"text_en": "Your English translation of the text",
"polarity": float_value,
"confidence_percent": int_value,
"reasoning_en": "Your reasoning for the assigned polarity"
}}

EXTREMELY IMPORTANT FOR JSON VALIDITY:
1. All string values MUST be properly escaped, especially when they contain quotes, newlines, or Greek characters
2. Never include unescaped newlines within string values
3. All quotes within strings must be escaped with a backslash
4. Do not truncate or modify the original Greek text
5. The JSON must be valid according to RFC 8259 standards
6. Do not include any explanation or comments outside the JSON structure

###TEXT SEGMENTS:

{segments_json}

###OUTPUT FORMAT:
Return ONLY a valid JSON object where keys are segment numbers and values are the analysis objects with fields as described above. Example:

{{
  "{example_segment_no1}": {{
    "segment_no": {example_segment_no1},
    "text_original": "{example_text1}",
    "text_en": "English translation of segment {example_segment_no1}",
    "polarity": 0.7,
    "confidence_percent": 85,
    "reasoning_en": "This segment expresses joy and positive divine revelation."
  }},
  "{example_segment_no2}": {{
    "segment_no": {example_segment_no2},
    "text_original": "{example_text2}",
    "text_en": "English translation of segment {example_segment_no2}",
    "polarity": -0.4,
    "confidence_percent": 75,
    "reasoning_en": "This segment describes suffering and hardship."
  }}
}}

I need complete analysis for ALL segments provided. Ensure your output is valid JSON that can be parsed with json.loads().
DO NOT INCLUDE ANY TEXT, EXPLANATION, OR CODE BLOCKS OUTSIDE THE JSON STRUCTURE.
"""

# --- Helper Functions ---

def read_input_file(file_path: Path) -> str:
    """Read the input file with improved handling for Ancient Greek text."""
    encodings_to_try = ['utf-8-sig', 'utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'iso-8859-7', 'windows-1253']
    
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
                
                # Check if content looks valid (contains Greek characters)
                if re.search(r'[\u0370-\u03FF\u1F00-\u1FFF]', content):
                    logging.info(f"Successfully read file using {encoding} encoding")
                    
                    # Remove any Byte Order Mark if present
                    if content.startswith('\ufeff'):
                        content = content[1:]
                        logging.info("Removed BOM character from text")
                    
                    return content
        except UnicodeDecodeError:
            continue
    
    # If all encodings fail
    raise ValueError(f"Could not decode file {file_path} with any of the attempted encodings")


def segment_text(text: str, pattern: str, flag_seg_rule=None) -> List[str]:
    """
    Split the text into segments based on the specified pattern or segmentation rule.
    
    Args:
        text (str): The input text to segment
        pattern (str): The regex pattern to use for splitting (default behavior)
        flag_seg_rule (Union[int, str], optional): 
            - If int: Minimum number of characters for each segment
            - If str: Custom regex pattern to use instead of pattern parameter
    
    Returns:
        List[str]: A list of non-empty text segments
    """
    segments = []
    
    # Case 1: Using a minimum character count rule
    if isinstance(flag_seg_rule, int) and flag_seg_rule > 0:
        min_chars = flag_seg_rule
        current_segment = ""
        words = text.split()
        
        for word in words:
            # Try adding this word to the current segment
            potential_segment = current_segment + " " + word if current_segment else word
            
            # If adding this word would exceed the minimum length and we already have content,
            # save the current segment and start a new one
            if len(current_segment) >= min_chars and len(potential_segment) > min_chars:
                segments.append(current_segment.strip())
                current_segment = word
            else:
                # Otherwise, add the word to the current segment
                current_segment = potential_segment
        
        # Add the last segment if it's not empty
        if current_segment:
            segments.append(current_segment.strip())
    
    # Case 2: Using a custom regex pattern provided in flag_seg_rule
    elif isinstance(flag_seg_rule, str) and flag_seg_rule.startswith('r\'') and flag_seg_rule.endswith('\''):
        # Extract the actual regex pattern from the string representation
        # (removing the r'' wrapper)
        regex_pattern = flag_seg_rule[2:-1]
        segments = re.split(regex_pattern, text)
    
    # Case 3: Default behavior - use the pattern parameter
    else:
        segments = re.split(pattern, text)
    
    # Filter out empty segments in all cases
    return [segment.strip() for segment in segments if segment.strip()]

def batch_segments(segments: List[str], batch_size: int) -> List[List[Tuple[int, str]]]:
    """Group segments into batches for processing."""
    batches = []
    current_batch = []
    
    for i, segment in enumerate(segments, 1):  # Start segment numbering from 1
        current_batch.append((i, segment))
        
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []
    
    # Add the last batch if it's not empty
    if current_batch:
        batches.append(current_batch)
        
    return batches

def find_last_processed_segment(output_file: Path) -> Tuple[int, Optional[str]]:
    """
    Check the output CSV to find the last successfully processed segment.
    Returns the segment number and its text to verify against input file.
    """
    if not output_file.exists():
        return 0, None
        
    last_segment_no = 0
    last_original_text = None
    
    try:
        with open(output_file, 'r', encoding='utf-8', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            
            for row in reader:
                try:
                    segment_no = int(row.get('segment_no', 0))
                    text_original = row.get('text_original', '')
                    
                    # Skip rows with empty text_original (error cases)
                    if text_original.strip():
                        if segment_no > last_segment_no:
                            last_segment_no = segment_no
                            last_original_text = text_original
                except ValueError:
                    # Skip rows with invalid segment numbers
                    continue
    except Exception as e:
        logging.error(f"Error reading output CSV: {e}")
        return 0, None
        
    return last_segment_no, last_original_text

def validate_restart_point(segments: List[str], last_segment_no: int, last_text: Optional[str]) -> bool:
    """
    Validates that the restart point matches the input file.
    Returns True if valid, False if mismatch detected.
    """
    if last_segment_no == 0 or last_text is None:
        return True
        
    # Check if the segment number is within range
    if last_segment_no > len(segments):
        logging.error(f"Last segment number {last_segment_no} exceeds total segments {len(segments)}")
        return False
        
    # Verify the text content matches
    expected_segment_text = segments[last_segment_no - 1]  # Adjust for 0-based indexing
    
    # Compare with some flexibility (ignoring whitespace differences)
    last_text_normalized = ' '.join(last_text.split())
    expected_text_normalized = ' '.join(expected_segment_text.split())
    
    # Check if the normalized texts match approximately (first 100 chars)
    last_text_start = last_text_normalized[:min(100, len(last_text_normalized))]
    expected_text_start = expected_text_normalized[:min(100, len(expected_text_normalized))]
    
    if last_text_start != expected_text_start:
        logging.error(f"Content mismatch at segment {last_segment_no}. Input file appears different from previous processing.")
        logging.error(f"Expected: {expected_text_start}")
        logging.error(f"Found in CSV: {last_text_start}")
        return False
        
    return True

def prepare_segments_json(batch: List[Tuple[int, str]]) -> str:
    """Prepare the segments as a JSON object with proper escaping for the prompt."""
    segments_dict = {}
    
    for segment_no, text in batch:
        # Clean and escape the text to ensure it doesn't break JSON
        cleaned_text = text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
        # Truncate extremely long segments to reduce complexity
        if len(cleaned_text) > 500:
            logging.warning(f"Truncating segment {segment_no} from {len(cleaned_text)} to 500 chars")
            cleaned_text = cleaned_text[:500] + '...'
        
        segments_dict[str(segment_no)] = cleaned_text
    
    return json.dumps(segments_dict, ensure_ascii=False)

def prepare_prompt(batch: List[Tuple[int, str]]) -> str:
    """Prepare a more robust prompt with better handling of Greek text."""
    # Clean the text segments to ensure they don't break JSON
    cleaned_batch = []
    for segment_no, text in batch:
        # Remove any control characters
        clean_text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        # Escape backslashes and quotes that might break JSON
        clean_text = clean_text.replace('\\', '\\\\').replace('"', '\\"')
        cleaned_batch.append((segment_no, clean_text))
    
    segments_json = prepare_segments_json(cleaned_batch)
    
    # Use the first two segments as examples (if available)
    example_segment_no1 = cleaned_batch[0][0] if len(cleaned_batch) > 0 else 1
    example_text1 = cleaned_batch[0][1][:20] if len(cleaned_batch) > 0 else "Example text 1"
    
    example_segment_no2 = cleaned_batch[1][0] if len(cleaned_batch) > 1 else 2
    example_text2 = cleaned_batch[1][1][:20] if len(cleaned_batch) > 1 else "Example text 2"
    
    # Modify prompt to explicitly request cleaner JSON output
    modified_prompt = PROMPT_TEMPLATE.format(
        segments_json=segments_json,
        example_segment_no1=example_segment_no1,
        example_text1=example_text1,
        example_segment_no2=example_segment_no2,
        example_text2=example_text2
    )
    
    # Add additional instructions for clean JSON
    modified_prompt += "\n\nIMPORTANT: Ensure all strings in your JSON output are properly escaped, especially when they contain quotes, newlines, or special characters. The JSON must be strictly valid and parseable."
    
    return modified_prompt

def call_gemini_api(model, prompt: str, attempt_num: int) -> str:
    """Call the Gemini API with the prepared prompt."""
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
            if hasattr(response, 'prompt_feedback'):
                logging.error(f"Attempt {attempt_num}: Prompt Feedback: {response.prompt_feedback}")
            raise ValueError("API response was empty or malformed (no candidates).")
            
        api_result_text = ""
        if response.candidates[0].content and response.candidates[0].content.parts:
            api_result_text = "".join(part.text for part in response.candidates[0].content.parts)
        
        if not api_result_text.strip():
            logging.warning(f"Attempt {attempt_num}: API returned an empty text response.")
            raise ValueError("API returned empty text response.")
            
        logging.info(f"Attempt {attempt_num}: Received response from Gemini API (Length: {len(api_result_text)}).")
        
        # Add detailed debugging of the API response
        logging.debug(f"API Response first 100 characters: {api_result_text[:100]}")
        logging.debug(f"API Response last 100 characters: {api_result_text[-100:]}")
        
        # Check if response appears to be JSON
        if not (api_result_text.strip().startswith('{') and api_result_text.strip().endswith('}')):
            logging.warning(f"Response doesn't appear to be valid JSON (doesn't start with {{ and end with }})")
        
        # Log characters around position 53 (where the error occurs)
        if len(api_result_text) > 60:
            error_area = api_result_text[40:70]
            logging.debug(f"Characters around position 53: '{error_area}'")
            # Show character codes around position 53 for debugging
            char_codes = [ord(c) for c in error_area]
            logging.debug(f"Character codes around position 53: {char_codes}")
        
        return api_result_text
        
    except Exception as e:
        logging.error(f"Attempt {attempt_num}: Error calling Gemini API: {e}")
        raise ValueError(f"Gemini API call failed: {e}") from e


def attempt_json_repair(json_text: str) -> str:
    """
    Attempt to repair malformed JSON as a last resort.
    This is a simplistic approach but might help in some cases.
    """
    logging.warning("Attempting emergency JSON repair")
    
    # Fix common issues with JSON strings
    # 1. Unclosed strings
    lines = json_text.split('\n')
    for i, line in enumerate(lines):
        # Look for lines with odd number of quotes
        quotes_count = line.count('"')
        if quotes_count % 2 == 1:
            # Try to find string boundaries and close them
            if ':' in line and line.strip().endswith(','):
                # This looks like a property line ending with a comma
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key_part = parts[0].strip()
                    value_part = parts[1].strip()
                    
                    # If value doesn't end with quote but has comma, add quote
                    if not value_part.endswith('"') and value_part.endswith(','):
                        value_part = value_part[:-1] + '",'.strip()
                        lines[i] = f"{key_part}: {value_part}"
    
    # Rejoin the lines
    repaired_text = '\n'.join(lines)
    
    # 2. Missing braces at the end
    open_braces = repaired_text.count('{')
    close_braces = repaired_text.count('}')
    if open_braces > close_braces:
        repaired_text += '}' * (open_braces - close_braces)
    
    return repaired_text

def validate_api_response(response_text: str, expected_segment_ids: List[int]) -> Dict[str, Any]:
    """
    Validate the API response with much more robust JSON parsing capabilities.
    """
    # Clean up the response to extract just the JSON
    response_text = response_text.strip()
    
    # Find JSON content - look for { at the beginning
    json_start = response_text.find('{')
    if json_start == -1:
        logging.error("No JSON object found in response")
        logging.debug(f"First 200 chars of response: {response_text[:200]}")
        raise ValueError("No JSON object found in response")
        
    # Extract the JSON part
    json_text = response_text[json_start:]
    
    # Handle potential markdown code block formatting
    if json_text.startswith("```json"):
        json_text = re.sub(r'^```json\s*', '', json_text)
        json_text = re.sub(r'\s*```$', '', json_text)
    elif json_text.startswith("```"):
        json_text = re.sub(r'^```\s*', '', json_text)
        json_text = re.sub(r'\s*```$', '', json_text)
    
    logging.debug(f"Attempting to parse JSON. First 100 chars: {json_text[:100]}")
    
    # Parse the JSON
    try:
        parsed_data = json.loads(json_text)
        logging.info("Successfully parsed JSON on first attempt")
        
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        error_line = int(str(e).split("line")[1].split()[0])
        error_col = int(str(e).split("column")[1].split()[0])
        
        # Calculate start and end positions for context
        lines = json_text.split('\n')
        if error_line <= len(lines):
            problem_line = lines[error_line-1]
            logging.error(f"Problem line: '{problem_line}'")
            if error_col <= len(problem_line):
                logging.error(f"Character at error position: '{problem_line[error_col-1]}' (code: {ord(problem_line[error_col-1])})")
                # Show 10 chars before and after the error position
                start_pos = max(0, error_col-11)
                end_pos = min(len(problem_line), error_col+10)
                context = problem_line[start_pos:end_pos]
                logging.error(f"Context around error: '{context}'")
        
        logging.error(f"First 500 chars of problematic JSON: {json_text[:500]}...")
        
        # Try much more aggressive JSON repair 
        try:
            repaired_json = advanced_json_repair(json_text, error_line, error_col)
            logging.debug(f"JSON after repair attempt - first 100 chars: {repaired_json[:100]}")
            parsed_data = json.loads(repaired_json)
            logging.info("Successfully parsed JSON after advanced repair")
        except Exception as e2:
            logging.error(f"Advanced repair failed: {e2}")
            
            # Last resort: try to extract individual segments and build our own JSON
            try:
                logging.info("Attempting manual JSON extraction as last resort")
                parsed_data = extract_segments_manually(json_text, expected_segment_ids)
                if parsed_data:
                    logging.info("Successfully extracted data using manual method")
                else:
                    raise ValueError("Manual extraction yielded no valid segments")
            except Exception as e3:
                logging.error(f"All parsing attempts failed: {e3}")
                raise ValueError(f"Failed to decode JSON after all repair attempts: {e}")
    
    # Validate structure and content (rest of the function)
    
    # Validate structure
    if not isinstance(parsed_data, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed_data)}")
    
    # Check if all expected segments are present
    expected_ids = {str(segment_id) for segment_id in expected_segment_ids}
    actual_ids = set(parsed_data.keys())
    
    missing_ids = expected_ids - actual_ids
    if missing_ids:
        raise ValueError(f"Missing segments in response: {missing_ids}")
    
    # Validate structure of each segment
    required_fields = {"segment_no", "text_original", "text_en", "polarity", "confidence_percent", "reasoning_en"}
    
    for segment_id, segment_data in parsed_data.items():
        if not isinstance(segment_data, dict):
            raise ValueError(f"Segment {segment_id} data is not a dictionary")
            
        missing_fields = required_fields - set(segment_data.keys())
        if missing_fields:
            raise ValueError(f"Segment {segment_id} missing required fields: {missing_fields}")
            
        # Validate data types
        try:
            float_polarity = float(segment_data["polarity"])
            if not -1.0 <= float_polarity <= 1.0 and float_polarity != 99.9:
                raise ValueError(f"Segment {segment_id} has invalid polarity value: {float_polarity}")
                
            int_confidence = int(segment_data["confidence_percent"])
            if not 0 <= int_confidence <= 100:
                raise ValueError(f"Segment {segment_id} has invalid confidence value: {int_confidence}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Segment {segment_id} has invalid data types: {e}")
    
    return parsed_data


def advanced_json_repair(json_text: str, error_line: int, error_col: int) -> str:
    """
    Much more sophisticated JSON repair function that handles various common issues
    especially with Ancient Greek text.
    """
    logging.warning("Attempting advanced JSON repair")
    
    # Convert to lines for easier processing
    lines = json_text.split('\n')
    
    # 1. Fix the specific error line
    if 0 < error_line <= len(lines):
        problem_line = lines[error_line-1]
        
        # Check for common issues:
        
        # 1.1 Unclosed string
        quote_positions = [i for i, char in enumerate(problem_line) if char == '"']
        if len(quote_positions) % 2 == 1:  # Odd number of quotes means unclosed string
            # Find the last quote in the line
            last_quote_pos = quote_positions[-1]
            
            # If it's a value in a key-value pair, add a closing quote
            if ':' in problem_line and last_quote_pos > problem_line.find(':'):
                if problem_line.rstrip().endswith(','):
                    # If line ends with comma, insert quote before comma
                    lines[error_line-1] = problem_line[:-1] + '"' + problem_line[-1]
                else:
                    # Otherwise just add quote at end
                    lines[error_line-1] = problem_line + '"'
                logging.info(f"Fixed unclosed string in line {error_line}")
        
        # 1.2 Special character breaking JSON
        # Look for Ancient Greek or other non-ASCII characters near error position
        if error_col > 0 and error_col <= len(problem_line):
            # Get 5 chars before and after error position
            start = max(0, error_col - 6)
            end = min(len(problem_line), error_col + 5)
            context = problem_line[start:end]
            
            # Check for any potentially problematic characters
            for i, char in enumerate(context):
                if ord(char) > 127:  # non-ASCII character
                    abs_pos = start + i
                    # If this char is within a string, escape it
                    if is_within_string(problem_line, abs_pos):
                        char_escaped = f"\\u{ord(char):04x}"
                        new_line = problem_line[:abs_pos] + char_escaped + problem_line[abs_pos+1:]
                        lines[error_line-1] = new_line
                        logging.info(f"Escaped non-ASCII character '{char}' at position {abs_pos} in line {error_line}")
                        break
    
    # 2. Fix common structural issues
    
    # 2.1 Check for and fix missing braces
    open_braces = json_text.count('{')
    close_braces = json_text.count('}')
    if open_braces > close_braces:
        # Add missing closing braces
        lines[-1] = lines[-1] + '}' * (open_braces - close_braces)
        logging.info(f"Added {open_braces - close_braces} missing closing braces")
    
    # 2.2 Fix trailing commas before closing braces
    for i, line in enumerate(lines):
        if re.search(r',\s*}', line):
            lines[i] = re.sub(r',(\s*})', r'\1', line)
            logging.info(f"Removed trailing comma before closing brace in line {i+1}")
    
    # 3. Check and fix newlines within strings
    in_string = False
    fixed_lines = []
    current_line = ""
    
    for line in lines:
        for char in line:
            if char == '"' and (len(current_line) == 0 or current_line[-1] != '\\'):
                in_string = not in_string
            
            current_line += char
        
        # If we're in the middle of a string, add an escaped newline
        if in_string:
            current_line += '\\n'
        else:
            fixed_lines.append(current_line)
            current_line = ""
    
    # If we have a pending line (e.g., ended while in a string)
    if current_line:
        fixed_lines.append(current_line)
    
    # If we fixed newlines, use the fixed lines
    if len(fixed_lines) < len(lines):
        logging.info("Fixed newlines inside strings")
        return '\n'.join(fixed_lines)
    
    # Return the repaired JSON
    return '\n'.join(lines)

def is_within_string(line: str, pos: int) -> bool:
    """
    Determine if a position in a line is within a JSON string.
    """
    # Count quotes before this position
    quote_count = line[:pos].count('"')
    # If the count is odd, we're inside a string
    return quote_count % 2 == 1


def adjust_batch_size(current_size: int, failure_count: int) -> int:
    """
    Dynamically adjust batch size based on failure patterns.
    Reduces batch size after failures to improve success rate.
    """
    if failure_count == 0:
        return current_size  # Keep current size if no failures
    
    # Reduce batch size after failures
    new_size = max(1, current_size - failure_count)
    logging.info(f"Adjusting batch size from {current_size} to {new_size} due to failures")
    return new_size


def extract_segments_manually(json_text: str, expected_segment_ids: List[int]) -> Dict[str, Dict]:
    """
    Last resort function that attempts to manually extract segment data
    without relying on JSON parsing.
    """
    result = {}
    
    # Look for patterns like "1": { ... }, "2": { ... }
    for segment_id in expected_segment_ids:
        segment_id_str = str(segment_id)
        
        # Find the start of this segment's data
        segment_pattern = rf'"{segment_id_str}"\s*:\s*{{'
        match = re.search(segment_pattern, json_text)
        
        if not match:
            logging.warning(f"Couldn't find segment {segment_id_str} in response")
            continue
        
        start_pos = match.start()
        
        # Find the matching closing brace
        # This is tricky because we need to account for nested braces
        brace_level = 0
        end_pos = start_pos
        found_closing = False
        
        for i in range(start_pos, len(json_text)):
            if json_text[i] == '{':
                brace_level += 1
            elif json_text[i] == '}':
                brace_level -= 1
                if brace_level == 0:
                    # Found matching closing brace
                    end_pos = i + 1
                    found_closing = True
                    break
        
        if not found_closing:
            logging.warning(f"Couldn't find closing brace for segment {segment_id_str}")
            continue
        
        # Extract the segment data including braces
        segment_text = json_text[start_pos:end_pos]
        
        # Now extract the fields we need
        segment_data = {
            "segment_no": segment_id,
            "text_original": extract_field(segment_text, "text_original"),
            "text_en": extract_field(segment_text, "text_en"),
            "polarity": extract_float_field(segment_text, "polarity"),
            "confidence_percent": extract_int_field(segment_text, "confidence_percent"),
            "reasoning_en": extract_field(segment_text, "reasoning_en")
        }
        
        # Only add if we got at least some data
        if any(segment_data.values()):
            result[segment_id_str] = segment_data
            logging.info(f"Manually extracted data for segment {segment_id_str}")
    
    return result

def extract_field(text: str, field_name: str) -> str:
    """Extract a string field from semi-structured JSON text."""
    pattern = rf'"{field_name}"\s*:\s*"(.*?)(?<!\\)"'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # Unescape common escape sequences
        value = match.group(1)
        value = value.replace('\\"', '"').replace('\\n', '\n').replace('\\r', '\r')
        return value
    return ""

def extract_float_field(text: str, field_name: str) -> float:
    """Extract a float field from semi-structured JSON text."""
    pattern = rf'"{field_name}"\s*:\s*(-?\d+\.?\d*)'
    match = re.search(pattern, text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return 0.0

def extract_int_field(text: str, field_name: str) -> int:
    """Extract an integer field from semi-structured JSON text."""
    pattern = rf'"{field_name}"\s*:\s*(\d+)'
    match = re.search(pattern, text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return 0


def write_to_csv(output_file: Path, results: Dict[str, Dict], create_new: bool = False):
    """
    Write the results to a CSV file.
    If create_new is True, create a new file; otherwise, append to existing file.
    """
    file_exists = output_file.exists() and not create_new
    
    mode = 'a' if file_exists else 'w'
    write_headers = not file_exists
    
    try:
        with open(output_file, mode, encoding='utf-8', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
            
            if write_headers:
                writer.writeheader()
            
            # Sort results by segment number to ensure ordered output
            sorted_segments = sorted(results.items(), key=lambda x: int(x[0]))
            
            for segment_id, segment_data in sorted_segments:
                # Prepare row with the exact column names
                row = {
                    "segment_no": segment_data["segment_no"],
                    "text_original": segment_data["text_original"],
                    "text_en": segment_data["text_en"],
                    "polarity": segment_data["polarity"],
                    "confidence_percent": segment_data["confidence_percent"],
                    "reasoning_en": segment_data["reasoning_en"]
                }
                writer.writerow(row)
                
        logging.info(f"Successfully wrote {len(results)} segments to CSV")
        
    except Exception as e:
        logging.error(f"Error writing to CSV: {e}")
        raise

def write_error_placeholders(output_file: Path, segment_ids: List[int]):
    """
    Write placeholder rows for segments that failed processing.
    Only populates the segment_no field.
    """
    try:
        with open(output_file, 'a', encoding='utf-8', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
            
            for segment_id in segment_ids:
                # Create empty placeholder row
                row = {
                    "segment_no": segment_id,
                    "text_original": "",
                    "text_en": "",
                    "polarity": "",
                    "confidence_percent": "",
                    "reasoning_en": ""
                }
                writer.writerow(row)
                
        logging.warning(f"Wrote {len(segment_ids)} error placeholder rows to CSV")
        
    except Exception as e:
        logging.error(f"Error writing error placeholders to CSV: {e}")
        raise

# --- Main Processing Function ---

def process_text_segments(model):
    """
    Main function to process the text segments.
    Handles segmentation, batching, API calls, and CSV output.
    """
    # Read the input file
    logging.info(f"Reading input file: {INPUT_FILE}")
    try:
        input_text = read_input_file(INPUT_FILE)
        logging.info(f"Successfully read input file ({len(input_text)} characters)")
    except Exception as e:
        logging.error(f"Failed to read input file: {e}")
        return False
    
    # Segment the text
    logging.info(f"Segmenting text using pattern: {SEGMENT_SEP_DEFAULT}")
    segments = segment_text(input_text, SEGMENT_SEP_DEFAULT, MIN_SEGMENT_CHAR)  # CUSTOMIZE: delete MIN_SEGMENT_CHAR to use default regex
    logging.info(f"Text segmented into {len(segments)} segments")
    
    # Check restart point if needed
    if FLAG_FULL_RESTART and OUTPUT_FILE.exists():
        logging.info("Full restart flag is set, removing existing output file")
        OUTPUT_FILE.unlink()
    
    start_segment_no = 0
    
    if not FLAG_FULL_RESTART:
        last_segment_no, last_text = find_last_processed_segment(OUTPUT_FILE)
        
        if last_segment_no > 0:
            logging.info(f"Found existing output with last processed segment: {last_segment_no}")
            
            # Validate restart point
            if not validate_restart_point(segments, last_segment_no, last_text):
                logging.error("Restart validation failed. Input file appears different from previous processing.")
                logging.error("Use --full-restart flag to force processing from the beginning.")
                return False
                
            start_segment_no = last_segment_no
            logging.info(f"Will resume processing from segment {start_segment_no + 1}")
    
    # Create output directory if it doesn't exist
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Create output file if it doesn't exist
    if not OUTPUT_FILE.exists():
        logging.info(f"Creating new output file: {OUTPUT_FILE}")
        with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
            writer.writeheader()
    
    # Process segments in batches
    segments_to_process = segments[start_segment_no:]
    logging.info(f"Will process {len(segments_to_process)} remaining segments in batches of {BATCH_SIZE}")
    
    if not segments_to_process:
        logging.info("No segments to process. Exiting.")
        return True
    
    # Create batches with adjusted segment numbers
    batches = batch_segments(segments_to_process, BATCH_SIZE)
    logging.info(f"Created {len(batches)} batches")
    
    batch_number = 0
    total_processed = 0
    total_errors = 0
    current_batch_size = BATCH_SIZE
    consecutive_failures = 0
    
    # Create batches with dynamic batch size
    segments_to_process = segments[start_segment_no:]
    
    # Process until all segments are handled
    remaining_segments = segments_to_process
    
    while remaining_segments:
        # Adjust batch size dynamically based on previous failures
        current_batch_size = adjust_batch_size(current_batch_size, consecutive_failures)
        
        # Take the next batch
        current_batch = batch_segments(remaining_segments[:current_batch_size], current_batch_size)[0]
        batch_number += 1
        
        batch_segment_ids = [segment_no for segment_no, _ in current_batch]
        logging.info(f"Processing batch {batch_number} with size {len(current_batch)} (segments {min(batch_segment_ids)}-{max(batch_segment_ids)})")
        
        # Process this batch with retries
        success = False
        processed_data = None
        
        for attempt in range(1, MAX_API_RETRIES + 1):
            if attempt > 1:
                logging.info(f"Retry attempt {attempt}/{MAX_API_RETRIES} for batch {batch_number}")
                time.sleep(RETRY_DELAY * attempt)  # Increasing delay for subsequent retries
            
            try:
                # Prepare the prompt for this batch
                prompt = prepare_prompt(current_batch)
                
                # Log key metrics about the prompt
                logging.debug(f"Prompt length: {len(prompt)} characters")
                logging.debug(f"Batch size: {len(current_batch)} segments")
                
                # Call the API
                api_response = call_gemini_api(model, prompt, attempt)
                
                # Validate the response
                processed_data = validate_api_response(api_response, batch_segment_ids)
                
                # If we got here, processing was successful
                success = True
                consecutive_failures = 0  # Reset failure counter on success
                break
                
            except ValueError as e:
                logging.warning(f"Batch {batch_number}, attempt {attempt} failed: {e}")
                if attempt == MAX_API_RETRIES:
                    logging.error(f"All retry attempts failed for batch {batch_number}")
                    consecutive_failures += 1
        
        # Handle the results
        if success and processed_data:
            try:
                # Write successful results to CSV
                write_to_csv(OUTPUT_FILE, processed_data)
                total_processed += len(processed_data)
                logging.info(f"Successfully processed batch {batch_number} ({len(processed_data)} segments)")
                
                # Remove successfully processed segments from the queue
                remaining_segments = remaining_segments[len(current_batch):]
                
            except Exception as e:
                logging.error(f"Failed to write results for batch {batch_number}: {e}")
                total_errors += len(current_batch)
                consecutive_failures += 1
                
                # Write error placeholders
                try:
                    write_error_placeholders(OUTPUT_FILE, batch_segment_ids)
                except:
                    logging.error(f"Failed to write error placeholders for batch {batch_number}")
                
                # Still remove these segments to avoid getting stuck
                remaining_segments = remaining_segments[len(current_batch):]
        else:
            # Handle failed batch - reduce batch size for difficult segments
            logging.error(f"Failed to process batch {batch_number} after {MAX_API_RETRIES} attempts")
            
            if current_batch_size > 1:
                # Try with smaller batch size rather than skipping
                logging.info(f"Will try again with smaller batch size")
                current_batch_size = max(1, current_batch_size // 2)
            else:
                # If we're already at batch size 1, we have to skip this segment
                total_errors += len(current_batch)
                logging.error(f"Skipping segment {batch_segment_ids[0]} after all attempts failed")
                
                # Write error placeholder and move on
                write_error_placeholders(OUTPUT_FILE, batch_segment_ids)
                remaining_segments = remaining_segments[1:]  # Remove just the problem segment
        
        # Delay between batches
        if remaining_segments:
            logging.debug(f"Waiting {API_CALL_DELAY}s before next batch...")
            time.sleep(API_CALL_DELAY)
    
    # Processing complete
    logging.info("=== Processing Complete ===")
    logging.info(f"Total segments processed successfully: {total_processed}")
    logging.info(f"Total segments with errors: {total_errors}")
    logging.info(f"Results saved to: {OUTPUT_FILE}")
    
    return total_errors == 0

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Ancient Greek text segments for sentiment analysis.")
    parser.add_argument("--full-restart", action="store_true", help="Ignore existing output and process from the beginning.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help=f"Name of the Gemini model to use (default: {DEFAULT_MODEL_NAME}).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Number of segments to process in each batch (default: {BATCH_SIZE}).")
    parser.add_argument("--delay", type=float, default=API_CALL_DELAY, help=f"Delay in seconds between batches (default: {API_CALL_DELAY}).")
    parser.add_argument("--retry-delay", type=float, default=RETRY_DELAY, help=f"Delay in seconds between retries (default: {RETRY_DELAY}).")
    parser.add_argument("--max-retries", type=int, default=MAX_API_RETRIES, help=f"Maximum retry attempts (default: {MAX_API_RETRIES}).")
    parser.add_argument('--debug', action='store_true', help='Enable debug logging.')
    args = parser.parse_args()
    
    # Update globals from args
    FLAG_FULL_RESTART = args.full_restart
    DEFAULT_MODEL_NAME = args.model
    BATCH_SIZE = args.batch_size
    API_CALL_DELAY = args.delay
    RETRY_DELAY = args.retry_delay
    MAX_API_RETRIES = args.max_retries
    
    # Setup logging
    log_file = setup_logging()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("DEBUG logging enabled.")
    
    logging.info(f"Script started. Log file: {log_file}")
    logging.info(f"Input file: {INPUT_FILE}")
    logging.info(f"Output file: {OUTPUT_FILE}")
    logging.info(f"Using model: {args.model}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Full restart: {args.full_restart}")
    logging.info(f"Delay between batches: {args.delay}s")
    logging.info(f"Retry delay: {args.retry_delay}s")
    logging.info(f"Max retry attempts: {args.max_retries}")
    
    # Validate API key
    if not API_KEY:
        logging.error("GOOGLE_API_KEY not found. Please set it in your .env file.")
        exit(1)
    
    try:
        # Configure Gemini API
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(args.model)
        logging.info(f"Successfully configured Gemini model: {args.model}")
        
        # Process the segments
        success = process_text_segments(model)
        
        if success:
            logging.info("Processing completed successfully.")
            exit(0)
        else:
            logging.error("Processing completed with errors.")
            exit(1)
            
    except Exception as e:
        logging.exception(f"Critical error in main execution: {e}")
        exit(1) 