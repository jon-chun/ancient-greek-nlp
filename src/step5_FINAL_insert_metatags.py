import pandas as pd
from pathlib import Path
import time
import ast
import logging
import sys
import os
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the project root directory (assuming the script is in /src)
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent  # Assumes script is in 'src'
    logging.info(f"Script directory: {SCRIPT_DIR}")
    logging.info(f"Project root: {PROJECT_ROOT}")
except NameError:
    logging.warning("__file__ not defined. Assuming execution from project root.")
    PROJECT_ROOT = Path.cwd()
    logging.info(f"Project root (from cwd): {PROJECT_ROOT}")

# Check if .env file exists
env_path = PROJECT_ROOT / ".env"
logging.info(f"Looking for .env file at: {env_path}")
logging.info(f".env file exists: {env_path.exists()}")

# Try to automatically find .env file
dotenv_path = find_dotenv()
if dotenv_path:
    logging.info(f"Automatically found .env at: {dotenv_path}")
else:
    logging.warning("Could not automatically find .env file")

# Load .env from the project root with verbose output
loaded = load_dotenv(env_path, verbose=True)
logging.info(f"Loaded .env file: {loaded}")

# List all environment variables (without their values for security)
env_vars = list(os.environ.keys())
logging.info(f"Available environment variables: {env_vars}")

# Check specifically for our API key
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    logging.info("GOOGLE_API_KEY found in environment variables")
    # Show just the beginning of the key for verification (don't log the full key!)
    masked_key = API_KEY[:4] + "..." if len(API_KEY) > 4 else "***"
    logging.info(f"GOOGLE_API_KEY starts with: {masked_key}")
else:
    logging.error("GOOGLE_API_KEY not found in environment variables")
    # Try direct file reading as a fallback
    try:
        if env_path.exists():
            with open(env_path, 'r') as f:
                env_content = f.read()
                # Fix the problematic line with backslash
                first_line = env_content.split('\n')[0] if env_content else "Empty file"
                logging.info(f".env file content (first line): {first_line}")
                if "GOOGLE_API_KEY" in env_content:
                    logging.info("GOOGLE_API_KEY string found in .env file content")
                else:
                    logging.error("GOOGLE_API_KEY string NOT found in .env file content")
    except Exception as e:
        logging.error(f"Error reading .env file directly: {e}")
    
    raise ValueError("Please set the GOOGLE_API_KEY in your .env file.")

# Configure the API
try:
    genai.configure(api_key=API_KEY)
    logging.info("Successfully configured genai with API key")
except Exception as e:
    logging.error(f"Error configuring genai: {e}")
    raise

# --- Select the Gemini model ---
MODEL_NAME = "gemini-2.5-flash-preview-04-17"
model = genai.GenerativeModel(MODEL_NAME)
logging.info(f"Successfully initialized model: {MODEL_NAME}")

# Rest of your code...
logging.info("Environment setup completed successfully")

# --- Determine Project Root based on script location ---
try:
    # __file__ is the path to the current script. resolve() makes it absolute.
    # .parent gets the directory containing the script (e.g., .../src/)
    # .parent again gets the parent of that directory (e.g., .../sentiment-arabic-news/)
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent  # Assumes script is in 'src'
except NameError:
    # Fallback if __file__ is not defined (e.g., running in an interactive environment)
    # Assumes the script is run from the project root in this case.
    logging.warning("__file__ not defined. Assuming execution from project root.")
    PROJECT_ROOT = Path.cwd()  # Use current working directory as project root

# --- Configuration ---
# Define paths relative to the determined PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"

INPUT_TOPIC_CSVFILE = METADATA_DIR / "aggregated_news_topic-sa_raw.csv"
OUTPUT_TOPIC_METATAG_CSVFILE = METADATA_DIR / "aggregated_news_topic-sa_topic-metatag.csv"

CHUNK_SIZE = 25
MAX_API_RETRIES = 3
API_RETRY_DELAY_SECONDS = 5  # Time to wait between retries

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Prompt Template ---
prompt_str_template = """
TASK: Classify each topic with the most specific applicable metatag from the taxonomy below.

CLASSIFICATION TAXONOMY:
[Politics & Administration]
- "Trump-Administration": Donald Trump policies, politics, impact
- "US-China": US-China trade war, tariffs, economic relations
- "US-Canada": US-Canada relations and disputes
- "US-Europe": European politics and US-Europe relations

[Conflicts & International Affairs]
- "Ukraine-Conflict": Ukraine conflict and related diplomacy
- "Gaza-Conflict": Gaza conflict and related issues
- "Syria-Situation": Syrian civil war and related diplomacy
- "Iran-Situation": Iran relations and nuclear program
- "Iraq-Situation": Iraq and regional diplomacy
- "Yemen-Situation": Yemen conflict including Houthi activities
- "Egypt-Situation": Egyptian domestic politics and issues
- "Tunisia-Situation": Tunisian domestic politics and issues
- "MENA-Other": Other Middle East & North Africa topics not covered above

[Economics & Technology]
- "Global-Economy": Global economy, finance, and broader trade issues
- "Economics": Business, finance, markets, and economic policy
- "Technology": Technology, resources, and supply chains

[Society & Culture]
- "Immigration": Immigration policy and human rights
- "Social-Culture": Social trends and cultural issues
- "Health-Food": Health, medicine, and food safety
- "Social-Other": Other societal issues not covered above

CLASSIFICATION RULES:
1. Assign EXACTLY ONE metatag to each topic
2. Always use the MOST SPECIFIC applicable metatag
3. For economic topics specific to US-China relations, use "US-China" not "Economics"
4. For topics related to Trump and Ukraine, use "Trump-Administration" if about policy positions
5. For topics related to Trump and Ukraine, use "Ukraine-Conflict" if about the conflict itself

EXAMPLE CLASSIFICATIONS:
"US Tariffs on Chinese Electronics" → "US-China"
"Global Recession Concerns" → "Global-Economy"
"Trump's Ukraine Policy" → "Trump-Administration"
"Russian Missile Strikes in Ukraine" → "Ukraine-Conflict"
"Saudi Arabian Oil Production" → "MENA-Other"
"Egyptian Food Store Regulations" → "Egypt-Situation"
"Rising Gold Prices" → "Economics"
"Immigration at US-Mexico Border" → "Immigration"

TOPICS TO CLASSIFY:
{topic_sublist_str}

OUTPUT FORMAT:
Return only a valid Python dictionary with the topic texts as keys and metatags as values:
{{
  "Topic 1": "Appropriate-Metatag",
  "Topic 2": "Appropriate-Metatag"
}}

IMPORTANT: Ensure your response is ONLY the Python dictionary, nothing else.
"""

# --- Actual API Call Function Using Google Gemini ---
def get_llm_response(prompt: str, topics_in_prompt: list) -> str:
    """
    Calls the Google Gemini API to classify topics.
    Returns a string representation of a Python dictionary.
    """
    logging.debug(f"Calling Google Gemini API for {len(topics_in_prompt)} topics.")
    try:
        # Make the API call
        response = model.generate_content(prompt)
        
        # Check for API response errors
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            raise ValueError(f"API prompt blocked: {response.prompt_feedback.block_reason}")
        
        # Extract the text response
        if not response.text:
            raise ValueError("API failed to produce output")
        
        response_text = response.text.strip()
        
        # Try to clean up the response if it has extra text
        # Sometimes APIs add explanatory text before or after the dict
        if response_text.startswith("```python"):
            # Extract content between code blocks if present
            response_text = response_text.split("```python")[1].split("```")[0].strip()
        elif response_text.startswith("```"):
            # Handle generic code blocks
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        # Try to extract just the dictionary part if there's explanatory text
        try:
            # Find the first { and last }
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                response_text = response_text[start_idx:end_idx+1]
                
            # Validate it's a proper dict by parsing it
            parsed = ast.literal_eval(response_text)
            if not isinstance(parsed, dict):
                raise ValueError("Response is not a dictionary")
                
            # Ensure all topics are classified
            missing_topics = [topic for topic in topics_in_prompt if topic not in parsed]
            if missing_topics:
                logging.warning(f"Topics missing in API response: {missing_topics}")
                for topic in missing_topics:
                    parsed[topic] = "Social-Other"  # Default classification
                response_text = str(parsed)
                
            return response_text
        except (SyntaxError, ValueError) as e:
            logging.error(f"Failed to parse API response as dictionary: {e}")
            logging.debug(f"Raw response: {response_text}")
            raise ValueError(f"API response is not a valid Python dictionary: {e}")
            
    except Exception as e:
        logging.error(f"API call failed: {e}")
        raise

# --- Main Script Logic ---
def main():
    # Corrected log message to reflect intended script name if desired
    logging.info("Starting script: step4b_assign_topic-group_metatag.py")

    # --- 0. Read Input CSV ---
    logging.info(f"Resolved Project Root: {PROJECT_ROOT}")
    logging.info(f"Attempting to read input CSV: {INPUT_TOPIC_CSVFILE}")
    try:
        # Ensure parent directories exist for output
        OUTPUT_TOPIC_METATAG_CSVFILE.parent.mkdir(parents=True, exist_ok=True)

        # Check for input file *before* trying to create dummy data
        if not INPUT_TOPIC_CSVFILE.exists():
            logging.warning(f"Input file {INPUT_TOPIC_CSVFILE} not found. Creating a dummy file.")
            # Ensure the directory exists before writing the dummy file
            INPUT_TOPIC_CSVFILE.parent.mkdir(parents=True, exist_ok=True)
            dummy_data = {
                'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 6,  # 60 dummy rows
                'topic_en': [
                    "US Mediation in Ukraine-Russia War", "Donald Trump's Stance on the War",
                    "Russian Strikes on Ukraine", "Gold Prices Rising", "US Tariffs on China",
                    "Global Trade Concerns and Recession", "Egyptian Authorities' decision to close food and sweets stores",
                    "Saudi Aramco IPO details", "Canadian response to US tariffs", "EU summit on energy policy"
                ] * 6,
                'other_col': ['data'] * 60
            }
            pd.DataFrame(dummy_data).to_csv(INPUT_TOPIC_CSVFILE, index=False, encoding='utf-8')
            logging.info(f"Dummy input file created at {INPUT_TOPIC_CSVFILE}")

        # Now read the file (either original or the dummy one)
        df = pd.read_csv(INPUT_TOPIC_CSVFILE, encoding='utf-8')
        if "topic_en" not in df.columns:
            logging.error(f"Column 'topic_en' not found in {INPUT_TOPIC_CSVFILE}")
            return
        logging.info(f"Successfully read {len(df)} rows from {INPUT_TOPIC_CSVFILE}")

    except FileNotFoundError:  # Should theoretically be caught by the .exists() check now
        logging.error(f"Input file not found despite check: {INPUT_TOPIC_CSVFILE}")
        return
    except Exception as e:
        logging.error(f"Error reading CSV file or creating dummy file: {e}")
        return

    # --- 1. Extract Topics ---
    topic_ls = df["topic_en"].tolist()
    logging.info(f"Extracted {len(topic_ls)} topics from 'topic_en' column.")

    if not topic_ls:
        logging.warning("No topics found in the input file. Exiting.")
        return

    # --- 2. Iterate and Call API ---
    all_metatags_ls = []
    num_topics = len(topic_ls)

    for i in range(0, num_topics, CHUNK_SIZE):
        chunk_start_time = time.time()
        start_index = i
        end_index = min(i + CHUNK_SIZE, num_topics)
        topic_sublist_ls = topic_ls[start_index:end_index]
        logging.info(f"Processing chunk {start_index+1}-{end_index} of {num_topics}...")

        topic_sublist_str = "\n".join([f'"{topic}"' for topic in topic_sublist_ls])
        prompt = prompt_str_template.format(topic_sublist_str=topic_sublist_str)

        metatag_dict = None
        last_exception = None

        for attempt in range(MAX_API_RETRIES):
            logging.info(f"  Attempt {attempt + 1}/{MAX_API_RETRIES} for chunk {start_index+1}-{end_index}...")
            try:
                response_str = get_llm_response(prompt, topic_sublist_ls)
                parsed_response = ast.literal_eval(response_str)

                if not isinstance(parsed_response, dict):
                    raise ValueError(f"API response is not a dictionary: {type(parsed_response)}")

                chunk_metatags = []
                missing_topics = []
                for topic in topic_sublist_ls:
                    if topic in parsed_response:
                        chunk_metatags.append(parsed_response[topic])
                    else:
                        logging.warning(f"Topic missing in API response for chunk {start_index+1}-{end_index}: '{topic}'")
                        missing_topics.append(topic)
                        chunk_metatags.append(None)  # Assign None for missing

                if missing_topics:
                    logging.error(f"API failed to return classification for {len(missing_topics)} topics in chunk {start_index+1}-{end_index}. Assigned None.")

                if len(chunk_metatags) != len(topic_sublist_ls):
                    raise ValueError(f"Mismatch in number of metatags received ({len(chunk_metatags)}) vs topics sent ({len(topic_sublist_ls)})")

                metatag_dict = parsed_response
                all_metatags_ls.extend(chunk_metatags)
                logging.info(f"  Successfully processed chunk {start_index+1}-{end_index}.")
                break

            except (SyntaxError, ValueError, TypeError) as e:
                last_exception = e
                logging.warning(f"  Malformed response or parsing error on attempt {attempt + 1}: {e}")
                if attempt < MAX_API_RETRIES - 1:
                    logging.info(f"  Retrying in {API_RETRY_DELAY_SECONDS} seconds...")
                    time.sleep(API_RETRY_DELAY_SECONDS)
                else:
                    logging.error(f"  Failed to process chunk {start_index+1}-{end_index} after {MAX_API_RETRIES} attempts.")
                    all_metatags_ls.extend([None] * len(topic_sublist_ls))
            except Exception as e:
                last_exception = e
                logging.error(f"  An unexpected error occurred during API call or processing on attempt {attempt + 1}: {e}")
                if attempt < MAX_API_RETRIES - 1:
                    logging.info(f"  Retrying in {API_RETRY_DELAY_SECONDS} seconds...")
                    time.sleep(API_RETRY_DELAY_SECONDS)
                else:
                    logging.error(f"  Failed to process chunk {start_index+1}-{end_index} after {MAX_API_RETRIES} attempts due to unexpected error.")
                    all_metatags_ls.extend([None] * len(topic_sublist_ls))

        chunk_end_time = time.time()
        logging.debug(f"Chunk {start_index+1}-{end_index} processing took {chunk_end_time - chunk_start_time:.2f} seconds.")

    # --- 3. Assemble and Insert Metatags ---
    logging.info("Finished processing all chunks.")

    if len(all_metatags_ls) != num_topics:
        logging.error(f"FATAL: Length mismatch! Expected {num_topics} metatags, but generated {len(all_metatags_ls)}.")
        return

    logging.info(f"Successfully assembled {len(all_metatags_ls)} metatags.")

    try:
        topic_en_index = df.columns.get_loc("topic_en")
    except KeyError:
        logging.error("Column 'topic_en' seems to have disappeared. Cannot insert metatag column correctly.")
        return

    df.insert(loc=topic_en_index + 1, column="topic_metatag", value=all_metatags_ls)
    logging.info("Inserted 'topic_metatag' column into DataFrame.")

    # --- 4. Write Output CSV ---
    logging.info(f"Writing output CSV: {OUTPUT_TOPIC_METATAG_CSVFILE}")
    try:
        df.to_csv(OUTPUT_TOPIC_METATAG_CSVFILE, index=False, encoding='utf-8')
        logging.info(f"Successfully wrote DataFrame with metatags to {OUTPUT_TOPIC_METATAG_CSVFILE}")
    except Exception as e:
        logging.error(f"Error writing output CSV file: {e}")

    logging.info("Script finished.")

# --- Run the script ---
if __name__ == "__main__":
    main()