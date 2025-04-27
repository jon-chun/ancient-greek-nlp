#!/usr/bin/env python3
"""
step1c_segment_document_metachunks-files.py - Segments an Ancient Greek text, 
                                             combines segments into metachunks,
                                             and distributes them into multiple files.

This script:
1. Reads an Ancient Greek text file (handling various encodings).
2. Segments the text into initial chunks based on a regex pattern.
3. Cleans each chunk by standardizing whitespace.
4. Combines consecutive cleaned chunks into larger "metachunks".
5. Divides the metachunks into a specified number of output files.
6. Saves each group of metachunks to a separate file in a designated subdirectory.
"""

import os
import re
import logging
import argparse
import math
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# --- Configuration ---

# Define script directory and base data directory
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DATA_DIR = SCRIPT_DIR.parent / 'data'
CORPUS_NAME = "bible_ancient-greek" # Example corpus name, adjust as needed

# Input file path
INPUT_FILE = BASE_DATA_DIR / CORPUS_NAME / "document" / f"{CORPUS_NAME}.txt"

# Output configuration
DEFAULT_OUTPUT_SUBDIR = BASE_DATA_DIR / CORPUS_NAME / "segments_metachunks"
METACHUNK_COUNT_DEFAULT = 2  # Number of original chunks per metachunk
FILE_COUNT_DEFAULT = 10      # Number of output files to distribute metachunks into

# Processing configuration
SEGMENT_SEP_DEFAULT = r"[\n]{2,}"  # Regex for initial chunk separation (e.g., double newlines)
METACHUNK_INTERNAL_SEP = "\n\n" # Separator used *within* a metachunk (between original chunks)
METACHUNK_FILE_SEP = "\n\n----\n\n" # Separator used *between* metachunks within an output file

# --- Logging Setup ---

def setup_logging():
    """Sets up logging to file and console."""
    # Ensure the base logs directory exists for the corpus
    log_base_dir = BASE_DATA_DIR / CORPUS_NAME / "logs"
    log_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a specific log directory for this script's runs
    log_dir = log_base_dir / "step1c_segment_metachunks"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"segment_metachunks_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized.")
    return log_filename

# --- Helper Functions ---

def read_input_file(file_path: Path) -> str:
    """Read the input file with improved handling for Ancient Greek text."""
    encodings_to_try = ['utf-8-sig', 'utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'iso-8859-7', 'windows-1253']
    
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
                
            # Basic check if content seems plausible (optional, can be refined)
            if content and len(content) > 10: # Avoid empty or tiny files
                # Check if content looks valid (contains Greek characters or common punctuation)
                # This is a basic check, might need adjustment based on actual content
                if re.search(r'[\u0370-\u03FF\u1F00-\u1FFF]|[.,;·\s]', content):
                    logging.info(f"Successfully read file '{file_path}' using {encoding} encoding")
                    
                    # Remove any Byte Order Mark (BOM) if present, especially for utf-8-sig
                    if content.startswith('\ufeff'):
                        content = content[1:]
                        logging.info("Removed BOM character from text")
                    
                    return content
                else:
                    logging.warning(f"File read with {encoding}, but content seems unusual. Trying next encoding.")
            else:
                 logging.warning(f"File read with {encoding}, but content is very short or empty. Trying next encoding.")

        except UnicodeDecodeError:
            logging.debug(f"Failed to decode file '{file_path}' with {encoding}")
            continue
        except FileNotFoundError:
            logging.error(f"Input file not found: {file_path}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred while reading '{file_path}' with {encoding}: {e}")
            # Depending on the error, you might want to try the next encoding or raise immediately
            continue # Try next encoding
            
    # If all encodings fail
    error_msg = f"Could not decode file '{file_path}' with any of the attempted encodings: {encodings_to_try}"
    logging.error(error_msg)
    raise ValueError(error_msg)

def segment_text(text: str, pattern: str) -> List[str]:
    """
    Split the text into segments based on the specified regex pattern.
    
    Args:
        text (str): The input text to segment.
        pattern (str): The regex pattern to use for splitting.
    
    Returns:
        List[str]: A list of non-empty text segments (chunks).
    """
    logging.debug(f"Segmenting text using regex pattern: '{pattern}'")
    segments = re.split(pattern, text)
    # Filter out empty segments that might result from splitting
    filtered_segments = [segment.strip() for segment in segments if segment and segment.strip()]
    logging.debug(f"Found {len(filtered_segments)} non-empty segments initially.")
    return filtered_segments

def clean_chunk(text: str) -> str:
    """
    Cleans a text chunk by stripping leading/trailing whitespace 
    and replacing contiguous internal whitespace with a single space.
    
    Args:
        text (str): The text chunk to clean.
        
    Returns:
        str: The cleaned text chunk.
    """
    # Step 1: Strip leading/trailing whitespace
    cleaned_text = text.strip()
    # Step 2: Replace one or more whitespace characters (\s+) with a single space ' '
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text

def create_metachunks(chunks: List[str], metachunk_size: int, internal_separator: str) -> List[str]:
    """
    Combines consecutive chunks into metachunks.
    
    Args:
        chunks (List[str]): The list of individual text chunks (already cleaned).
        metachunk_size (int): The number of chunks to combine into one metachunk.
        internal_separator (str): The string to use to join chunks within a metachunk.
        
    Returns:
        List[str]: A list of metachunks.
    """
    if not chunks:
        return []
    if metachunk_size <= 0:
        logging.warning("Metachunk size must be positive. Defaulting to 1.")
        metachunk_size = 1
        
    metachunks = []
    num_chunks = len(chunks)
    
    for i in range(0, num_chunks, metachunk_size):
        # Get the slice of chunks for the current metachunk
        chunk_group = chunks[i : i + metachunk_size]
        # Join the chunks in the group using the specified separator
        metachunk = internal_separator.join(chunk_group)
        metachunks.append(metachunk)
        
    logging.info(f"Combined {num_chunks} chunks into {len(metachunks)} metachunks (target size: {metachunk_size})")
    return metachunks

def write_metachunks_to_files(metachunks: List[str], file_count: int, output_dir: Path, corpus_name: str, file_separator: str):
    """
    Distributes metachunks into a specified number of files.
    
    Args:
        metachunks (List[str]): The list of metachunks to distribute.
        file_count (int): The desired number of output files.
        output_dir (Path): The directory to write the output files to.
        corpus_name (str): The name of the corpus, used for filenames.
        file_separator (str): Separator used between metachunks *within* an output file.
    """
    if not metachunks:
        logging.warning("No metachunks to write.")
        return
        
    if file_count <= 0:
        logging.warning("File count must be positive. Defaulting to 1.")
        file_count = 1
        
    # Ensure the output directory exists
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Ensured output directory exists: {output_dir}")
    except OSError as e:
        logging.error(f"Could not create output directory '{output_dir}': {e}")
        raise
        
    total_metachunks = len(metachunks)
    # Use math.ceil to ensure all metachunks are included, even if division isn't perfect
    metachunks_per_file = math.ceil(total_metachunks / file_count)
    
    if metachunks_per_file == 0 and total_metachunks > 0:
        # Handle edge case where file_count > total_metachunks
        metachunks_per_file = 1
        logging.warning(f"File count ({file_count}) is greater than total metachunks ({total_metachunks}). Each file will contain at most one metachunk.")
        # Adjust file_count to not create empty files unnecessarily
        file_count = total_metachunks
        
    logging.info(f"Distributing {total_metachunks} metachunks into {file_count} files (approx. {metachunks_per_file} metachunks per file).")
    
    metachunk_idx = 0
    files_written = 0
    for i in range(file_count):
        # Determine the slice of metachunks for this file
        start_idx = metachunk_idx
        end_idx = min(metachunk_idx + metachunks_per_file, total_metachunks)
        
        # Get the metachunks for the current file
        file_metachunks = metachunks[start_idx:end_idx]
        
        if not file_metachunks:
            # This can happen if file_count > total_metachunks
            logging.debug(f"Skipping file {i+1} as there are no more metachunks.")
            continue 
            
        # Format filename with zero-padding (e.g., 001, 002, ..., 010)
        # Determine padding width based on file_count
        padding_width = len(str(file_count))
        filename = f"{i+1:0{padding_width}d}_{corpus_name}_metachunks.txt"
        filepath = output_dir / filename
        
        logging.debug(f"Writing file {i+1}/{file_count}: {filepath} ({len(file_metachunks)} metachunks)")
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f_out:
                # Join the metachunks with the specified file separator and write
                file_content = file_separator.join(file_metachunks)
                f_out.write(file_content)
            files_written += 1
        except IOError as e:
            logging.error(f"Could not write to output file '{filepath}': {e}")
            # Decide whether to continue or raise the error
            # For robustness, let's log and continue to process other files
            continue 
            
        # Update the index for the next iteration
        metachunk_idx = end_idx
        
        # Safety break if index goes beyond total (shouldn't happen with min())
        if metachunk_idx >= total_metachunks:
            break
            
    logging.info(f"Successfully wrote {files_written} files to '{output_dir}'.")
    if files_written < file_count and total_metachunks > 0:
         logging.warning(f"Note: Wrote fewer files ({files_written}) than requested ({file_count}) because the number of metachunks was limited.")


# --- Main Processing Function ---

def process_document(input_file: Path, 
                     output_dir: Path, 
                     segment_pattern: str, 
                     metachunk_size: int, 
                     file_count: int, 
                     corpus_name: str,
                     internal_separator: str,
                     file_separator: str):
    """
    Main function to orchestrate the document processing steps.
    """
    try:
        # 1. Read the input file
        logging.info(f"Reading input file: {input_file}")
        input_text = read_input_file(input_file)
        logging.info(f"Successfully read input file ({len(input_text)} characters).")
        
        # 2. Segment the text into initial chunks
        logging.info(f"Segmenting text using pattern: {segment_pattern}")
        initial_chunks = segment_text(input_text, segment_pattern)
        logging.info(f"Initial segmentation yielded {len(initial_chunks)} chunks.")
        
        if not initial_chunks:
            logging.warning("No chunks were found after segmentation. Exiting.")
            return False
            
        # 3. Clean each chunk
        logging.info("Cleaning individual chunks (whitespace standardization)...")
        cleaned_chunks = [clean_chunk(chunk) for chunk in initial_chunks]
        # Filter out any chunks that became empty *after* cleaning (unlikely but possible)
        cleaned_chunks = [chunk for chunk in cleaned_chunks if chunk]
        logging.info(f"Processing {len(cleaned_chunks)} non-empty cleaned chunks.")
        
        if not cleaned_chunks:
            logging.warning("No non-empty chunks remained after cleaning. Exiting.")
            return False

        # 4. Combine cleaned chunks into metachunks
        logging.info(f"Combining cleaned chunks into metachunks of size {metachunk_size}.")
        metachunks = create_metachunks(cleaned_chunks, metachunk_size, internal_separator)
        
        if not metachunks:
             logging.warning("No metachunks were created. Exiting.")
             return False
             
        # 5. Divide metachunks into files and write them
        logging.info(f"Distributing {len(metachunks)} metachunks into {file_count} files in directory: {output_dir}")
        write_metachunks_to_files(metachunks, file_count, output_dir, corpus_name, file_separator)
        
        logging.info("=== Processing Complete ===")
        return True

    except FileNotFoundError:
        # Already logged in read_input_file
        return False
    except ValueError as e:
        # Could be from read_input_file or other issues
        logging.error(f"Processing failed due to a value error: {e}")
        return False
    except OSError as e:
        # Could be from directory creation or file writing
        logging.error(f"Processing failed due to an OS error: {e}")
        return False
    except Exception as e:
        logging.exception(f"An unexpected critical error occurred during processing: {e}")
        return False

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment a document, create metachunks, and distribute them into multiple files.")
    
    parser.add_argument("--input-file", type=Path, default=INPUT_FILE,
                        help=f"Path to the input text file (default: {INPUT_FILE}).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_SUBDIR,
                        help=f"Directory to save the output segment files (default: {DEFAULT_OUTPUT_SUBDIR}).")
    parser.add_argument("--corpus-name", type=str, default=CORPUS_NAME,
                        help=f"Name of the corpus, used for subdirectories and filenames (default: {CORPUS_NAME}).")
    parser.add_argument("--segment-pattern", type=str, default=SEGMENT_SEP_DEFAULT,
                        help=f"Regex pattern to split the document into initial chunks (default: '{SEGMENT_SEP_DEFAULT}').")
    parser.add_argument("--metachunk-size", type=int, default=METACHUNK_COUNT_DEFAULT,
                        help=f"Number of consecutive chunks to combine into one metachunk (default: {METACHUNK_COUNT_DEFAULT}).")
    parser.add_argument("--file-count", type=int, default=FILE_COUNT_DEFAULT,
                        help=f"Number of output files to distribute the metachunks into (default: {FILE_COUNT_DEFAULT}).")
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug logging.')

    args = parser.parse_args()

    # --- Update Configuration from Arguments ---
    # Handle corpus name potentially changing base paths
    if args.corpus_name != CORPUS_NAME:
        CORPUS_NAME = args.corpus_name
        # Recalculate paths based on the potentially new corpus name
        INPUT_FILE = BASE_DATA_DIR / CORPUS_NAME / "document" / f"{CORPUS_NAME}.txt"
        # Output dir is handled directly by args.output_dir if provided, 
        # but if default is used, recalculate it.
        if args.output_dir == DEFAULT_OUTPUT_SUBDIR:
             args.output_dir = BASE_DATA_DIR / CORPUS_NAME / "segments_metachunks"
        # Note: Log directory setup happens after this, so it will use the updated CORPUS_NAME
        
    # Use resolved paths for clarity and consistency
    input_file_path = args.input_file.resolve()
    output_dir_path = args.output_dir.resolve()

    # Setup logging (now uses potentially updated CORPUS_NAME)
    log_file = setup_logging()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("DEBUG logging enabled.")
        # Log arguments explicitly in debug mode
        logging.debug(f"Arguments received: {args}")
        
    logging.info(f"Script started: step1c_segment_document_metachunks-files.py")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Processing Corpus: {CORPUS_NAME}")
    logging.info(f"Input file: {input_file_path}")
    logging.info(f"Output directory: {output_dir_path}")
    logging.info(f"Segmentation pattern: '{args.segment_pattern}'")
    logging.info(f"Chunks per metachunk: {args.metachunk_size}")
    logging.info(f"Number of output files: {args.file_count}")
    logging.info(f"Internal metachunk separator: '{METACHUNK_INTERNAL_SEP}'")
    logging.info(f"Separator between metachunks in file: '{METACHUNK_FILE_SEP}'")


    # --- Execute Main Processing ---
    success = process_document(
        input_file=input_file_path,
        output_dir=output_dir_path,
        segment_pattern=args.segment_pattern,
        metachunk_size=args.metachunk_size,
        file_count=args.file_count,
        corpus_name=CORPUS_NAME, # Use the potentially updated corpus name
        internal_separator=METACHUNK_INTERNAL_SEP,
        file_separator=METACHUNK_FILE_SEP
    )

    # --- Exit ---
    if success:
        logging.info("Script finished successfully.")
        exit(0)
    else:
        logging.error("Script finished with errors.")
        exit(1)