#!/usr/bin/env python3
"""
step1c_segment_document_metachunks-files.py - Segments an Ancient Greek text,
                                             handles long segments, cleans,
                                             combines into metachunks,
                                             and distributes into multiple files.

This script:
1. Reads an Ancient Greek text file (handling various encodings).
2. Segments the text into initial chunks based on a regex pattern (e.g., double newlines).
3. Further splits any chunk exceeding MAX_CHUNK_CHARS based on sentence terminators (.?!).
4. Cleans *each final chunk* by stripping whitespace and collapsing internal whitespace to single spaces.
5. Combines consecutive cleaned chunks into larger "metachunks" (joined by a single space).
6. Divides the metachunks into a specified number of output files.
7. Saves each group of metachunks to a separate file (one metachunk per line)
   in a designated subdirectory.
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
METACHUNK_COUNT_DEFAULT = 2  # Number of final cleaned chunks per metachunk
FILE_COUNT_DEFAULT = 5      # Number of output files to distribute metachunks into

# Processing configuration
SEGMENT_SEP_DEFAULT = r"[\n]{2,}"  # Regex for *initial* chunk separation
MAX_CHUNK_CHARS = 110           # Max chars for a chunk before sentence splitting
# Regex to split long chunks, capturing terminators. Handles optional space after.
SENTENCE_SPLIT_PATTERN = r'([.?!])\s*'
METACHUNK_INTERNAL_SEP = " " # Separator used *within* a metachunk (between final cleaned chunks)
METACHUNK_FILE_SEP = "\n"   # Separator used *between* metachunks within an output file

# --- Logging Setup ---

def setup_logging():
    """Sets up logging to file and console."""
    log_base_dir = BASE_DATA_DIR / CORPUS_NAME / "logs"
    log_base_dir.mkdir(parents=True, exist_ok=True)
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
                
            if content and len(content) > 10:
                if re.search(r'[\u0370-\u03FF\u1F00-\u1FFF]|[.,;·\s]', content):
                    logging.info(f"Successfully read file '{file_path}' using {encoding} encoding")
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
            continue
            
    error_msg = f"Could not decode file '{file_path}' with any of the attempted encodings: {encodings_to_try}"
    logging.error(error_msg)
    raise ValueError(error_msg)

def segment_text_initial(text: str, pattern: str) -> List[str]:
    """
    Performs the *initial* splitting of text based on the primary separator pattern.
    """
    logging.debug(f"Performing initial segmentation using regex pattern: '{pattern}'")
    segments = re.split(pattern, text)
    filtered_segments = [segment.strip() for segment in segments if segment and segment.strip()]
    logging.debug(f"Found {len(filtered_segments)} non-empty initial segments.")
    return filtered_segments

def split_long_chunk(chunk: str, max_len: int, split_pattern: str) -> List[str]:
    """
    Splits a chunk if it exceeds max_len, using sentence terminators.
    Tries to keep terminators with their sentences.
    """
    if len(chunk) <= max_len:
        return [chunk] # No need to split

    logging.debug(f"Chunk exceeds {max_len} chars ({len(chunk)}), attempting sentence split.")
    # Use re.split with capturing group to keep delimiters
    parts = re.split(split_pattern, chunk)
    
    result_chunks = []
    current_sentence = ""
    
    # Iterate through the parts: [sentence1, delim1, sentence2, delim2, ...]
    # Need to handle potential empty strings from split if delimiter is at start/end
    i = 0
    while i < len(parts):
        text_part = parts[i]
        current_sentence += text_part # Add the text part
        
        # Look ahead for the delimiter part
        if i + 1 < len(parts):
            delimiter_part = parts[i+1]
            current_sentence += delimiter_part # Add the delimiter
            # Add the reconstructed sentence (strip extraneous whitespace added by split logic)
            cleaned_sentence = current_sentence.strip()
            if cleaned_sentence: # Avoid adding empty strings
                 result_chunks.append(cleaned_sentence)
            current_sentence = "" # Reset for next sentence
            i += 2 # Move past text and delimiter
        else:
            # Last part (might not have a delimiter)
            cleaned_sentence = current_sentence.strip()
            if cleaned_sentence:
                result_chunks.append(cleaned_sentence)
            i += 1 # Move past the last text part
            
    # Filter out potentially empty strings again just in case
    final_sub_chunks = [sub for sub in result_chunks if sub]
    logging.debug(f"Split long chunk into {len(final_sub_chunks)} sub-chunks.")
    return final_sub_chunks


def clean_chunk(text: str) -> str:
    """
    Cleans a text chunk: strips outer whitespace, collapses internal whitespace to single spaces.
    """
    cleaned_text = text.strip()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text

def create_metachunks(final_cleaned_chunks: List[str], metachunk_size: int, internal_separator: str) -> List[str]:
    """
    Combines consecutive *final, cleaned* chunks into metachunks.
    """
    if not final_cleaned_chunks:
        return []
    if metachunk_size <= 0:
        logging.warning("Metachunk size must be positive. Defaulting to 1.")
        metachunk_size = 1
        
    metachunks = []
    num_chunks = len(final_cleaned_chunks)
    
    for i in range(0, num_chunks, metachunk_size):
        chunk_group = final_cleaned_chunks[i : i + metachunk_size]
        # Join the cleaned chunks with the specified separator (e.g., single space)
        metachunk = internal_separator.join(chunk_group)
        metachunks.append(metachunk) # Already cleaned, no extra strip needed
        
    logging.info(f"Combined {num_chunks} final cleaned chunks into {len(metachunks)} metachunks (target size: {metachunk_size})")
    return metachunks

def write_metachunks_to_files(metachunks: List[str], file_count: int, output_dir: Path, corpus_name: str, file_separator: str):
    """
    Distributes metachunks into a specified number of files.
    Writes one metachunk per line in each file.
    """
    if not metachunks:
        logging.warning("No metachunks to write.")
        return
        
    if file_count <= 0:
        logging.warning("File count must be positive. Defaulting to 1.")
        file_count = 1
        
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Ensured output directory exists: {output_dir}")
    except OSError as e:
        logging.error(f"Could not create output directory '{output_dir}': {e}")
        raise
        
    total_metachunks = len(metachunks)
    metachunks_per_file = math.ceil(total_metachunks / file_count)
    
    if metachunks_per_file == 0 and total_metachunks > 0:
        metachunks_per_file = 1
        logging.warning(f"File count ({file_count}) >= total metachunks ({total_metachunks}). Each file will contain at most one metachunk.")
        file_count = total_metachunks # Adjust to avoid empty files
        
    logging.info(f"Distributing {total_metachunks} metachunks into {file_count} files (approx. {metachunks_per_file} metachunks per file).")
    
    metachunk_idx = 0
    files_written = 0
    for i in range(file_count):
        start_idx = metachunk_idx
        end_idx = min(metachunk_idx + metachunks_per_file, total_metachunks)
        file_metachunks = metachunks[start_idx:end_idx]
        
        if not file_metachunks:
            logging.debug(f"Skipping file {i+1} as there are no more metachunks.")
            continue
            
        padding_width = len(str(file_count))
        # Adjusted filename slightly
        filename = f"{i+1:0{padding_width}d}_{corpus_name}_metachunks.txt" 
        filepath = output_dir / filename
        
        logging.debug(f"Writing file {i+1}/{file_count}: {filepath} ({len(file_metachunks)} metachunks)")
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f_out:
                # Join the metachunks for *this file* with the file separator (newline)
                # This ensures one metachunk per line.
                file_content = file_separator.join(file_metachunks)
                f_out.write(file_content)
                # Optionally add a final newline if the separator itself doesn't add one
                if not file_content.endswith('\n'):
                    f_out.write('\n')
            files_written += 1
        except IOError as e:
            logging.error(f"Could not write to output file '{filepath}': {e}")
            continue
            
        metachunk_idx = end_idx
        if metachunk_idx >= total_metachunks:
            break
            
    logging.info(f"Successfully wrote {files_written} files to '{output_dir}'.")
    if files_written < file_count and total_metachunks > 0:
         logging.warning(f"Note: Wrote fewer files ({files_written}) than requested ({file_count}) because the number of metachunks was limited or file count was high.")


# --- Main Processing Function ---

def process_document(input_file: Path,
                     output_dir: Path,
                     initial_segment_pattern: str,
                     max_chunk_chars: int,
                     sentence_split_pattern: str,
                     metachunk_size: int,
                     file_count: int,
                     corpus_name: str,
                     internal_separator: str,
                     file_separator: str):
    """
    Main function to orchestrate the document processing steps in the correct order.
    """
    try:
        # 1. Read the input file
        logging.info(f"Reading input file: {input_file}")
        input_text = read_input_file(input_file)
        logging.info(f"Successfully read input file ({len(input_text)} characters).")

        # 2. Initial Segmentation
        logging.info(f"Performing initial segmentation using pattern: {initial_segment_pattern}")
        initial_chunks = segment_text_initial(input_text, initial_segment_pattern)
        logging.info(f"Initial segmentation yielded {len(initial_chunks)} chunks.")

        if not initial_chunks:
            logging.warning("No chunks found after initial segmentation. Exiting.")
            return False

        # 3. Secondary Segmentation (for long chunks)
        logging.info(f"Performing secondary split for chunks longer than {max_chunk_chars} characters.")
        intermediate_chunks = []
        for chunk in initial_chunks:
            intermediate_chunks.extend(split_long_chunk(chunk, max_chunk_chars, sentence_split_pattern))
        logging.info(f"After secondary split, resulted in {len(intermediate_chunks)} intermediate chunks.")

        if not intermediate_chunks:
             logging.warning("No chunks remained after secondary splitting. Exiting.")
             return False

        # 4. Clean *each* resulting chunk
        logging.info("Cleaning all individual chunks (whitespace standardization)...")
        final_cleaned_chunks = [clean_chunk(chunk) for chunk in intermediate_chunks]
        # Filter out any chunks that became empty *after* cleaning
        final_cleaned_chunks = [chunk for chunk in final_cleaned_chunks if chunk]
        logging.info(f"Processing {len(final_cleaned_chunks)} non-empty, final cleaned chunks.")

        if not final_cleaned_chunks:
            logging.warning("No non-empty chunks remained after cleaning. Exiting.")
            return False

        # 5. Combine cleaned chunks into metachunks
        logging.info(f"Combining final cleaned chunks into metachunks of size {metachunk_size}.")
        metachunks = create_metachunks(final_cleaned_chunks, metachunk_size, internal_separator)

        if not metachunks:
             logging.warning("No metachunks were created. Exiting.")
             return False

        # 6. Divide metachunks into files and write them
        logging.info(f"Distributing {len(metachunks)} metachunks into {file_count} files in directory: {output_dir}")
        write_metachunks_to_files(metachunks, file_count, output_dir, corpus_name, file_separator)

        logging.info("=== Processing Complete ===")
        return True

    except FileNotFoundError:
        return False
    except ValueError as e:
        logging.error(f"Processing failed due to a value error: {e}")
        return False
    except OSError as e:
        logging.error(f"Processing failed due to an OS error: {e}")
        return False
    except Exception as e:
        logging.exception(f"An unexpected critical error occurred during processing: {e}")
        return False

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment a document, handle long chunks, clean, create metachunks, and distribute into multiple files.")
    
    parser.add_argument("--input-file", type=Path, default=INPUT_FILE,
                        help=f"Path to the input text file (default: {INPUT_FILE}).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_SUBDIR,
                        help=f"Directory to save the output segment files (default: {DEFAULT_OUTPUT_SUBDIR}).")
    parser.add_argument("--corpus-name", type=str, default=CORPUS_NAME,
                        help=f"Name of the corpus, used for subdirectories and filenames (default: {CORPUS_NAME}).")
    parser.add_argument("--segment-pattern", type=str, default=SEGMENT_SEP_DEFAULT,
                        help=f"Regex pattern for *initial* document splitting (default: '{SEGMENT_SEP_DEFAULT}').")
    parser.add_argument("--max-chunk-chars", type=int, default=MAX_CHUNK_CHARS,
                        help=f"Maximum characters per chunk before attempting sentence split (default: {MAX_CHUNK_CHARS}).")
    parser.add_argument("--metachunk-size", type=int, default=METACHUNK_COUNT_DEFAULT,
                        help=f"Number of consecutive *final cleaned* chunks per metachunk (default: {METACHUNK_COUNT_DEFAULT}).")
    parser.add_argument("--file-count", type=int, default=FILE_COUNT_DEFAULT,
                        help=f"Number of output files to distribute metachunks into (default: {FILE_COUNT_DEFAULT}).")
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug logging.')

    args = parser.parse_args()

    # --- Update Configuration from Arguments ---
    if args.corpus_name != CORPUS_NAME:
        CORPUS_NAME = args.corpus_name
        INPUT_FILE = BASE_DATA_DIR / CORPUS_NAME / "document" / f"{CORPUS_NAME}.txt"
        if args.output_dir == DEFAULT_OUTPUT_SUBDIR:
             args.output_dir = BASE_DATA_DIR / CORPUS_NAME / "segments_metachunks"
        
    input_file_path = args.input_file.resolve()
    output_dir_path = args.output_dir.resolve()

    # Setup logging
    log_file = setup_logging()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("DEBUG logging enabled.")
        logging.debug(f"Arguments received: {args}")
        
    logging.info(f"Script started: step1c_segment_document_metachunks-files.py")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Processing Corpus: {CORPUS_NAME}")
    logging.info(f"Input file: {input_file_path}")
    logging.info(f"Output directory: {output_dir_path}")
    logging.info(f"Initial segmentation pattern: '{args.segment_pattern}'")
    logging.info(f"Max chunk chars before sentence split: {args.max_chunk_chars}")
    logging.info(f"Sentence split pattern: '{SENTENCE_SPLIT_PATTERN}'")
    logging.info(f"Final cleaned chunks per metachunk: {args.metachunk_size}")
    logging.info(f"Number of output files: {args.file_count}")
    logging.info(f"Internal metachunk separator: '{METACHUNK_INTERNAL_SEP}' (space)")
    logging.info(f"Separator between metachunks in file: '{METACHUNK_FILE_SEP}' (newline)")

    # --- Execute Main Processing ---
    success = process_document(
        input_file=input_file_path,
        output_dir=output_dir_path,
        initial_segment_pattern=args.segment_pattern,
        max_chunk_chars=args.max_chunk_chars,
        sentence_split_pattern=SENTENCE_SPLIT_PATTERN,
        metachunk_size=args.metachunk_size,
        file_count=args.file_count,
        corpus_name=CORPUS_NAME,
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