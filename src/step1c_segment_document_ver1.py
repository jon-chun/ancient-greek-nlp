#!/usr/bin/env python3
"""
segment_text.py - A tool to segment Ancient Greek biblical text into chunks of minimum size.
"""

import re
import os
from pathlib import Path

# Global variables with corrected paths
CORPUS_NAME = "bible_ancient-greek"
INPUT_TEXTFILE = Path("../data") / CORPUS_NAME / "document" / f"{CORPUS_NAME}.txt"
OUTPUT_SUBDIR = Path("../data") / CORPUS_NAME / "segments"
MIN_SEGMENT_CHAR = 2000

def detect_language(text):
    """
    Simple language detection for Ancient Greek text.
    Checks for presence of Greek characters.
    """
    # Check for Greek characters in the text
    greek_chars = set('αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ')
    text_chars = set(text.lower())
    
    # If there's significant overlap with Greek characters, assume it's Greek
    if len(text_chars.intersection(greek_chars)) > 5:
        return "Ancient Greek"
    else:
        return "Unknown"

def read_and_split_text(file_path):
    """
    Read the text file and split it into chunks based on two or more consecutive newlines.
    Preserves the encoding of Ancient Greek text.
    """
    # Read the file with UTF-8 encoding to preserve Greek characters
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Split the text on two or more consecutive newlines
    chunks = re.split(r'\n{3,}', text)
    
    # Remove empty chunks and strip whitespace
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    return chunks

def agglomerate_chunks(chunks, min_chars):
    """
    Agglomerate small chunks until all chunks are at least min_chars in length.
    Makes multiple passes to ensure minimum chunk size is achieved where possible.
    """
    # Create a copy to work with
    result = chunks.copy()
    
    # Continue until all chunks are at least min_chars in length or we can't merge any more
    made_changes = True
    while made_changes and len(result) > 1:
        made_changes = False
        
        # If all chunks are above minimum size, we're done
        if all(len(chunk) >= min_chars for chunk in result):
            break
        
        # Find the smallest chunk
        sizes = [len(chunk) for chunk in result]
        smallest_idx = sizes.index(min(sizes))
        
        # If the smallest chunk is already large enough, we're done
        if sizes[smallest_idx] >= min_chars:
            break
        
        # Determine whether to merge with the previous or next chunk
        if smallest_idx == 0:  # First chunk
            # Merge with the next chunk
            result[0] = result[0] + "\n\n" + result[1]
            result.pop(1)
            made_changes = True
        elif smallest_idx == len(result) - 1:  # Last chunk
            # Merge with the previous chunk
            result[smallest_idx - 1] = result[smallest_idx - 1] + "\n\n" + result[smallest_idx]
            result.pop(smallest_idx)
            made_changes = True
        else:
            # Compare sizes of adjacent chunks and merge with the smaller one
            prev_size = sizes[smallest_idx - 1]
            next_size = sizes[smallest_idx + 1]
            
            if prev_size <= next_size:
                # Merge with the previous chunk
                result[smallest_idx - 1] = result[smallest_idx - 1] + "\n\n" + result[smallest_idx]
                result.pop(smallest_idx)
            else:
                # Merge with the next chunk
                result[smallest_idx] = result[smallest_idx] + "\n\n" + result[smallest_idx + 1]
                result.pop(smallest_idx + 1)
            
            made_changes = True
    
    return result

def write_segments(chunks, output_dir, corpus_name):
    """
    Write each chunk to a separate file in the output directory.
    Files are named as segment_number_{corpus_name}.txt
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write each chunk to a file with UTF-8 encoding
    for i, chunk in enumerate(chunks, 1):
        output_file = output_dir / f"{i}_{corpus_name}.txt"
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(chunk)
        print(f"Wrote segment {i} to {output_file}")

def main():
    """
    Main function to orchestrate the text segmentation process.
    """
    # Check if input file exists
    if not INPUT_TEXTFILE.exists():
        print(f"Error: Input file {INPUT_TEXTFILE} does not exist.")
        return
    
    # Read and split the text
    chunks = read_and_split_text(INPUT_TEXTFILE)
    print(f"Split text into {len(chunks)} initial chunks.")
    
    # Detect language from the first substantial chunk
    if chunks:
        lang = detect_language(chunks[0])
        print(f"Detected language: {lang}")
    
    # Agglomerate small chunks
    agglomerated_chunks = agglomerate_chunks(chunks, MIN_SEGMENT_CHAR)
    print(f"Agglomerated into {len(agglomerated_chunks)} chunks.")
    
    # Write segments to files
    write_segments(agglomerated_chunks, OUTPUT_SUBDIR, CORPUS_NAME)
    print("Text segmentation complete.")

if __name__ == "__main__":
    main()