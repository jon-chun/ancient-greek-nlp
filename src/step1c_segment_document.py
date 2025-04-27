#!/usr/bin/env python3
"""
segment_text.py - A tool to segment Ancient Greek biblical text into chunks of appropriate size.
"""

import re
import os
import argparse
from pathlib import Path
from typing import List, Union, Callable

# Default configuration
CORPUS_NAME = "bible_ancient-greek"
INPUT_TEXTFILE = Path("../data") / CORPUS_NAME / "document" / f"{CORPUS_NAME}.txt"
OUTPUT_SUBDIR = Path("../data") / CORPUS_NAME / "segments"
TARGET_SEGMENT_CHARS = 2000  # Target size for each segment

def detect_language(text):
    """
    Simple language detection for Ancient Greek text.
    Checks for presence of Greek characters.
    """
    greek_chars = set('αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ')
    text_chars = set(text.lower())
    
    # If there's significant overlap with Greek characters, assume it's Greek
    if len(text_chars.intersection(greek_chars)) > 5:
        return "Ancient Greek"
    else:
        return "Unknown"

def read_file(file_path):
    """
    Read a text file with appropriate encoding detection for Ancient Greek.
    """
    encodings = ['utf-8-sig', 'utf-8', 'utf-16', 'iso-8859-7', 'windows-1253']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                text = file.read()
                
                # Check if the text appears to be valid
                if detect_language(text[:1000]) == "Ancient Greek":
                    print(f"Successfully read file using {encoding} encoding")
                    return text
        except UnicodeDecodeError:
            continue
    
    # If all encodings fail, try with error handling
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        text = file.read()
        print("Warning: File read with replacement characters due to encoding issues")
        return text

def split_into_paragraphs(text):
    """
    Split text into paragraphs using multiple newlines as separators.
    """
    # Split on two or more newlines
    paragraphs = re.split(r'\n{2,}', text)
    
    # Filter out empty paragraphs and strip whitespace
    return [para.strip() for para in paragraphs if para.strip()]

def create_segments_by_target_size(paragraphs, target_chars):
    """
    Create segments by grouping paragraphs to reach the target character count.
    
    This function ensures:
    1. No paragraphs are split across segments
    2. Segments are as close as possible to the target size
    3. All segments (except possibly the last one) are within a reasonable
       range of the target size
    """
    if not paragraphs:
        return []
    
    segments = []
    current_segment = []
    current_length = 0
    
    for paragraph in paragraphs:
        para_length = len(paragraph)
        
        # If adding this paragraph would exceed 1.5x the target and we already have content,
        # start a new segment
        if (current_length >= target_chars and 
            current_length + para_length > target_chars * 1.5):
            
            # Join the current segment and add to results
            segment_text = "\n\n".join(current_segment)
            segments.append(segment_text)
            
            # Start a new segment with this paragraph
            current_segment = [paragraph]
            current_length = para_length
        else:
            # Add to the current segment
            current_segment.append(paragraph)
            current_length += para_length
            
            # Add newlines to the length calculation
            if len(current_segment) > 1:
                current_length += 2  # Account for "\n\n" between paragraphs
    
    # Add the last segment if it's not empty
    if current_segment:
        segment_text = "\n\n".join(current_segment)
        segments.append(segment_text)
    
    # Check the distribution of segment sizes
    check_segments(segments, target_chars)
    
    # If we have very uneven segments, try to balance them
    segments = balance_segments(segments, target_chars)
    
    return segments

def check_segments(segments, target_chars):
    """
    Check the size distribution of segments and log information.
    """
    if not segments:
        print("Warning: No segments created!")
        return
    
    lengths = [len(segment) for segment in segments]
    min_length = min(lengths)
    max_length = max(lengths)
    avg_length = sum(lengths) / len(lengths)
    
    print(f"\nSegment size check:")
    print(f"Target size: {target_chars} characters")
    print(f"Number of segments: {len(segments)}")
    print(f"Minimum size: {min_length} characters ({min_length/target_chars:.2f}x target)")
    print(f"Maximum size: {max_length} characters ({max_length/target_chars:.2f}x target)")
    print(f"Average size: {avg_length:.1f} characters ({avg_length/target_chars:.2f}x target)")
    
    # Alert if any segments are much too large
    large_segments = [(i, length) for i, length in enumerate(lengths, 1) 
                       if length > target_chars * 2]
    if large_segments:
        print(f"Warning: {len(large_segments)} segments are more than 2x the target size:")
        for i, length in large_segments[:5]:  # Show first 5 large segments
            print(f"  Segment {i}: {length} chars ({length/target_chars:.2f}x target)")
        if len(large_segments) > 5:
            print(f"  ... and {len(large_segments)-5} more")

def balance_segments(segments, target_chars):
    """
    Attempt to balance segment sizes by redistributing content from very large
    segments to smaller ones, while respecting paragraph boundaries.
    """
    if len(segments) <= 1:
        return segments
    
    # First, re-split all segments into their paragraphs
    all_paragraphs = []
    for segment in segments:
        all_paragraphs.extend(split_into_paragraphs(segment))
    
    # Then redistribute the paragraphs more evenly
    return create_segments_by_smart_distribution(all_paragraphs, target_chars)

def create_segments_by_smart_distribution(paragraphs, target_chars):
    """
    Create segments using a smarter distribution algorithm that aims for
    more consistently sized segments.
    """
    if not paragraphs:
        return []
    
    # Calculate paragraph sizes
    para_sizes = [len(p) for p in paragraphs]
    
    segments = []
    current_segment = []
    current_size = 0
    
    # First pass: group paragraphs together
    for i, (para, size) in enumerate(zip(paragraphs, para_sizes)):
        # Should we start a new segment?
        # We start a new segment if:
        # 1. The current segment is already at or above target size AND
        # 2. Adding this paragraph would make it significantly larger
        if (current_size >= target_chars and 
            current_size + size > target_chars * 1.3):
            
            # Complete current segment
            segments.append("\n\n".join(current_segment))
            current_segment = []
            current_size = 0
        
        # Add paragraph to current segment
        current_segment.append(para)
        current_size += size
        
        # Add newline length if not the first paragraph
        if len(current_segment) > 1:
            current_size += 2  # For "\n\n"
    
    # Add the final segment if not empty
    if current_segment:
        segments.append("\n\n".join(current_segment))
    
    # Second pass: check if we can improve balance by moving paragraphs
    # between adjacent segments
    balanced_segments = optimize_segment_distribution(segments, target_chars)
    
    # Log the final segment distribution
    check_segments(balanced_segments, target_chars)
    
    return balanced_segments

def optimize_segment_distribution(segments, target_chars):
    """
    Try to optimize the distribution of content across segments.
    
    This function attempts to reduce the variance in segment sizes by moving
    paragraphs between adjacent segments.
    """
    if len(segments) <= 1:
        return segments
    
    # Re-split segments into paragraphs for redistribution
    segment_paragraphs = [split_into_paragraphs(segment) for segment in segments]
    
    # Calculate segment sizes
    segment_sizes = [len(segment) for segment in segments]
    
    # Optimize by moving paragraphs from larger segments to smaller ones
    changes_made = True
    iteration = 0
    max_iterations = 5  # Limit iterations to prevent excessive processing
    
    while changes_made and iteration < max_iterations:
        changes_made = False
        iteration += 1
        
        for i in range(len(segment_paragraphs) - 1):
            # Check if moving the last paragraph from segment i to segment i+1 would
            # improve overall balance
            if not segment_paragraphs[i]:
                continue
                
            current_diff = abs(segment_sizes[i] - segment_sizes[i+1])
            
            # Size of the last paragraph in segment i
            last_para_size = len(segment_paragraphs[i][-1])
            
            # Calculate what the sizes would be after moving the paragraph
            new_size_i = segment_sizes[i] - last_para_size - (2 if len(segment_paragraphs[i]) > 1 else 0)
            new_size_i_plus_1 = segment_sizes[i+1] + last_para_size + (2 if segment_paragraphs[i+1] else 0)
            
            new_diff = abs(new_size_i - new_size_i_plus_1)
            
            # If moving would improve balance and not make either segment too small
            if new_diff < current_diff and new_size_i >= target_chars * 0.5:
                # Move the paragraph
                para_to_move = segment_paragraphs[i].pop()
                segment_paragraphs[i+1].insert(0, para_to_move)
                
                # Update segment sizes
                segment_sizes[i] = new_size_i
                segment_sizes[i+1] = new_size_i_plus_1
                
                changes_made = True
    
    # Reassemble segments from paragraphs
    balanced_segments = []
    for paras in segment_paragraphs:
        if paras:
            balanced_segments.append("\n\n".join(paras))
    
    return balanced_segments

def analyze_segment_stats(segments, target_chars):
    """
    Analyze and display detailed statistics about the segments.
    """
    if not segments:
        print("No segments to analyze")
        return
    
    lengths = [len(segment) for segment in segments]
    min_length = min(lengths)
    max_length = max(lengths)
    avg_length = sum(lengths) / len(lengths)
    
    print(f"\nSegment Statistics:")
    print(f"Number of segments: {len(segments)}")
    print(f"Total text length: {sum(lengths)} characters")
    print(f"Minimum segment length: {min_length} characters ({min_length/target_chars:.2f}x target)")
    print(f"Maximum segment length: {max_length} characters ({max_length/target_chars:.2f}x target)")
    print(f"Average segment length: {avg_length:.1f} characters ({avg_length/target_chars:.2f}x target)")
    
    # Calculate standard deviation
    variance = sum((length - avg_length) ** 2 for length in lengths) / len(lengths)
    std_dev = variance ** 0.5
    print(f"Standard deviation: {std_dev:.1f} characters ({std_dev/target_chars:.2f}x target)")
    
    # Show distribution relative to target
    under_target = sum(1 for length in lengths if length < target_chars)
    near_target = sum(1 for length in lengths if target_chars * 0.9 <= length <= target_chars * 1.1)
    over_target = sum(1 for length in lengths if length > target_chars * 1.5)
    
    print(f"\nSize distribution:")
    print(f"Segments < target: {under_target} ({under_target/len(segments)*100:.1f}%)")
    print(f"Segments within ±10% of target: {near_target} ({near_target/len(segments)*100:.1f}%)")
    print(f"Segments > 1.5x target: {over_target} ({over_target/len(segments)*100:.1f}%)")
    
    # Show histogram of segment sizes
    print("\nSize histogram (relative to target):")
    bins = [(0, 0.5), (0.5, 0.9), (0.9, 1.1), (1.1, 1.5), (1.5, 2.0), (2.0, float('inf'))]
    bin_labels = ["<50%", "50-90%", "90-110%", "110-150%", "150-200%", ">200%"]
    
    for (lower, upper), label in zip(bins, bin_labels):
        count = sum(1 for length in lengths if lower * target_chars <= length < upper * target_chars)
        bar = '#' * int(count / len(segments) * 50)
        print(f"{label}: {bar} {count} ({count/len(segments)*100:.1f}%)")

def write_segments(segments, output_dir, corpus_name, prefix="segment"):
    """
    Write each segment to a separate file in the output directory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write each segment to a file with UTF-8 encoding
    for i, segment in enumerate(segments, 1):
        output_file = output_dir / f"{prefix}_{i}_{corpus_name}.txt"
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(segment)
        print(f"Wrote {prefix} {i} to {output_file} ({len(segment)} characters)")

def parse_arguments():
    """
    Parse command-line arguments for flexible configuration.
    """
    parser = argparse.ArgumentParser(description='Segment Ancient Greek biblical text into chunks of appropriate size.')
    
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Path to input text file (default: auto-generated based on corpus name)')
    
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to output directory (default: auto-generated based on corpus name)')
    
    parser.add_argument('--corpus', '-c', type=str, default=CORPUS_NAME,
                        help=f'Name of the corpus (default: {CORPUS_NAME})')
    
    parser.add_argument('--target-chars', '-t', type=int, default=TARGET_SEGMENT_CHARS,
                        help=f'Target characters per segment (default: {TARGET_SEGMENT_CHARS})')
    
    parser.add_argument('--prefix', '-p', type=str, default="segment",
                        help='Prefix for output filenames (default: "segment")')
    
    parser.add_argument('--analyze-only', '-a', action='store_true',
                        help='Only analyze the text without writing files')
    
    args = parser.parse_args()
    
    # Derive input and output paths if not specified
    if args.input is None:
        args.input = Path("../data") / args.corpus / "document" / f"{args.corpus}.txt"
    else:
        args.input = Path(args.input)
    
    if args.output is None:
        args.output = Path("../data") / args.corpus / "segments"
    else:
        args.output = Path(args.output)
    
    return args

def main():
    """
    Main function with improved segmentation logic.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Check if input file exists
    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist.")
        return
    
    # Read the text file
    text = read_file(args.input)
    print(f"Read {len(text)} characters from {args.input}")
    
    # Split into paragraphs
    paragraphs = split_into_paragraphs(text)
    print(f"Split text into {len(paragraphs)} paragraphs")
    
    # Create segments with target size
    segments = create_segments_by_target_size(paragraphs, args.target_chars)
    print(f"Created {len(segments)} segments")
    
    # Display detailed segment statistics
    analyze_segment_stats(segments, args.target_chars)
    
    # Write segments to files if not in analyze-only mode
    if not args.analyze_only:
        write_segments(segments, args.output, args.corpus, args.prefix)
        print(f"Text segmentation complete. Files written to {args.output}")
    else:
        print("Analysis complete. No files written (analyze-only mode).")

if __name__ == "__main__":
    main()