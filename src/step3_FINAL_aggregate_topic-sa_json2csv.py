import json
import pandas as pd
from pathlib import Path

def transform_json_to_csv(input_file, news_source='bbc_arabic'):
    """
    Transform a JSON file into CSV rows based on specified output format.
    
    Args:
    input_file (Path): Path to the input JSON file
    news_source (str): Source of the news article
    
    Returns:
    pd.DataFrame: DataFrame with transformed data
    """
    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Prepare rows for CSV
    csv_rows = []
    
    # Extract article metadata
    title_en = data['article_metadata']['title_en']
    summary_en = data['article_metadata']['text_summary_en']
    filename = data['article_metadata']['original_filename']
    
    # Process each topic
    for index, topic_data in enumerate(data['topic_analysis']):
        row = {
            'index': 0,  # Will be updated later with global index
            'source': news_source,
            'title_en': title_en,
            'summary_en': summary_en,
            'filename': filename,
            'topic_en': topic_data['topic'],
            
            # Sentiment details
            'sentiment_polarity': topic_data['sentiment_polarity']['score'],
            'sentiment_confidence': topic_data['sentiment_polarity']['confidence'],
            'sentiment_reasoning': topic_data['sentiment_polarity']['reasoning'],
            
            # Emotions
            'joy_weight': topic_data['emotions']['Joy']['weight'],
            'joy_confidence': topic_data['emotions']['Joy']['confidence'],
            'joy_reasoning': topic_data['emotions']['Joy']['reasoning'],
            
            'trust_weight': topic_data['emotions']['Trust']['weight'],
            'trust_confidence': topic_data['emotions']['Trust']['confidence'],
            'trust_reasoning': topic_data['emotions']['Trust']['reasoning'],
            
            'fear_weight': topic_data['emotions']['Fear']['weight'],
            'fear_confidence': topic_data['emotions']['Fear']['confidence'],
            'fear_reasoning': topic_data['emotions']['Fear']['reasoning'],
            
            'surprise_weight': topic_data['emotions']['Surprise']['weight'],
            'surprise_confidence': topic_data['emotions']['Surprise']['confidence'],
            'surprise_reasoning': topic_data['emotions']['Surprise']['reasoning'],
            
            'sadness_weight': topic_data['emotions']['Sadness']['weight'],
            'sadness_confidence': topic_data['emotions']['Sadness']['confidence'],
            'sadness_reasoning': topic_data['emotions']['Sadness']['reasoning'],
            
            'disgust_weight': topic_data['emotions']['Disgust']['weight'],
            'disgust_confidence': topic_data['emotions']['Disgust']['confidence'],
            'digust_reasoning': topic_data['emotions']['Disgust']['reasoning'],
            
            'anger_weight': topic_data['emotions']['Anger']['weight'],
            'anger_confidence': topic_data['emotions']['Anger']['confidence'],
            'anger_reasoning': topic_data['emotions']['Anger']['reasoning'],
            
            'anticipation_weight': topic_data['emotions']['Anticipation']['weight'],
            'anticipation_confidence': topic_data['emotions']['Anticipation']['confidence'],
            'anticipation_reasoning': topic_data['emotions']['Anticipation']['reasoning']
        }
        csv_rows.append(row)
    
    return pd.DataFrame(csv_rows)

def process_all_json_files(news_sources=None):
    """
    Process all JSON files across specified news sources.
    
    Args:
    news_sources (list, optional): List of news sources to process. 
                                   Defaults to ['bbc_arabic', 'bbc_english', 'cnn_arabic', 'cnn_english']
    
    Returns:
    pd.DataFrame: Combined DataFrame of all processed JSON files
    """
    # Default news sources if not provided
    if news_sources is None:
        news_sources = ['bbc_arabic', 'bbc_english', 'cnn_arabic', 'cnn_english']
    
    # Base input directory
    base_input_dir = Path(__file__).parent.parent / 'data'
    
    # Process each news source
    all_dataframes = []
    global_index = 0
    
    for news_source in news_sources:
        # Construct input directory for this news source
        input_root_dir = base_input_dir / news_source / 'analysis_json'
        
        # Skip if directory doesn't exist
        if not input_root_dir.exists():
            print(f"Directory not found for {news_source}: {input_root_dir}")
            continue
        
        # Find all JSON files in this directory and subdirectories
        json_files = sorted(list(input_root_dir.rglob('*.json')))
        
        if not json_files:
            print(f"No JSON files found in {input_root_dir}")
            continue
        
        # Process each JSON file for this news source
        for json_file in json_files:
            try:
                df = transform_json_to_csv(json_file, news_source)
                
                # Update the index column to be sequential across all files
                df['index'] = range(global_index, global_index + len(df))
                global_index += len(df)
                
                all_dataframes.append(df)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
    
    # Combine all dataframes
    if not all_dataframes:
        print("No dataframes to combine.")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_dataframes, ignore_index=False)
    
    return combined_df

def main():
    # Define output directory
    output_dir = Path(__file__).parent.parent / 'data' / 'metadata'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # News sources to process
    NEWS_SOURCES = ['bbc_arabic', 'bbc_english', 'cnn_arabic', 'cnn_english']
    
    # Output CSV path
    output_csv_path = output_dir / 'aggregated_news_topic-sa_raw.csv'
    
    try:
        combined_df = process_all_json_files(NEWS_SOURCES)
        
        if not combined_df.empty:
            # Save to CSV
            combined_df.to_csv(output_csv_path, index=False, encoding='utf-8')
            print(f"CSV file saved successfully to {output_csv_path}")
            print(f"Total rows processed: {len(combined_df)}")
            print(f"Sources processed: {', '.join(combined_df['source'].unique())}")
        else:
            print("No data to save.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()