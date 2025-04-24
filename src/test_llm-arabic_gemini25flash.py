import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure API key (retrieve from environment variable)
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY in your .env file.")
genai.configure(api_key=API_KEY)

# Select the Gemini model
model_name = "gemini-2.5-flash-preview-04-17"
model = genai.GenerativeModel(model_name)

def translate_arabic_text(text, target_language="en"):
    """Translates Arabic text to the specified target language using the selected Gemini model."""
    try:
        prompt = f"Translate the following Arabic text to {target_language}:\n\n{text}"
        response = model.generate_content(prompt)
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            return f"Error during translation (prompt blocked): {response.prompt_feedback.block_reason}"
        if response.text:
            return response.text
        else:
            return "Error: Translation failed to produce output."
    except Exception as e:
        return f"Error during translation: {e}"

def process_arabic_text(text, instructions="Summarize this text"):
    """Processes Arabic text based on the provided instructions using the selected Gemini model."""
    try:
        prompt = f"{instructions} the following Arabic text:\n\n{text}"
        response = model.generate_content(prompt)
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            return f"Error during processing (prompt blocked): {response.prompt_feedback.block_reason}"
        if response.text:
            return response.text
        else:
            return "Error: Processing failed to produce output."
    except Exception as e:
        return f"Error during processing: {e}"

if __name__ == "__main__":
    # Create a .env file in the same directory as your Python script
    # and add your API key like this:
    # GOOGLE_API_KEY=YOUR_ACTUAL_API_KEY

    arabic_text_to_translate = "هذا مثال لنص عربي نريد ترجمته إلى الإنجليزية."
    target_language = "en"
    translated_text = translate_arabic_text(arabic_text_to_translate, target_language)
    print(f"Original Arabic Text: {arabic_text_to_translate}")
    print(f"Translated to {target_language}: {translated_text}")

    arabic_text_to_summarize = """تعتبر اللغة العربية من أغنى اللغات في العالم من حيث المفردات والتراكيب البلاغية. يتحدث بها ملايين الأشخاص في منطقة الشرق الأوسط وشمال أفريقيا، بالإضافة إلى العديد من الجاليات حول العالم. تتميز العربية الفصحى بتاريخ طويل وأهمية ثقافية ودينية كبيرة."""
    summary_instructions = "Summarize this Arabic text in a few sentences, (a) first in Arabic, (b) then in English."
    summary = process_arabic_text(arabic_text_to_summarize, summary_instructions)
    print(f"\nOriginal Arabic Text:\n{arabic_text_to_summarize}")
    print(f"\nSummary:\n{summary}")