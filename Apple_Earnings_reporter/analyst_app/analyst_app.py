import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import types
from PyPDF2 import PdfReader
from token_tracker import tracker
from resource_monitor import monitor_resources, profile_memory

load_dotenv()

# Update PDF directory to current folder
PDF_DIR = os.path.dirname(__file__)

# Ensure logs directory exists
os.makedirs(os.path.join(PDF_DIR, "logs"), exist_ok=True)
PDF_FILES = [
    os.path.join(PDF_DIR, f)
    for f in os.listdir(PDF_DIR)
    if f.endswith('.pdf')
]

def extract_text_from_pdfs(pdf_files):
    """Extract text from PDF files and return a dictionary mapping filenames to their content."""
    pdf_contents = {}
    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            content = ""
            for page in reader.pages:
                content += page.extract_text() + "\n"
            pdf_contents[os.path.basename(pdf_file)] = content
        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")
    return pdf_contents

def format_number(number_str):
    """Convert number strings to more readable formats (billions or millions)."""
    try:
        number = float(number_str.replace(',', ''))
        if abs(number) >= 1_000_000_000:
            return f"{number/1_000_000_000:.2f}B"
        elif abs(number) >= 1_000_000:
            return f"{number/1_000_000:.2f}M"
        else:
            return f"{number:,.2f}"
    except:
        return number_str

def main():
    print("Welcome to the Apple Earnings Report Analyst!")
    print("Loading earnings reports...")
    
    pdf_contents = extract_text_from_pdfs(PDF_FILES)
    if not pdf_contents:
        print("No PDF files found. Exiting.")
        return

    # Initialize Gemini
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")

    # First, analyze available quarters
    quarters_prompt = f"""
    Analyze the following earnings reports and list which quarters and years they cover.
    Format the response as a simple list of available quarters.
    Reports content: {pdf_contents}
    """
    
    quarters_response = model.generate_content(quarters_prompt)
    available_quarters = quarters_response.text if quarters_response.text else "Unknown quarters"
    
    system_prompt = f"""
    You are an expert financial analyst AI specializing in Apple's earnings reports.
    
    Available Information:
    {available_quarters}
    
    Guidelines:
    1. Only answer questions based on the available earnings reports.
    2. Present financial numbers in billions (B) for large figures and millions (M) for smaller ones.
    3. If asked about quarters/years not in our data or about other companies, respond:
       "I apologize, but I only have information about Apple's earnings for {available_quarters}. 
        I cannot provide information about other periods or companies."
    4. Always mention which quarter/year you're referring to in your answers.
    5. Be precise and clear with financial data.
    """

    print("\nAvailable Quarters for Analysis:")
    print(available_quarters)
    print("\nYou can now ask questions about Apple's earnings for these periods.")
    
    while True:
        user_input = input("\nAsk your question (or type 'exit' to quit): ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            break

        # Combine system prompt, PDF content, and user question
        prompt = f"{system_prompt}\n\nEarnings Reports:\n{pdf_contents}\n\nUser Question: {user_input}"
        
        print("Analyst:", end=" ")
        response = model.generate_content(
            contents=[prompt],
            generation_config=types.GenerationConfig(
                temperature=0.7  # Slightly lower temperature for more focused responses
            )
        )
        
        if response.text:
            print(response.text)
        else:
            print("Sorry, I couldn't generate a response.")

if __name__ == "__main__":
    main()
