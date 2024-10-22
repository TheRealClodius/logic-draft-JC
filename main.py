import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from job_data import JOBS_DATA  # Import the job data
from user_data import USER_PROFILE
import json
from flask import Flask, request, jsonify, send_from_directory, render_template
from pdf_processor import process_pdf
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Configure the Google GenerativeAI library
genai.configure(api_key=api_key)

# Initialize the Google Gemini Flash LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Step 1: Fetch job listings from the API (Now using imported data)
def fetch_job_listings():
    return JOBS_DATA

# Step 2: Process uploaded document (user's CV or profile)
def process_uploaded_document():
    return USER_PROFILE['current_profile']

# Step 3: Generate an initial analysis with the LLM
def generate_user_overview(user_context):
    template = """
    Given the following user context, provide a short introduction about the user,
    along with an analysis of their strengths and weaknesses:

    {user_context}

    User Overview:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    
    return chain.invoke({"user_context": user_context})

# Step 4: Perform first job analysis using jobs and user context
def first_job_analysis(jobs, user_context):
    template = """
    Analyze the following job in the context of the user's background:

    User Context:
    {user_context}

    Job Details:
    Title: {job_title}
    Company: {job_company}
    Location: {job_location}
    Description: {job_description}
    Key Responsibilities: {key_responsibilities}
    Skills Required: {skills_required}

    Please provide your analysis in the following format, referring to the user as "you" or "your":
    Reasons for the match (provide 3 short, concise bullet points):
    • [First reason]
    • [Second reason]
     [Third reason]

    Suggestions for improvement:
    1. [First suggestion]
    2. [Second suggestion]
    3. [Third suggestion]

    Analysis:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = RunnablePassthrough.assign(analysis=prompt | llm)
    
    analyses = []
    for job in jobs:
        result = chain.invoke({
            "user_context": user_context,
            "job_title": job["title"],
            "job_company": job["company"],
            "job_location": job["location"],
            "job_description": job["job_description"],
            "key_responsibilities": ", ".join(job["key_responsibilities"]),
            "skills_required": ", ".join(job["skills_required"])
        })
        analyses.append({"job": job, "analysis": result["analysis"]})
    
    return analyses

# Step 5: Rank jobs based on the analysis
def rank_jobs(first_analysis):
    template = """
    Based on the following job analyses, rank the jobs in order of suitability for the user.
    Provide a brief explanation for each ranking, referring to the user as "you" or "your".

    Job Analyses:
    {first_analysis}

    Ranked Jobs (from most suitable to least suitable):
    1. [Job Title] at [Company Name]
    [Explanation for this ranking using "you" or "your"]

    2. [Job Title] at [Company Name]
    [Explanation for this ranking using "you" or "your"]

    ... (continue for all jobs)

    Please ensure each job ranking is on a new line, followed by its explanation on the next line.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    
    first_analysis_str = "\n\n".join([f"Job: {a['job']['title']} at {a['job']['company']}\nAnalysis: {a['analysis']}" for a in first_analysis])
    result = chain.invoke({"first_analysis": first_analysis_str})
    
    return result.content

# Main workflow
def job_matching_workflow(profile_type="senior_product_designer"):
    # Fetch job listings
    jobs = fetch_job_listings()

    # Process uploaded document (CV/Profile)
    user_context = process_uploaded_document()

    # Generate user overview based on user context
    user_overview = generate_user_overview(user_context)
    
    # Perform the first job analysis
    first_analysis = first_job_analysis(jobs["jobs"], user_context)
    
    # Rank the jobs based on the first analysis
    ranking_order = rank_jobs(first_analysis)
    
    # Parse ranking order to separate explanations
    ranked_jobs = []
    job_explanations = {}
    lines = ranking_order.split('\n')
    current_job = None
    current_explanation = []

    for line in lines:
        line = line.strip()
        if line.startswith(tuple(str(i)+'.' for i in range(1, 11))):
            if current_job:
                job_explanations[current_job] = ' '.join(current_explanation)
            current_job = line
            ranked_jobs.append(current_job)
            current_explanation = []
        elif line and current_job:
            current_explanation.append(line)

    if current_job:
        job_explanations[current_job] = ' '.join(current_explanation)

    # Prepare results for JSON output
    results = {
        "userOverview": user_overview.content if hasattr(user_overview, 'content') else str(user_overview),
        "jobAnalysis": [
            {
                "job": {k: v for k, v in job['job'].items() if k not in ['key_responsibilities', 'skills_required']},
                "analysis": job['analysis'].content if hasattr(job['analysis'], 'content') else str(job['analysis']),
                "ranking": next((i+1 for i, title in enumerate(ranked_jobs) if job['job']['title'] in title), None),
                "explanation": next((exp for title, exp in job_explanations.items() if job['job']['title'] in title), "No explanation provided.")
            } for job in first_analysis
        ],
        "jobRanking": ranked_jobs
    }
    
    # Write results to a JSON file
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Results have been written to results.json")

# Example Usage
if __name__ == "__main__":
    job_matching_workflow("senior_product_designer")

app = Flask(__name__, static_folder='.', static_url_path='')

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/upload', methods=['POST'])
def upload_file():
    app.logger.info("Upload request received")
    if 'file' not in request.files:
        app.logger.error("No file part in the request")
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        app.logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        try:
            file.save(file_path)
            app.logger.info(f"File saved: {file_path}")
        except Exception as e:
            app.logger.error(f"Failed to save file: {str(e)}")
            return jsonify({'error': f'Failed to save file: {str(e)}'}), 500
        
        try:
            text = process_pdf(file_path)
            os.remove(file_path)  # Remove the file after processing
            app.logger.info("PDF processed successfully")
            
            # Update user profile with extracted text and re-run job matching
            update_user_profile(text)
            app.logger.info("User profile updated")
            
            # Read the updated results
            with open('results.json', 'r') as f:
                updated_results = json.load(f)
            
            return jsonify({'message': 'File processed successfully and profile updated', 'results': updated_results}), 200
        except Exception as e:
            app.logger.error(f"Error processing file: {str(e)}")
            if os.path.exists(file_path):
                os.remove(file_path)  # Remove the file if processing fails
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    else:
        app.logger.error(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type'}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}

def update_user_profile(text):
    # Clean the extracted text
    cleaned_text = clean_extracted_text(text)
    
    # Update the user profile with the cleaned text
    USER_PROFILE['current_profile'] = cleaned_text

    # Write the updated profile to a file
    with open('user_data.py', 'w') as f:
        f.write(f"USER_PROFILE = {repr(USER_PROFILE)}")

    # Re-run the job matching workflow
    job_matching_workflow()

def clean_extracted_text(text):
    # Remove any non-printable characters
    cleaned_text = ''.join(char for char in text if char.isprintable())
    
    # Remove excessive whitespace
    cleaned_text = ' '.join(cleaned_text.split())
    
    # Remove any potential markdown formatting
    cleaned_text = cleaned_text.replace('*', '').replace('#', '').replace('_', '')
    
    # Limit the text to a reasonable length (e.g., 2000 characters)
    cleaned_text = cleaned_text[:2000]
    
    return cleaned_text

if not os.path.exists('uploads'):
    os.makedirs('uploads')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
