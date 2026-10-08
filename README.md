# resume-shortlister

Python application that compares resumes with a job description, scores candidates against job requirements, and reports strengths and gaps with supporting evidence.

## Requirements

- Python 3.9 or later
- A Hugging Face access token with permission to use Inference Providers, or another OpenAI-compatible chat endpoint

## Setup

```sh
cd Codebase
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your Hugging Face token, then provide a folder of resumes and a job description file. Supported formats are PDF, DOCX, TXT, and Markdown.

```sh
export HF_TOKEN="your-token"
python main.py --cv-dir ./cvs --jd ./job_description.txt --interactive
```

Results are written to `./output`. Resume and job-description text is sent to the configured model endpoint for analysis; do not use documents you are not permitted to share with that provider.

## Repository layout

- `Codebase/` contains the application source and run notes.
- `Report/Report.pdf` contains the project report.