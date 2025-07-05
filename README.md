Zoom Recording → Code Implementation Analyzer
This project takes a Zoom recording (video/audio of your meeting), automatically transcribes the discussion, extracts main engineering topics, and then matches those topics to actual code implementations in your codebase (Python, JS, Java, C/C++, and more).
It generates a beautiful, actionable report linking your meetings to real code, with scoring and insights.

Features
Automatic Zoom Meeting Transcription: Extracts audio and transcribes it with timestamps using OpenAI Whisper.

AI Topic Extraction: Uses GPT to summarize and identify actionable topics or requirements from the transcript.

Multi-Language Codebase Analysis: Scans your codebase for functions/methods in Python, JavaScript, TypeScript, Java, C, C++, Go, Rust, PHP, and more.

Semantic Code Matching: Uses TF-IDF and GPT for similarity scoring and human-like reasoning for why code matches each topic.

Beautiful Reports: Generates a markdown report with summary stats, matching tables, and actionable recommendations.

Command Line Workflow: Simple terminal prompts; no GUI setup required.

Rich Console Output: Uses the rich library for pretty progress bars and colorful status messages.

 Requirements
Python 3.8+

ffmpeg and ffprobe (installed & available in your system PATH)

Python packages (install with pip):

nginx
Copy
Edit
pip install numpy scikit-learn openai python-dotenv chardet rich
OpenAI API Key
Get one from https://platform.openai.com/ and set it in your .env file or environment.

⚡️ Quickstart
Clone the repository and install dependencies:

bash
Copy
Edit
git clone <your-repo-url>
cd <your-repo-dir>
pip install -r requirements.txt
Install ffmpeg and ffprobe (if not already installed):

On Mac:
brew install ffmpeg

On Ubuntu:
sudo apt-get install ffmpeg

On Windows:
Download from ffmpeg.org

Set your OpenAI API Key:
Add to .env file in the root directory:

ini
Copy
Edit
OPENAI_API_KEY=sk-...
Run the script:

bash
Copy
Edit
python your_script.py
You’ll be prompted for:

Path to your Zoom recording (.mp4, .m4a, etc.)

Path to your codebase directory

Wait for the analysis to complete.
The script will show colorful progress and print out the report location (e.g., zoom_code_analysis_YYYYMMDD_HHMMSS.md).

 What Happens Under the Hood?
Audio Extraction:
If you give it a video, it extracts audio using ffmpeg.

Transcription:
Splits audio if necessary and transcribes each chunk using OpenAI Whisper, with timestamps for each segment.

Topic Extraction:
Sends transcript chunks to GPT to summarize discussion topics and software requirements, with context and related programming concepts.

Codebase Analysis:








You said:
generate a readme


ChatGPT said:
Absolutely! Here’s a clean, professional README.md for your project, formatted for GitHub (or any code repo):

 Zoom Recording → Code Implementation Analyzer
This project automatically analyzes your engineering meetings and links discussion topics directly to relevant code in your repository. It takes a Zoom recording (audio or video), transcribes it, extracts actionable topics using GPT, scans your codebase for functions/methods, and semantically matches each meeting topic to code—generating a clear, beautiful markdown report.

 Features
Zoom Meeting Transcription: Extracts and transcribes audio with timestamps from Zoom recordings.

AI-Powered Topic Extraction: Uses OpenAI GPT to summarize and identify actionable engineering requirements.

Multi-language Codebase Support: Scans for functions/methods in Python, JS, TypeScript, Java, C/C++, Go, Rust, PHP, and more.

Semantic Code Matching: TF-IDF and GPT-powered similarity scoring to map discussion topics to code, with human-like explanations.

Rich, Actionable Reports: Generates a Markdown report summarizing matches, scores, and next steps.

Intuitive CLI with Beautiful Console Output: Friendly prompts and colorful progress via the rich library.

 Requirements
Python 3.8+

ffmpeg and ffprobe (for audio extraction)

OpenAI API Key (Get one here)

Python packages:

sh
Copy
Edit
pip install numpy scikit-learn openai python-dotenv chardet rich
Setup
Clone this repo:

sh
Copy
Edit
git clone <your-repo-url>
cd <your-repo-dir>
Install requirements:
See above.
On Mac: brew install ffmpeg
On Ubuntu: sudo apt-get install ffmpeg

Add your OpenAI API key:
Create a .env file in your project root:

ini
Copy
Edit
OPENAI_API_KEY=sk-...
 Usage
Run the script:

sh
Copy
Edit
python your_script.py
You’ll be prompted for:

The path to your Zoom recording (.mp4, .m4a, .mp3, etc.)

The path to your codebase directory

Wait for analysis to finish.
The script will:

Transcribe the recording

Extract topics

Analyze your codebase

Match topics to code

Generate a Markdown report (e.g. zoom_code_analysis_YYYYMMDD_HHMMSS.md)

🧩 How it Works
Audio Extraction: Uses ffmpeg to extract audio if you provide video.

Transcription: Uses OpenAI Whisper API for accurate, timestamped transcripts.

Topic Extraction: Sends transcript segments to GPT for concise summaries & requirement extraction.

Code Analysis: Recursively scans your codebase for all supported languages & extracts all functions/methods.

Semantic Matching: Uses TF-IDF vectors + GPT for smart, human-readable topic-to-code matches.

Reporting: Renders a beautiful, actionable report with summary stats and detailed match breakdowns.

📝 Example Report
markdown
Copy
Edit
# 🎯 Zoom Recording → Code Implementation Analysis
Generated on July 5, 2025 at 2:10 PM

## 📊 Summary Statistics
- Topics Analyzed: 7
- Code Matches Found: 15
- Average Match Score: 0.64
- Source: Zoom Recording Transcript

...

## 🚀 Key Insights
- Found 15 code implementations matching meeting discussions
- 4 high-confidence matches (>0.7)
- 6 medium-confidence matches (>0.5)
⚠️ Troubleshooting
ffmpeg not found:
Make sure ffmpeg is installed and available in your system PATH.

API errors:
Double-check your OpenAI API key and account limits.

No code matches found:

Ensure your codebase directory is correct.

Try lowering the match threshold if needed.

🙏 Credits
Built by Sadhika Kamchetty and contributors.

Uses OpenAI, scikit-learn, and rich.

📄 License
MIT License

