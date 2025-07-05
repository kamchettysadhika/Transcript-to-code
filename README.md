# Zoom Recording → Code Implementation Analyzer

Transform your engineering meetings into actionable code insights! This tool automatically analyzes Zoom recordings, extracts discussion topics using AI, and matches them to relevant code implementations in your repository—generating beautiful, actionable reports that bridge the gap between meetings and code.

## Features

- **Automatic Zoom Meeting Transcription**: Extracts audio and transcribes with timestamps using OpenAI Whisper
- **AI Topic Extraction**: Uses GPT to summarize and identify actionable topics from transcripts
- **Multi-Language Codebase Analysis**: Scans Python, JavaScript, TypeScript, Java, C/C++, Go, Rust, PHP, and more
- **Semantic Code Matching**: TF-IDF and GPT-powered similarity scoring with human-readable explanations
- **Beautiful Reports**: Generates markdown reports with summary stats, matching tables, and recommendations
- **Command Line Workflow**: Simple terminal prompts—no GUI setup required
- **Rich Console Output**: Pretty progress bars and colorful status messages

## Requirements

- **Python 3.8+**
- **ffmpeg and ffprobe** (installed & available in your system PATH)
- **OpenAI API Key** ([Get one here](https://platform.openai.com/))

### Python Dependencies

```bash
pip install numpy scikit-learn openai python-dotenv chardet rich
```

## Quick Start

### 1. Clone and Install

```bash
git clone <repository-url>
cd zoom-code-analyzer
pip install -r requirements.txt
```

### 2. Install ffmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/)

### 3. Set Your OpenAI API Key

Create a `.env` file in the root directory:

```ini
OPENAI_API_KEY=sk-your-api-key-here
```

### 4. Run the Analyzer

```bash
python your_script.py
```

You'll be prompted for:
- Path to your Zoom recording (`.mp4`, `.m4a`, `.mp3`, etc.)
- Path to your codebase directory

### 5. Get Your Report

The script will generate a timestamped report like `zoom_code_analysis_YYYYMMDD_HHMMSS.md` with all your insights!

## How It Works

1. **Audio Extraction**: Uses ffmpeg to extract audio from video files
2. **Transcription**: Leverages OpenAI Whisper API for accurate, timestamped transcripts
3. **Topic Extraction**: Sends transcript segments to GPT for concise summaries and requirement extraction
4. **Code Analysis**: Recursively scans your codebase for functions/methods in all supported languages
5. **Semantic Matching**: Uses TF-IDF vectors + GPT for intelligent topic-to-code matching
6. **Report Generation**: Creates beautiful, actionable markdown reports with detailed insights

## Example Report Output

```markdown
# Zoom Recording → Code Implementation Analysis
Generated on July 5, 2025 at 2:10 PM

## Summary Statistics
- **Topics Analyzed**: 7
- **Code Matches Found**: 15
- **Average Match Score**: 0.64
- **Source**: Zoom Recording Transcript

## Key Insights
- Found 15 code implementations matching meeting discussions
- 4 high-confidence matches (>0.7)
- 6 medium-confidence matches (>0.5)

## Topic-to-Code Matches
| Topic | Code Function | Match Score | Explanation |
|-------|---------------|-------------|-------------|
| User Authentication | `authenticate_user()` | 0.85 | Direct match for login functionality discussed |
| Database Optimization | `optimize_queries()` | 0.72 | Relates to performance improvements mentioned |
```

## Supported Languages

- Python (`.py`)
- JavaScript (`.js`)
- TypeScript (`.ts`)
- Java (`.java`)
- C/C++ (`.c`, `.cpp`, `.h`)
- Go (`.go`)
- Rust (`.rs`)
- PHP (`.php`)
- And more!

## Troubleshooting

**ffmpeg not found:**
- Ensure ffmpeg is installed and available in your system PATH
- Test with `ffmpeg -version` in your terminal

**API errors:**
- Double-check your OpenAI API key in the `.env` file
- Verify your OpenAI account has sufficient credits

**No code matches found:**
- Ensure your codebase directory path is correct
- Check that your code files are in supported languages
- Consider adjusting the match threshold if needed

## Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Credits

Built with love by Sadhika Kamchetty and contributors.

**Powered by:**
- [OpenAI](https://openai.com/) - GPT and Whisper APIs
- [scikit-learn](https://scikit-learn.org/) - Machine learning utilities
- [rich](https://github.com/Textualize/rich) - Beautiful terminal output

---

**Ready to bridge your meetings and code?** Star this repo and start analyzing your Zoom recordings today!
