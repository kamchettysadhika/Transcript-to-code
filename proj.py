import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import asyncio
from dataclasses import dataclass
import json
import subprocess
from pathlib import Path
import glob
import os 
import chardet
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
from openai import AsyncOpenAI
import time
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
import requests
import base64
import urllib.parse
import asyncio
import os
from dotenv import load_dotenv
import tempfile
import shutil

# Initialize Rich console for beautiful output
console = Console()

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

@dataclass
class TranscriptTopic:
    keyword: str
    context: str
    description: str
    related_concepts: List[str]
    timestamp: Optional[str] = None  # Add timestamp for video segments

@dataclass
class CodeBlock:
    file_path: str
    function_name: str
    content: str
    start_line: int
    end_line: int

@dataclass 
class CodeMatch:
    code_block: 'CodeBlock'
    topic: TranscriptTopic
    similarity_score: float
    reasoning: str

class ZoomRecordingProcessor:
    """Process Zoom recordings and extract transcripts"""
    
    def __init__(self, openai_api_key: str):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.temp_dir = None
    
    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                 capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio from video file using ffmpeg"""
        if not self._check_ffmpeg():
            raise RuntimeError("ffmpeg not found. Please install ffmpeg to process video files.")
        
        # Create temporary directory if it doesn't exist
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()
        
        video_file = Path(video_path)
        audio_file = Path(self.temp_dir) / f"{video_file.stem}_audio.mp3"
        
        console.print(f"🎵 [blue]Extracting audio from video: {video_file.name}[/blue]")
        
        try:
            # Extract audio using ffmpeg
            cmd = [
                'ffmpeg', '-i', str(video_file),
                '-vn',  # No video
                '-acodec', 'mp3',  # Audio codec
                '-ab', '192k',  # Audio bitrate
                '-ar', '44100',  # Audio sample rate
                '-y',  # Overwrite output file
                str(audio_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")
            
            console.print(f"✅ [green]Audio extracted successfully: {audio_file.name}[/green]")
            return str(audio_file)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Audio extraction timed out (5 minutes)")
        except Exception as e:
            raise RuntimeError(f"Error extracting audio: {e}")
    
    def _split_audio_for_whisper(self, audio_path: str, chunk_duration: int = 600) -> List[str]:
        """Split audio file into chunks for Whisper API (max 25MB per chunk)"""
        audio_file = Path(audio_path)
        chunks = []
        
        console.print(f"✂️ [blue]Splitting audio into {chunk_duration}-second chunks for processing[/blue]")
        
        try:
            # Get audio duration first
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(audio_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                duration = float(info['format']['duration'])
                total_chunks = int(duration // chunk_duration) + 1
                
                console.print(f"📊 [cyan]Audio duration: {duration:.1f}s, creating {total_chunks} chunks[/cyan]")
            else:
                console.print("⚠️ [yellow]Could not determine audio duration, creating chunks anyway[/yellow]")
                total_chunks = 10  # Fallback estimate
            
            # Split audio into chunks
            chunk_num = 0
            start_time = 0
            
            while True:
                chunk_file = Path(self.temp_dir) / f"{audio_file.stem}_chunk_{chunk_num:03d}.mp3"
                
                cmd = [
                    'ffmpeg', '-i', str(audio_file),
                    '-ss', str(start_time),
                    '-t', str(chunk_duration),
                    '-acodec', 'copy',
                    '-y',
                    str(chunk_file)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode != 0:
                    break
                
                # Check if chunk file exists and has content
                if chunk_file.exists() and chunk_file.stat().st_size > 1000:  # At least 1KB
                    chunks.append(str(chunk_file))
                    chunk_num += 1
                    start_time += chunk_duration
                else:
                    # No more content to extract
                    if chunk_file.exists():
                        chunk_file.unlink()
                    break
            
            console.print(f"✅ [green]Created {len(chunks)} audio chunks[/green]")
            return chunks
            
        except Exception as e:
            console.print(f"❌ [red]Error splitting audio: {e}[/red]")
            return [audio_path]  # Return original file if splitting fails
    
    async def _transcribe_audio_chunk(self, audio_chunk_path: str, chunk_index: int) -> Tuple[str, int]:
        """Transcribe a single audio chunk using OpenAI Whisper"""
        try:
            with open(audio_chunk_path, 'rb') as audio_file:
                # Check file size (Whisper API has 25MB limit)
                file_size = os.path.getsize(audio_chunk_path)
                if file_size > 25 * 1024 * 1024:  # 25MB
                    console.print(f"⚠️ [yellow]Chunk {chunk_index} is too large ({file_size/1024/1024:.1f}MB), skipping[/yellow]")
                    return "", chunk_index
                
                console.print(f"🎙️ [blue]Transcribing chunk {chunk_index} ({file_size/1024/1024:.1f}MB)[/blue]")
                
                transcript = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",  # Get timestamps
                    language="en"  # Specify language for better accuracy
                )
                
                # Extract text with timestamps if available
                if hasattr(transcript, 'segments') and transcript.segments:
                    # Create timestamped transcript
                    timestamped_text = []
                    for segment in transcript.segments:
                        start_time = segment.get('start', 0)
                        text = segment.get('text', '').strip()
                        if text:
                            timestamp = f"[{start_time//60:02d}:{start_time%60:02d}]"
                            timestamped_text.append(f"{timestamp} {text}")
                    
                    result_text = '\n'.join(timestamped_text)
                else:
                    result_text = transcript.text
                
                console.print(f"✅ [green]Chunk {chunk_index} transcribed successfully[/green]")
                return result_text, chunk_index
                
        except Exception as e:
            console.print(f"❌ [red]Error transcribing chunk {chunk_index}: {e}[/red]")
            return "", chunk_index
    
    async def transcribe_zoom_recording(self, recording_path: str) -> str:
        """Main method to transcribe Zoom recording"""
        recording_file = Path(recording_path)
        
        if not recording_file.exists():
            raise FileNotFoundError(f"Recording file not found: {recording_path}")
        
        console.print(f"🎬 [bold blue]Processing Zoom recording: {recording_file.name}[/bold blue]")
        
        # Check if file is audio or video
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        
        file_ext = recording_file.suffix.lower()
        
        if file_ext in audio_extensions:
            console.print("🎵 [green]Audio file detected, processing directly[/green]")
            audio_path = str(recording_file)
        elif file_ext in video_extensions:
            console.print("🎬 [green]Video file detected, extracting audio[/green]")
            audio_path = self._extract_audio_from_video(str(recording_file))
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Split audio into chunks
        audio_chunks = self._split_audio_for_whisper(audio_path)
        
        if not audio_chunks:
            raise RuntimeError("No audio chunks created for transcription")
        
        # Transcribe all chunks concurrently
        console.print(f"🔄 [blue]Transcribing {len(audio_chunks)} audio chunks...[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("🎙️ Transcribing audio chunks...", total=len(audio_chunks))
            
            # Process chunks with limited concurrency to avoid rate limits
            chunk_results = []
            batch_size = 3  # Process 3 chunks at a time
            
            for i in range(0, len(audio_chunks), batch_size):
                batch = audio_chunks[i:i+batch_size]
                batch_tasks = [
                    self._transcribe_audio_chunk(chunk, i + j) 
                    for j, chunk in enumerate(batch)
                ]
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        console.print(f"❌ [red]Transcription error: {result}[/red]")
                        chunk_results.append(("", -1))
                    else:
                        chunk_results.append(result)
                
                progress.advance(task, len(batch))
                
                # Rate limiting between batches
                if i + batch_size < len(audio_chunks):
                    await asyncio.sleep(2)
        
        # Combine all transcription results
        chunk_results.sort(key=lambda x: x[1])  # Sort by chunk index
        full_transcript = '\n\n'.join([text for text, _ in chunk_results if text.strip()])
        
        if not full_transcript.strip():
            raise RuntimeError("No transcription text was generated")
        
        console.print(f"✅ [green bold]Transcription complete! Generated {len(full_transcript)} characters[/green bold]")
        
        # Save transcript to file
        transcript_file = recording_file.parent / f"{recording_file.stem}_transcript.txt"
        try:
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(full_transcript)
            console.print(f"💾 [green]Transcript saved to: {transcript_file}[/green]")
        except Exception as e:
            console.print(f"⚠️ [yellow]Could not save transcript file: {e}[/yellow]")
        
        return full_transcript
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                console.print("🧹 [green]Temporary files cleaned up[/green]")
            except Exception as e:
                console.print(f"⚠️ [yellow]Could not clean up temporary files: {e}[/yellow]")

class TranscriptCodeMatcher:
    def __init__(self, openai_api_key: str):
        """Initialize with OpenAI API key"""
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        console.print("✅ [green bold]OpenAI client initialized successfully[/green bold]")

    async def extract_topics_from_transcript(self, transcript: str) -> List[TranscriptTopic]:
        """Extract topics from full transcript blocks using ChatGPT."""
        # Handle timestamped transcript format
        entries = []
        current_entry = []
        
        for line in transcript.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with timestamp [MM:SS]
            if re.match(r'^\[\d{2}:\d{2}\]', line):
                # Save previous entry if exists
                if current_entry:
                    entries.append('\n'.join(current_entry))
                    current_entry = []
                current_entry.append(line)
            else:
                # Continue current entry
                if current_entry:
                    current_entry.append(line)
                else:
                    # Handle non-timestamped content
                    entries.append(line)
        
        # Add final entry
        if current_entry:
            entries.append('\n'.join(current_entry))
        
        # If no timestamped entries found, split by paragraphs
        if not entries:
            entries = re.split(r"\n\s*\n", transcript)
        
        topics = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("🔍 Extracting topics from transcript...", total=len(entries))
            
            # Process in batches to avoid API rate limits
            batch_size = 5
            for i in range(0, len(entries), batch_size):
                batch = entries[i:i+batch_size]
                batch_topics = await self._process_batch_with_chatgpt(batch)
                topics.extend(batch_topics)
                progress.advance(task, min(batch_size, len(entries) - i))
                
                # Rate limiting - wait between batches
                if i + batch_size < len(entries):
                    await asyncio.sleep(0.5)

        return topics

    async def _process_batch_with_chatgpt(self, batch: List[str]) -> List[TranscriptTopic]:
        async def safe_analyze(block: str) -> TranscriptTopic:
            # Extract timestamp if present
            timestamp = None
            keyword = block[:40] if len(block) > 40 else block
            
            timestamp_match = re.search(r'^\[(\d{2}:\d{2})\]', block)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                # Remove timestamp from keyword
                keyword = re.sub(r'^\[\d{2}:\d{2}\]\s*', '', keyword)[:40]
            
            try:
                description, concepts = await asyncio.wait_for(
                    self._analyze_topic_with_chatgpt(keyword, block), timeout=15
                )
                return TranscriptTopic(
                    keyword=keyword.strip(),
                    context=block.strip(),
                    description=description,
                    related_concepts=concepts,
                    timestamp=timestamp
                )
            except Exception as e:
                return TranscriptTopic(
                    keyword=keyword.strip(),
                    context=block.strip(),
                    description=f"Programming task related to {keyword}",
                    related_concepts=self._extract_keywords_simple(block),
                    timestamp=timestamp
                )

        # Filter valid blocks
        valid_blocks = [b for b in batch if len(b.strip()) >= 20]
        
        # Launch concurrent analysis
        results = await asyncio.gather(
            *(safe_analyze(block) for block in valid_blocks),
            return_exceptions=False
        )
        
        return results

    def _extract_keywords_simple(self, text: str) -> List[str]:
        """Simple keyword extraction fallback"""
        programming_terms = re.findall(r'\b(?:function|method|class|api|data|process|system|code|implement|create|build|develop|update|fix|debug|test)\b', text.lower())
        identifiers = re.findall(r'\b[a-z][a-zA-Z0-9_]*[A-Z][a-zA-Z0-9_]*\b|\b[a-z][a-z0-9_]*_[a-z0-9_]+\b', text)
        
        return list(set(programming_terms + identifiers))[:8]
    
    async def _analyze_topic_with_chatgpt(self, keyword: str, context: str) -> Tuple[str, List[str]]:
        """Use ChatGPT to understand what programming concepts relate to a transcript topic"""
        
        prompt = f"""
        Analyze this meeting transcript excerpt and identify programming concepts:
        
        Keyword: {keyword}
        Context: {context}
        
        Please provide:
        1. A brief description of what code functionality this might require
        2. Up to 5 related programming concepts (e.g., "API development", "data processing", "user interface")
        
        Respond in JSON format:
        {{
            "description": "brief description",
            "concepts": ["concept1", "concept2", "concept3"]
        }}
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a technical analyst helping to map meeting discussions to programming requirements. Keep responses concise and focused on actionable development tasks."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content.strip()
            
            try:
                result_json = json.loads(result_text)
                description = result_json.get("description", f"Code implementation for {keyword}")
                concepts = result_json.get("concepts", [])[:5]
                return description, concepts
            except json.JSONDecodeError:
                return self._parse_chatgpt_fallback(result_text, keyword)
            
        except Exception as e:
            logger.warning(f"Error analyzing topic with ChatGPT: {e}")
            return self._generate_description_heuristic(keyword, context, [])
    
    def _parse_chatgpt_fallback(self, text: str, keyword: str) -> Tuple[str, List[str]]:
        """Parse ChatGPT response when JSON parsing fails"""
        lines = text.split('\n')
        description = f"Code implementation for {keyword}"
        concepts = []
        
        for line in lines:
            if 'description' in line.lower():
                desc_match = re.search(r'[:"](.*?)[",]', line)
                if desc_match:
                    description = desc_match.group(1).strip()
            elif 'concept' in line.lower() and any(char in line for char in ['"', "'"]):
                concept_matches = re.findall(r'["\']([^"\']+)["\']', line)
                concepts.extend(concept_matches)
        
        return description, concepts[:5]
    
    def _generate_description_heuristic(self, keyword: str, context: str, concepts: List[str]) -> Tuple[str, List[str]]:
        """Generate description using heuristics instead of LLM generation"""
        
        if any(word in context.lower() for word in ['api', 'endpoint', 'request']):
            return f"API functionality related to {keyword}", ["API development", "HTTP requests", "endpoint design"]
        elif any(word in context.lower() for word in ['data', 'database', 'store']):
            return f"Data processing and storage for {keyword}", ["data processing", "database operations", "data validation"]
        elif any(word in context.lower() for word in ['ui', 'interface', 'display', 'view']):
            return f"User interface implementation for {keyword}", ["user interface", "frontend development", "UI components"]
        elif any(word in context.lower() for word in ['test', 'testing', 'verify']):
            return f"Testing and validation for {keyword}", ["testing", "validation", "quality assurance"]
        elif any(word in context.lower() for word in ['config', 'setup', 'install']):
            return f"Configuration and setup for {keyword}", ["configuration", "setup", "deployment"]
        else:
            return f"General functionality implementation for {keyword}", ["general development", "code implementation"]
    
    async def find_matching_code_blocks(self, topics: List[TranscriptTopic], 
                                      code_blocks: List['CodeBlock']) -> List[CodeMatch]:
        """Find code blocks that implement functionality mentioned in transcript topics using TF-IDF similarity"""
        
        # FIXED: Handle empty code_blocks case
        if not code_blocks:
            console.print("⚠️ [yellow]No code blocks found. Skipping similarity analysis.[/yellow]")
            return []
        
        if not topics:
            console.print("⚠️ [yellow]No topics found. Skipping similarity analysis.[/yellow]")
            return []
        
        matches = []
        
        topic_texts = [f"{topic.keyword} {topic.description} {' '.join(topic.related_concepts)}" 
                      for topic in topics]
        code_texts = [f"{cb.function_name} {cb.content[:500]}" for cb in code_blocks]
        
        all_texts = topic_texts + code_texts
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("🔍 Computing TF-IDF vectors for semantic matching...", total=1)
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            progress.advance(task, 1)
        
        topic_matrix = tfidf_matrix[:len(topic_texts)]
        code_matrix = tfidf_matrix[len(topic_texts):]
        
        similarities = cosine_similarity(topic_matrix, code_matrix)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("🤖 Finding semantic matches...", total=len(topics))
            
            for i, topic in enumerate(topics):
                topic_similarities = similarities[i]
                
                for j, similarity_score in enumerate(topic_similarities):
                    if similarity_score > 0.3:
                        reasoning = await self._generate_match_reasoning_with_chatgpt(
                            topic, code_blocks[j], similarity_score
                        )
                        
                        matches.append(CodeMatch(
                            code_block=code_blocks[j],
                            topic=topic,
                            similarity_score=float(similarity_score),
                            reasoning=reasoning
                        ))
                
                progress.advance(task, 1)
        
        return sorted(matches, key=lambda x: x.similarity_score, reverse=True)
    
    async def _generate_match_reasoning_with_chatgpt(self, topic: TranscriptTopic, 
                                                   code_block: CodeBlock, score: float) -> str:
        """Generate reasoning for why a topic matches a code block using ChatGPT"""
        
        if score < 0.5:
            return self._generate_match_reasoning_heuristic(topic, code_block, score)
        
        prompt = f"""
        Explain why this meeting topic matches this code function:
        
        Meeting Topic: {topic.keyword}
        Topic Description: {topic.description}
        Related Concepts: {', '.join(topic.related_concepts)}
        {f"Timestamp: {topic.timestamp}" if topic.timestamp else ""}
        
        Function Name: {code_block.function_name}
        Code Preview: {code_block.content[:200]}
        
        Similarity Score: {score:.2f}
        
        Provide a brief explanation (1-2 sentences) of why they match.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are analyzing code-to-requirements matches. Be concise and technical."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.2
            )
            
            reasoning = response.choices[0].message.content.strip()
            return reasoning
            
        except Exception as e:
            logger.warning(f"Error generating reasoning with ChatGPT: {e}")
            return self._generate_match_reasoning_heuristic(topic, code_block, score)
    
    def _generate_match_reasoning_heuristic(self, topic: TranscriptTopic, code_block: CodeBlock, score: float) -> str:
        """Generate reasoning for why a topic matches a code block using heuristics"""
        
        function_name_words = re.findall(r'[A-Z][a-z]*|[a-z]+', code_block.function_name.replace('_', ' '))
        topic_words = topic.keyword.lower().split()
        
        common_words = set(word.lower() for word in function_name_words) & set(topic_words)
        
        if common_words:
            return f"Function name '{code_block.function_name}' contains keywords {list(common_words)} matching topic '{topic.keyword}'. Semantic similarity: {score:.2f}"
        
        code_content_lower = code_block.content.lower()
        matching_concepts = [concept for concept in topic.related_concepts 
                           if any(word in code_content_lower for word in concept.lower().split())]
        
        if matching_concepts:
            return f"Code implements concepts: {matching_concepts}. Semantic similarity: {score:.2f}"
        
        return f"High semantic similarity ({score:.2f}) between topic '{topic.keyword}' and function '{code_block.function_name}'"
    
    def generate_beautiful_report(self, matches: List[CodeMatch]) -> str:
        """Generate a beautiful, appealing report and save to file"""
        
        # Handle empty matches case
        if not matches:
            console.print("\n")
            console.print(Panel.fit(
                "[bold yellow]⚠️ No code matches found[/bold yellow]\n"
                "[dim]This could be due to GitHub rate limits or no semantic matches above threshold.[/dim]",
                border_style="yellow"
            ))
            return "No matches found"
        
        # Group matches by topic
        by_topic = {}
        for match in matches:
            topic_key = match.topic.keyword
            if topic_key not in by_topic:
                by_topic[topic_key] = []
            by_topic[topic_key].append(match)
        
        # Display header
        console.print("\n")
        console.print(Panel.fit(
            "[bold cyan]🎯 Zoom Recording → Code Implementation Analysis[/bold cyan]\n"
            f"[dim]Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}[/dim]",
            border_style="cyan"
        ))
        
        # Summary statistics
        stats_table = Table(show_header=False, box=None, padding=(0, 2))
        stats_table.add_column(style="bold blue")
        stats_table.add_column(style="green")
        
        stats_table.add_row("📊 Topics Analyzed:", str(len(by_topic)))
        stats_table.add_row("💻 Code Matches Found:", str(len(matches)))
        stats_table.add_row("🎯 Average Match Score:", f"{sum(m.similarity_score for m in matches) / len(matches):.2f}" if matches else "N/A")
        stats_table.add_row("🎬 Source:", "Zoom Recording Transcript")
        
        console.print(Panel(stats_table, title="[bold]Summary Statistics[/bold]", border_style="green"))
        
        # Generate file report content
        report_content = []
        report_content.append("# 🎯 Zoom Recording → Code Implementation Analysis")
        report_content.append(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
        report_content.append("\n## 📊 Summary Statistics")
        report_content.append(f"- **Topics Analyzed:** {len(by_topic)}")
        report_content.append(f"- **Code Matches Found:** {len(matches)}")
        report_content.append(f"- **Average Match Score:** {sum(m.similarity_score for m in matches) / len(matches):.2f}" if matches else "N/A")
        report_content.append(f"- **Source:** Zoom Recording Transcript")
        report_content.append("\n---\n")
        
        # Display matches by topic
        for topic_keyword, topic_matches in by_topic.items():
            if not topic_matches:
                continue
                
            topic = topic_matches[0].topic
            
            # Console output
            console.print(f"\n[bold magenta]🎯 {topic_keyword}[/bold magenta]")
            if topic.timestamp:
                console.print(f"[dim]⏰ Timestamp: {topic.timestamp}[/dim]")
             # File report content
            report_content.append(f"## 🎯 {topic_keyword}")
            if topic.timestamp:
                report_content.append(f"**⏰ Timestamp:** {topic.timestamp}")
            report_content.append(f"**📝 Context:** {topic.context[:200]}...")
            report_content.append(f"**🎯 Description:** {topic.description}")
            report_content.append(f"**🔗 Related Concepts:** {', '.join(topic.related_concepts)}")
            report_content.append("\n### 💻 Matching Code Implementations:")
            
            # Create table for code matches
            matches_table = Table(show_header=True, header_style="bold blue")
            matches_table.add_column("🏷️ Function", style="cyan", width=25)
            matches_table.add_column("📁 File", style="magenta", width=30)
            matches_table.add_column("🎯 Score", style="green", width=10)
            matches_table.add_column("💡 Reasoning", style="yellow")
            
            for match in sorted(topic_matches, key=lambda x: x.similarity_score, reverse=True):
                matches_table.add_row(
                    match.code_block.function_name,
                    str(Path(match.code_block.file_path).name),
                    f"{match.similarity_score:.2f}",
                    match.reasoning[:80] + "..." if len(match.reasoning) > 80 else match.reasoning
                )
                
                # Add to file report
                report_content.append(f"#### 📋 {match.code_block.function_name}")
                report_content.append(f"- **📁 File:** `{match.code_block.file_path}`")
                report_content.append(f"- **📍 Lines:** {match.code_block.start_line}-{match.code_block.end_line}")
                report_content.append(f"- **🎯 Match Score:** {match.similarity_score:.2f}")
                report_content.append(f"- **💡 Reasoning:** {match.reasoning}")
                report_content.append("```python")
                report_content.append(match.code_block.content[:300] + "..." if len(match.code_block.content) > 300 else match.code_block.content)
                report_content.append("```")
                report_content.append("")
            
            console.print(matches_table)
            report_content.append("\n---\n")
        
        # Generate actionable insights
        console.print("\n")
        console.print(Panel.fit(
            "[bold green]🚀 Key Insights[/bold green]\n"
            f"[dim]• Found {len(matches)} code implementations matching meeting discussions[/dim]\n"
            f"[dim]• {len([m for m in matches if m.similarity_score > 0.7])} high-confidence matches (>0.7)[/dim]\n"
            f"[dim]• {len([m for m in matches if m.similarity_score > 0.5])} medium-confidence matches (>0.5)[/dim]",
            border_style="green"
        ))
        
        # Add insights to file report
        report_content.append("## 🚀 Key Insights")
        report_content.append(f"- Found **{len(matches)}** code implementations matching meeting discussions")
        report_content.append(f"- **{len([m for m in matches if m.similarity_score > 0.7])}** high-confidence matches (>0.7)")
        report_content.append(f"- **{len([m for m in matches if m.similarity_score > 0.5])}** medium-confidence matches (>0.5)")
        report_content.append("\n## 📝 Recommendations")
        
        # Generate recommendations
        high_matches = [m for m in matches if m.similarity_score > 0.7]
        medium_matches = [m for m in matches if 0.5 < m.similarity_score <= 0.7]
        
        if high_matches:
            report_content.append("### ✅ High Priority Items")
            for match in high_matches[:5]:  # Top 5 high priority
                report_content.append(f"- **{match.topic.keyword}** → `{match.code_block.function_name}` (Score: {match.similarity_score:.2f})")
                report_content.append(f"  - {match.reasoning}")
        
        if medium_matches:
            report_content.append("### ⚠️ Review Required")
            for match in medium_matches[:3]:  # Top 3 medium priority
                report_content.append(f"- **{match.topic.keyword}** → `{match.code_block.function_name}` (Score: {match.similarity_score:.2f})")
                report_content.append(f"  - {match.reasoning}")
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"zoom_code_analysis_{timestamp}.md"
        
        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_content))
            console.print(f"\n💾 [bold green]Report saved to: {report_filename}[/bold green]")
        except Exception as e:
            console.print(f"⚠️ [yellow]Could not save report file: {e}[/yellow]")
        
        return '\n'.join(report_content)


class CodeAnalyzer:
    def __init__(self):
        self.supported_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt'}
        
    def find_code_files(self, directory: str) -> List[str]:
        """Find all supported code files in directory"""
        code_files = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            console.print(f"❌ [red]Directory not found: {directory}[/red]")
            return []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("🔍 Scanning for code files...", total=None)
            
            for ext in self.supported_extensions:
                pattern = f"**/*{ext}"
                files = list(directory_path.glob(pattern))
                code_files.extend(str(f) for f in files if f.is_file())
                progress.advance(task, len(files))
        
        console.print(f"📁 [green]Found {len(code_files)} code files[/green]")
        return code_files
    
    def extract_functions_from_file(self, file_path: str) -> List[CodeBlock]:
        """Extract functions/methods from a code file"""
        try:
            # Detect encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
            
            # Read file content
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()
            
            return self._extract_functions_by_language(file_path, content)
            
        except Exception as e:
            logger.warning(f"Error processing file {file_path}: {e}")
            return []
    
    def _extract_functions_by_language(self, file_path: str, content: str) -> List[CodeBlock]:
        """Extract functions based on file extension"""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.py':
            return self._extract_python_functions(file_path, content)
        elif ext in ['.js', '.ts']:
            return self._extract_javascript_functions(file_path, content)
        elif ext in ['.java', '.cs']:
            return self._extract_java_csharp_functions(file_path, content)
        elif ext in ['.cpp', '.c', '.h']:
            return self._extract_c_cpp_functions(file_path, content)
        elif ext == '.php':
            return self._extract_php_functions(file_path, content)
        elif ext == '.rb':
            return self._extract_ruby_functions(file_path, content)
        elif ext == '.go':
            return self._extract_go_functions(file_path, content)
        elif ext == '.rs':
            return self._extract_rust_functions(file_path, content)
        elif ext == '.swift':
            return self._extract_swift_functions(file_path, content)
        elif ext == '.kt':
            return self._extract_kotlin_functions(file_path, content)
        else:
            return self._extract_generic_functions(file_path, content)
    
    def _extract_python_functions(self, file_path: str, content: str) -> List[CodeBlock]:
        """Extract Python functions and methods"""
        functions = []
        lines = content.split('\n')
        
        # Pattern for function/method definitions
        func_pattern = re.compile(r'^\s*(def|async def)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
        class_pattern = re.compile(r'^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)')
        
        current_class = None
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check for class definition
            class_match = class_pattern.match(line)
            if class_match:
                current_class = class_match.group(1)
                i += 1
                continue
            
            # Check for function definition
            func_match = func_pattern.match(line)
            if func_match:
                func_name = func_match.group(2)
                if current_class:
                    func_name = f"{current_class}.{func_name}"
                
                start_line = i + 1
                indent_level = len(line) - len(line.lstrip())
                
                # Find end of function
                j = i + 1
                while j < len(lines):
                    if lines[j].strip() == '':
                        j += 1
                        continue
                    
                    line_indent = len(lines[j]) - len(lines[j].lstrip())
                    if line_indent <= indent_level and lines[j].strip():
                        break
                    j += 1
                
                end_line = j
                func_content = '\n'.join(lines[i:end_line])
                
                functions.append(CodeBlock(
                    file_path=file_path,
                    function_name=func_name,
                    content=func_content,
                    start_line=start_line,
                    end_line=end_line
                ))
                
                i = j
            else:
                i += 1
        
        return functions
    
    def _extract_javascript_functions(self, file_path: str, content: str) -> List[CodeBlock]:
        """Extract JavaScript/TypeScript functions"""
        functions = []
        lines = content.split('\n')
        
        # Multiple patterns for JS/TS functions
        patterns = [
            re.compile(r'^\s*function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('),
            re.compile(r'^\s*const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\([^)]*\)\s*=>\s*{'),
            re.compile(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*function\s*\('),
            re.compile(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*{'),
        ]
        
        for i, line in enumerate(lines):
            for pattern in patterns:
                match = pattern.match(line)
                if match:
                    func_name = match.group(1)
                    start_line = i + 1
                    
                    # Find matching closing brace
                    brace_count = line.count('{') - line.count('}')
                    j = i + 1
                    
                    while j < len(lines) and brace_count > 0:
                        brace_count += lines[j].count('{') - lines[j].count('}')
                        j += 1
                    
                    end_line = j
                    func_content = '\n'.join(lines[i:end_line])
                    
                    functions.append(CodeBlock(
                        file_path=file_path,
                        function_name=func_name,
                        content=func_content,
                        start_line=start_line,
                        end_line=end_line
                    ))
                    break
        
        return functions
    
    def _extract_java_csharp_functions(self, file_path: str, content: str) -> List[CodeBlock]:
        """Extract Java/C# methods"""
        functions = []
        lines = content.split('\n')
        
        # Pattern for method definitions
        method_pattern = re.compile(r'^\s*(?:public|private|protected|static|final|override|async|virtual)*\s*\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
        
        for i, line in enumerate(lines):
            match = method_pattern.match(line)
            if match:
                func_name = match.group(1)
                start_line = i + 1
                
                # Find matching closing brace
                brace_count = line.count('{') - line.count('}')
                j = i + 1
                
                while j < len(lines) and brace_count > 0:
                    brace_count += lines[j].count('{') - lines[j].count('}')
                    j += 1
                
                end_line = j
                func_content = '\n'.join(lines[i:end_line])
                
                functions.append(CodeBlock(
                    file_path=file_path,
                    function_name=func_name,
                    content=func_content,
                    start_line=start_line,
                    end_line=end_line
                ))
        
        return functions
    
    def _extract_c_cpp_functions(self, file_path: str, content: str) -> List[CodeBlock]:
        """Extract C/C++ functions"""
        functions = []
        lines = content.split('\n')
        
        # Pattern for function definitions
        func_pattern = re.compile(r'^\s*\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*{')
        
        for i, line in enumerate(lines):
            match = func_pattern.match(line)
            if match:
                func_name = match.group(1)
                start_line = i + 1
                
                # Find matching closing brace
                brace_count = line.count('{') - line.count('}')
                j = i + 1
                
                while j < len(lines) and brace_count > 0:
                    brace_count += lines[j].count('{') - lines[j].count('}')
                    j += 1
                
                end_line = j
                func_content = '\n'.join(lines[i:end_line])
                
                functions.append(CodeBlock(
                    file_path=file_path,
                    function_name=func_name,
                    content=func_content,
                    start_line=start_line,
                    end_line=end_line
                ))
        
        return functions
    
    def _extract_php_functions(self, file_path: str, content: str) -> List[CodeBlock]:
        """Extract PHP functions"""
        functions = []
        lines = content.split('\n')
        
        # Pattern for PHP function definitions
        func_pattern = re.compile(r'^\s*(?:public|private|protected|static)?\s*function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
        
        for i, line in enumerate(lines):
            match = func_pattern.match(line)
            if match:
                func_name = match.group(1)
                start_line = i + 1
                
                # Find matching closing brace
                brace_count = line.count('{') - line.count('}')
                j = i + 1
                
                while j < len(lines) and brace_count > 0:
                    brace_count += lines[j].count('{') - lines[j].count('}')
                    j += 1
                
                end_line = j
                func_content = '\n'.join(lines[i:end_line])
                
                functions.append(CodeBlock(
                    file_path=file_path,
                    function_name=func_name,
                    content=func_content,
                    start_line=start_line,
                    end_line=end_line
                ))
        
        return functions
    
    def _extract_ruby_functions(self, file_path: str, content: str) -> List[CodeBlock]:
        """Extract Ruby methods"""
        functions = []
        lines = content.split('\n')
        
        # Pattern for Ruby method definitions
        func_pattern = re.compile(r'^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*[?!]?)')
        
        for i, line in enumerate(lines):
            match = func_pattern.match(line)
            if match:
                func_name = match.group(1)
                start_line = i + 1
                
                # Find matching 'end'
                j = i + 1
                while j < len(lines) and not re.match(r'^\s*end\s*$', lines[j]):
                    j += 1
                
                end_line = j + 1
                func_content = '\n'.join(lines[i:end_line])
                
                functions.append(CodeBlock(
                    file_path=file_path,
                    function_name=func_name,
                    content=func_content,
                    start_line=start_line,
                    end_line=end_line
                ))
        
        return functions
    
    def _extract_go_functions(self, file_path: str, content: str) -> List[CodeBlock]:
        """Extract Go functions"""
        functions = []
        lines = content.split('\n')
        
        # Pattern for Go function definitions
        func_pattern = re.compile(r'^\s*func\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
        
        for i, line in enumerate(lines):
            match = func_pattern.match(line)
            if match:
                func_name = match.group(1)
                start_line = i + 1
                
                # Find matching closing brace
                brace_count = line.count('{') - line.count('}')
                j = i + 1
                
                while j < len(lines) and brace_count > 0:
                    brace_count += lines[j].count('{') - lines[j].count('}')
                    j += 1
                
                end_line = j
                func_content = '\n'.join(lines[i:end_line])
                
                functions.append(CodeBlock(
                    file_path=file_path,
                    function_name=func_name,
                    content=func_content,
                    start_line=start_line,
                    end_line=end_line
                ))
        
        return functions
    
    def _extract_rust_functions(self, file_path: str, content: str) -> List[CodeBlock]:
        """Extract Rust functions"""
        functions = []
        lines = content.split('\n')
        
        # Pattern for Rust function definitions
        func_pattern = re.compile(r'^\s*(?:pub\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
        
        for i, line in enumerate(lines):
            match = func_pattern.match(line)
            if match:
                func_name = match.group(1)
                start_line = i + 1
                
                # Find matching closing brace
                brace_count = line.count('{') - line.count('}')
                j = i + 1
                
                while j < len(lines) and brace_count > 0:
                    brace_count += lines[j].count('{') - lines[j].count('}')
                    j += 1
                
                end_line = j
                func_content = '\n'.join(lines[i:end_line])
                
                functions.append(CodeBlock(
                    file_path=file_path,
                    function_name=func_name,
                    content=func_content,
                    start_line=start_line,
                    end_line=end_line
                ))
        
        return functions
    
    def _extract_swift_functions(self, file_path: str, content: str) -> List[CodeBlock]:
        """Extract Swift functions"""
        functions = []
        lines = content.split('\n')
        
        # Pattern for Swift function definitions
        func_pattern = re.compile(r'^\s*(?:public|private|internal|fileprivate|open)?\s*func\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
        
        for i, line in enumerate(lines):
            match = func_pattern.match(line)
            if match:
                func_name = match.group(1)
                start_line = i + 1
                
                # Find matching closing brace
                brace_count = line.count('{') - line.count('}')
                j = i + 1
                
                while j < len(lines) and brace_count > 0:
                    brace_count += lines[j].count('{') - lines[j].count('}')
                    j += 1
                
                end_line = j
                func_content = '\n'.join(lines[i:end_line])
                
                functions.append(CodeBlock(
                    file_path=file_path,
                    function_name=func_name,
                    content=func_content,
                    start_line=start_line,
                    end_line=end_line
                ))
        
        return functions
    
    def _extract_kotlin_functions(self, file_path: str, content: str) -> List[CodeBlock]:
        """Extract Kotlin functions"""
        functions = []
        lines = content.split('\n')
        
        # Pattern for Kotlin function definitions
        func_pattern = re.compile(r'^\s*(?:public|private|protected|internal)?\s*fun\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
        
        for i, line in enumerate(lines):
            match = func_pattern.match(line)
            if match:
                func_name = match.group(1)
                start_line = i + 1
                
                # Find matching closing brace
                brace_count = line.count('{') - line.count('}')
                j = i + 1
                
                while j < len(lines) and brace_count > 0:
                    brace_count += lines[j].count('{') - lines[j].count('}')
                    j += 1
                
                end_line = j
                func_content = '\n'.join(lines[i:end_line])
                
                functions.append(CodeBlock(
                    file_path=file_path,
                    function_name=func_name,
                    content=func_content,
                    start_line=start_line,
                    end_line=end_line
                ))
        
        return functions
    
    def _extract_generic_functions(self, file_path: str, content: str) -> List[CodeBlock]:
        """Generic function extraction for unsupported languages"""
        functions = []
        lines = content.split('\n')
        
        # Generic patterns that might catch function-like structures
        patterns = [
            re.compile(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*{'),
            re.compile(r'^\s*function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('),
        ]
        
        for i, line in enumerate(lines):
            for pattern in patterns:
                match = pattern.match(line)
                if match:
                    func_name = match.group(1)
                    start_line = i + 1
                    
                    # Simple brace matching
                    brace_count = line.count('{') - line.count('}')
                    j = i + 1
                    
                    while j < len(lines) and brace_count > 0:
                        brace_count += lines[j].count('{') - lines[j].count('}')
                        j += 1
                    
                    end_line = j
                    func_content = '\n'.join(lines[i:end_line])
                    
                    functions.append(CodeBlock(
                        file_path=file_path,
                        function_name=func_name,
                        content=func_content,
                        start_line=start_line,
                        end_line=end_line
                    ))
                    break
        
        return functions
    
    def analyze_codebase(self, directory: str) -> List[CodeBlock]:
        """Main method to analyze entire codebase"""
        console.print(f"🔍 [bold blue]Analyzing codebase in: {directory}[/bold blue]")
        
        # Find all code files
        code_files = self.find_code_files(directory)
        
        if not code_files:
            console.print("❌ [red]No code files found in the specified directory[/red]")
            return []
        
        # Extract functions from all files
        all_functions = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("🔍 Extracting functions from code files...", total=len(code_files))
            
            for file_path in code_files:
                functions = self.extract_functions_from_file(file_path)
                all_functions.extend(functions)
                progress.advance(task, 1)
        
        console.print(f"✅ [green]Found {len(all_functions)} functions across {len(code_files)} files[/green]")
        return all_functions


async def main():
    """Main function to run the Zoom recording to code analysis"""
    load_dotenv()
    
    # Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        console.print("❌ [red]OpenAI API key not found. Please set OPENAI_API_KEY environment variable.[/red]")
        return
    
    console.print(Panel.fit(
        "[bold cyan]🎯 Zoom Recording → Code Implementation Analyzer[/bold cyan]\n"
        "[dim]Analyzes Zoom recordings and matches discussion topics to code implementations[/dim]",
        border_style="cyan"
    ))
    
    # Get input parameters
    recording_path = input("📹 Enter path to Zoom recording (video/audio): ").strip()
    if not recording_path:
        console.print("❌ [red]Recording path is required[/red]")
        return
    
    codebase_path = input("📁 Enter path to codebase directory: ").strip()
    if not codebase_path:
        console.print("❌ [red]Codebase path is required[/red]")
        return
    
    try:
        # Initialize processors
        zoom_processor = ZoomRecordingProcessor(OPENAI_API_KEY)
        code_analyzer = CodeAnalyzer()
        matcher = TranscriptCodeMatcher(OPENAI_API_KEY)
        
        # Step 1: Process Zoom recording
        console.print("\n📹 [bold blue]Step 1: Processing Zoom Recording[/bold blue]")
        transcript = await zoom_processor.transcribe_zoom_recording(recording_path)
        
        # Step 2: Analyze codebase
        console.print("\n💻 [bold blue]Step 2: Analyzing Codebase[/bold blue]")
        code_blocks = code_analyzer.analyze_codebase(codebase_path)
        
        # Step 3: Extract topics from transcript
        console.print("\n🎯 [bold blue]Step 3: Extracting Topics from Transcript[/bold blue]")
        topics = await matcher.extract_topics_from_transcript(transcript)
        
        # Step 4: Find matching code blocks
        console.print("\n🔍 [bold blue]Step 4: Finding Code Matches[/bold blue]")
        matches = await matcher.find_matching_code_blocks(topics, code_blocks)
        
        # Step 5: Generate report
        console.print("\n📊 [bold blue]Step 5: Generating Analysis Report[/bold blue]")
        report = matcher.generate_beautiful_report(matches)
        
        console.print("\n🎉 [bold green]Analysis Complete![/bold green]")
        
    except Exception as e:
        console.print(f"❌ [red]Error during analysis: {e}[/red]")
        logger.error(f"Analysis error: {e}", exc_info=True)
    
    finally:
        # Cleanup
        if 'zoom_processor' in locals():
            zoom_processor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
            