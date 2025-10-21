"""
Transcription service for converting YouTube videos to text.

This module provides functionality to get YouTube video transcripts using
YouTube's built-in captions API (youtube-transcript-api) as the primary method,
with OpenAI Whisper as a fallback for videos without captions.

Priority order:
1. YouTube Transcript API (fast, free, accurate for videos with captions)
2. OpenAI Whisper (slower, requires download, works for any video)
"""

import os
import tempfile
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.formatters import TextFormatter
    YOUTUBE_TRANSCRIPT_AVAILABLE = True
except ImportError:
    YOUTUBE_TRANSCRIPT_AVAILABLE = False
    raise ImportError(
        "youtube-transcript-api is not installed. Please install it with: pip install youtube-transcript-api"
    )

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

from .config_service import config_service


logger = logging.getLogger(__name__)


class TranscriptionService:
    """Service for transcribing YouTube videos using YouTube Transcript API and Whisper fallback."""

    def __init__(self):
        """Initialize transcription service."""
        self.whisper_model: Optional[Any] = None
        self.whisper_model_name = "base"  # Can be changed to "small", "medium", "large"

    def _load_whisper_model(self) -> Any:
        """Load Whisper model lazily."""
        if not WHISPER_AVAILABLE:
            raise ImportError("OpenAI Whisper is not available. Install with: pip install openai-whisper")

        if self.whisper_model is None:
            logger.info(f"Loading Whisper model: {self.whisper_model_name}")
            import whisper
            self.whisper_model = whisper.load_model(self.whisper_model_name)
        return self.whisper_model

    def get_youtube_transcript(self, video_id: str, language: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get transcript using YouTube Transcript API.

        Args:
            video_id: YouTube video ID
            language: Preferred language code

        Returns:
            Transcript data or None if not available
        """
        if not YOUTUBE_TRANSCRIPT_AVAILABLE:
            return None

        try:
            # Get available transcripts for the video
            transcript_list = YouTubeTranscriptApi().list(video_id)

            # Try preferred language first
            if language:
                try:
                    transcript = transcript_list.find_transcript([language])
                except:
                    transcript = None
            else:
                transcript = None

            # If no preferred language transcript, try English, then any available
            if not transcript:
                try:
                    transcript = transcript_list.find_transcript(['en'])  # Try English first
                except:
                    try:
                        # Get first available transcript
                        available_transcripts = list(transcript_list)
                        if available_transcripts:
                            transcript = available_transcripts[0]
                        else:
                            return None
                    except:
                        return None

            # Fetch the transcript data
            transcript_data = transcript.fetch()

            # Convert to dict format for consistency
            segments = []
            full_text = ""
            for snippet in transcript_data:
                segments.append({
                    'start': snippet.start,
                    'text': snippet.text,
                    'duration': snippet.duration
                })
                full_text += snippet.text + " "

            return {
                'text': full_text.strip(),
                'language': transcript.language_code,
                'segments': segments,
                'source': 'youtube_transcript_api',
                'duration': None  # YouTube API doesn't provide duration in transcript
            }

        except Exception as e:
            logger.debug(f"YouTube transcript not available for video {video_id}: {e}")
            return None

    def download_audio(self, video_url: str, output_path: str) -> bool:
        """
        Download audio from YouTube video for Whisper fallback.

        Args:
            video_url: YouTube video URL or ID
            output_path: Path to save the audio file

        Returns:
            True if download successful, False otherwise
        """
        if not YT_DLP_AVAILABLE:
            raise ImportError("yt-dlp is not available. Install with: pip install yt-dlp")

        try:
            import yt_dlp
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': output_path.replace('.mp3', ''),
                'quiet': True,
                'no_warnings': True,
            }

            with yt_dlp.YoutubeDL(params=ydl_opts) as ydl:  # type: ignore[arg-type]
                ydl.download([video_url])

            return True

        except Exception as e:
            logger.error(f"Failed to download audio: {e}")
            return False

    def transcribe_with_whisper(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio file using Whisper (fallback method).

        Args:
            audio_path: Path to audio file
            language: Language code (optional, auto-detect if None)

        Returns:
            Transcription result with text and metadata
        """
        try:
            model = self._load_whisper_model()

            # Transcribe with optional language specification
            result = model.transcribe(audio_path, language=language)

            return {
                'text': result['text'],
                'language': result.get('language'),
                'segments': result.get('segments', []),
                'duration': result.get('duration'),
                'source': 'whisper'
            }

        except Exception as e:
            logger.error(f"Failed to transcribe audio with Whisper: {e}")
            raise Exception(f"Whisper transcription failed: {e}")

    def transcribe_video(self, video_input: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe a YouTube video using YouTube Transcript API first, then Whisper as fallback.

        Args:
            video_input: YouTube video URL or ID
            language: Language code for transcription (optional)

        Returns:
            Transcription result
        """
        # Extract video ID
        from .youtube_service import YouTubeAPIService
        youtube_service = YouTubeAPIService()
        video_id = youtube_service.extract_video_id(video_input)

        # Try YouTube Transcript API first (fast and free)
        logger.info(f"Attempting to get transcript for video: {video_id} using YouTube Transcript API")
        transcript_result = self.get_youtube_transcript(video_id, language)

        if transcript_result:
            logger.info(f"Successfully got transcript using YouTube Transcript API")
            transcript_result['video_id'] = video_id
            transcript_result['video_url'] = f"https://www.youtube.com/watch?v={video_id}"
            return transcript_result

        # Fallback to Whisper if transcript not available
        logger.info(f"YouTube transcript not available, falling back to Whisper for video: {video_id}")

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, f"{video_id}.mp3")

            # Download audio
            logger.info(f"Downloading audio for video: {video_id}")
            if not self.download_audio(video_input, audio_path):
                raise Exception("Failed to download video audio")

            # Transcribe with Whisper
            logger.info(f"Transcribing audio for video: {video_id} using Whisper")
            result = self.transcribe_with_whisper(audio_path, language)

            # Add video metadata
            result['video_id'] = video_id
            result['video_url'] = f"https://www.youtube.com/watch?v={video_id}"

            return result

    def get_supported_languages(self) -> list:
        """Get list of supported languages for transcription."""
        return [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl",
            "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro",
            "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy",
            "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu",
            "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km",
            "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo",
            "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg",
            "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
        ]


# Global transcription service instance - lazy initialization
class _TranscriptionServiceLazy:
    """Lazy initialization wrapper for TranscriptionService."""

    def __init__(self):
        self._instance = None
        self._instance_error = None

    def __call__(self):
        if self._instance_error:
            raise self._instance_error
        if self._instance is None:
            try:
                self._instance = TranscriptionService()
            except Exception as e:
                self._instance_error = e
                raise
        return self._instance

    def __getattr__(self, name):
        return getattr(self(), name)

# Global transcription service instance
transcription_service = _TranscriptionServiceLazy()


def transcribe_youtube_video(video_input: str, language: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to transcribe a YouTube video.

    Args:
        video_input: YouTube video URL or ID
        language: Language code for transcription (optional)

    Returns:
        Transcription result
    """
    return transcription_service.transcribe_video(video_input, language)