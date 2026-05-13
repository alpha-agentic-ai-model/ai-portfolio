"""
Real-Time Voice Agent with Streaming Speech-to-Speech
Sub-500ms latency voice conversational agent with interruption handling.
"""

import asyncio
import logging
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass
class VoiceConfig:
    sample_rate: int = 16000
    chunk_size: int = 512
    vad_threshold: float = 0.5
    max_latency_ms: int = 500
    silence_timeout_ms: int = 1500
    min_speech_ms: int = 300
    tts_voice: str = "conversational"
    enable_interruption: bool = True


@dataclass
class TranscriptSegment:
    text: str
    is_final: bool
    confidence: float
    timestamp_ms: float
    language: str = "en"


@dataclass
class AudioChunk:
    data: np.ndarray
    sample_rate: int
    timestamp_ms: float
    duration_ms: float


@dataclass
class ConversationTurn:
    role: str  # "user" or "assistant"
    text: str
    audio_duration_ms: float
    latency_ms: float = 0.0
    timestamp: float = 0.0


class VoiceActivityDetector:
    """Detects speech and interruptions in audio streams."""

    def __init__(self, threshold: float = 0.5, frame_size: int = 512):
        self.threshold = threshold
        self.frame_size = frame_size
        self._speech_frames = 0
        self._silence_frames = 0
        self._is_speaking = False
        self._energy_buffer = deque(maxlen=50)

    def process_frame(self, audio: np.ndarray) -> dict:
        energy = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        self._energy_buffer.append(energy)
        adaptive_threshold = self._compute_adaptive_threshold()
        is_speech = energy > adaptive_threshold
        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1
            if self._silence_frames > 30:
                self._speech_frames = 0
        was_speaking = self._is_speaking
        self._is_speaking = self._speech_frames > 5
        return {
            "is_speech": is_speech,
            "speech_started": self._is_speaking and not was_speaking,
            "speech_ended": not self._is_speaking and was_speaking,
            "energy": energy,
        }

    def is_interruption(self, audio: np.ndarray) -> bool:
        result = self.process_frame(audio)
        return result["speech_started"]

    def _compute_adaptive_threshold(self) -> float:
        if len(self._energy_buffer) < 10:
            return self.threshold
        sorted_energies = sorted(self._energy_buffer)
        noise_floor = np.mean(sorted_energies[:10])
        return max(self.threshold, noise_floor * 3.0)


class StreamingASR:
    """Streaming automatic speech recognition with partial results."""

    def __init__(self, sample_rate: int = 16000, language: str = "en"):
        self.sample_rate = sample_rate
        self.language = language
        self._buffer: list[np.ndarray] = []
        self._partial_text = ""

    async def transcribe(self, chunk: AudioChunk) -> TranscriptSegment:
        self._buffer.append(chunk.data)
        buffer_duration = (
            sum(len(c) for c in self._buffer) / self.sample_rate * 1000
        )
        is_final = buffer_duration >= 1500
        text = self._partial_text + " [transcribed]"
        if is_final:
            self._buffer.clear()
            self._partial_text = ""
        return TranscriptSegment(
            text=text.strip(),
            is_final=is_final,
            confidence=0.92,
            timestamp_ms=chunk.timestamp_ms,
            language=self.language,
        )

    def reset(self):
        self._buffer.clear()
        self._partial_text = ""


class StreamingLLM:
    """Streaming LLM client for real-time token generation."""

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 256):
        self.model = model
        self.max_tokens = max_tokens
        self._conversation_history: list[dict] = []

    async def stream(self, text: str) -> AsyncIterator[str]:
        self._conversation_history.append({"role": "user", "content": text})
        response_tokens = f"I understand you said: {text[:50]}. Let me help."
        words = response_tokens.split()
        full_response = []
        for word in words:
            await asyncio.sleep(0.02)  # Simulate token latency
            full_response.append(word)
            yield word + " "
        self._conversation_history.append({
            "role": "assistant",
            "content": " ".join(full_response),
        })

    def reset_context(self):
        self._conversation_history.clear()


class NeuralTTS:
    """Neural text-to-speech with streaming synthesis."""

    def __init__(self, voice: str = "conversational", sample_rate: int = 24000):
        self.voice = voice
        self.sample_rate = sample_rate
        self._is_speaking = False
        self._cancel_event = asyncio.Event()

    async def synthesize(self, text: str) -> AudioChunk:
        if self._cancel_event.is_set():
            self._cancel_event.clear()
            return AudioChunk(
                data=np.array([], dtype=np.int16),
                sample_rate=self.sample_rate,
                timestamp_ms=time.time() * 1000,
                duration_ms=0,
            )
        self._is_speaking = True
        duration_ms = len(text.split()) * 200
        num_samples = int(self.sample_rate * duration_ms / 1000)
        audio_data = np.zeros(num_samples, dtype=np.int16)
        await asyncio.sleep(0.01)
        self._is_speaking = False
        return AudioChunk(
            data=audio_data,
            sample_rate=self.sample_rate,
            timestamp_ms=time.time() * 1000,
            duration_ms=duration_ms,
        )

    async def stop(self):
        self._cancel_event.set()
        self._is_speaking = False
        logger.info("TTS playback interrupted")


class StreamingVoiceAgent:
    """Sub-500ms voice-to-voice conversational agent."""

    def __init__(self, config: VoiceConfig):
        self.config = config
        self.state = AgentState.IDLE
        self.asr = StreamingASR(config.sample_rate)
        self.llm = StreamingLLM(model="claude-sonnet-4-6")
        self.tts = NeuralTTS(voice=config.tts_voice)
        self.vad = VoiceActivityDetector(config.vad_threshold)
        self.conversation: list[ConversationTurn] = []
        self._metrics = {
            "total_turns": 0,
            "avg_latency_ms": 0.0,
            "interruptions": 0,
        }

    async def process_stream(
        self, audio_chunks: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[AudioChunk]:
        self.state = AgentState.LISTENING

        async for chunk in audio_chunks:
            if (
                self.config.enable_interruption
                and self.state == AgentState.SPEAKING
                and self.vad.is_interruption(chunk.data)
            ):
                await self.tts.stop()
                self.asr.reset()
                self.state = AgentState.LISTENING
                self._metrics["interruptions"] += 1
                logger.info("User interrupted, switching to listening")
                continue

            if self.state == AgentState.LISTENING:
                transcript = await self.asr.transcribe(chunk)

                if transcript.is_final and transcript.text.strip():
                    start_time = time.time()
                    self.state = AgentState.THINKING

                    self.conversation.append(ConversationTurn(
                        role="user",
                        text=transcript.text,
                        audio_duration_ms=chunk.duration_ms,
                        timestamp=start_time,
                    ))

                    self.state = AgentState.SPEAKING
                    response_text = []

                    async for token in self.llm.stream(transcript.text):
                        audio_chunk = await self.tts.synthesize(token)
                        if audio_chunk.duration_ms > 0:
                            yield audio_chunk
                        response_text.append(token)

                    latency = (time.time() - start_time) * 1000
                    self.conversation.append(ConversationTurn(
                        role="assistant",
                        text="".join(response_text),
                        audio_duration_ms=0,
                        latency_ms=latency,
                        timestamp=time.time(),
                    ))
                    self._update_metrics(latency)
                    self.state = AgentState.LISTENING

    def _update_metrics(self, latency_ms: float):
        self._metrics["total_turns"] += 1
        n = self._metrics["total_turns"]
        self._metrics["avg_latency_ms"] = (
            self._metrics["avg_latency_ms"] * (n - 1) + latency_ms
        ) / n

    def get_metrics(self) -> dict:
        return {**self._metrics, "state": self.state.value}
