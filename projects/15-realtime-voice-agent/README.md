# Real-Time Voice Agent with Streaming Speech-to-Speech

## Category: Deep Learning

## Description
A low-latency voice AI agent that uses streaming ASR, real-time LLM inference, and neural TTS to create natural conversational experiences. Features voice activity detection, interruption handling, emotional tone adaptation, and sub-500ms end-to-end latency through pipeline parallelism.

## Architecture
```
[Microphone] → [Streaming ASR] → [LLM (streaming)]
        ↓
[Neural TTS] → [VAD + Interruption Handler] → [Speaker]
```

## Tech Stack
- Whisper V3
- Claude API
- Coqui TTS
- WebSocket
- PyAudio
- asyncio

## Key Features
- Sub-500ms end-to-end voice latency
- Voice activity detection with interruption handling
- Streaming ASR with partial results
- Pipeline parallelism (ASR → LLM → TTS)
- Emotional tone adaptation
- WebSocket-based real-time communication
