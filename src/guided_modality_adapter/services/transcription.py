import torch
import whisper
from pyannote.audio import Pipeline
from typing import List, Dict, Optional
import os

class DiarizationPipeline:
    def __init__(self, hf_token: str, model_size: str = "base", device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_token = hf_token
        self.model_size = model_size

        print(f"Loading Whisper {model_size} on {self.device}...")
        self.whisper_model = whisper.load_model(model_size, device=self.device)
        
        print("Loading Pyannote pipeline...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        ).to(torch.device(self.device))

    def process(self, audio_path: str, num_speakers: Optional[int] = None) -> List[Dict]:
        # 1. Transcribe
        transcription = self.whisper_model.transcribe(audio_path)
        
        # 2. Diarize
        diarization = self.diarization_pipeline(audio_path, num_speakers=num_speakers)
        
        # 3. Merge
        return self._merge_results(transcription['segments'], diarization)

    def _merge_results(self, segments, diarization_result) -> List[Dict]:
        final_output = []
        
        for segment in segments:
            start = segment['start']
            end = segment['end']
            text = segment['text']
            
            # Find the speaker who spoke the most during this segment
            segment_speakers = []
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                overlap_start = max(start, turn.start)
                overlap_end = min(end, turn.end)
                duration = max(0, overlap_end - overlap_start)
                
                if duration > 0:
                    segment_speakers.append((speaker, duration))
            
            if segment_speakers:
                best_speaker = sorted(segment_speakers, key=lambda x: x[1], reverse=True)[0][0]
            else:
                best_speaker = "Unknown"
                
            final_output.append({
                "start": start,
                "end": end,
                "speaker": best_speaker,
                "text": text.strip()
            })
            
        return final_output