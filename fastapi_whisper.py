#!/usr/bin/env python3
# coding: utf-8

import os

import uvicorn
import whisper
from fastapi import FastAPI, File, UploadFile
from moviepy.editor import AudioClip, VideoFileClip
from starlette.responses import RedirectResponse


class FastAPI_Whisper:
    __audio_temp = "audio.mp3"
    host_dev = "127.0.0.1"
    port_dev = "8888"
    api = FastAPI(title="FastAPI to access AI Whisper's features for audio and video transcription", version="Alpha")
    
    def __init__(self):
        if len(os.sys.argv) == 1:
            host = self.host_dev
            port = int(self.port_dev)
        elif len(os.sys.argv) == 2:
            host = os.sys.argv[1]
            port = int(self.port_dev)
        elif len(os.sys.argv) == 3:
            host = os.sys.argv[1]
            port = int(os.sys.argv[2])
        uvicorn.run("fastapi_whisper:FastAPI_Whisper.api", host=host, port=port, reload=True)
    
    @api.get("/", description="The root redirect to docs")
    def root(self):
        return RedirectResponse(url="/docs")
    
    @api.post("/audio", description="This is the audio transcription")
    async def audio(self, file:UploadFile=File(media_type="audio/*")):
        audio_file = await file.read()
        audio = AudioClip(audio_file)
        audio.write_audiofile(self.__audio_temp, codec="mp3")
        model = whisper.load_model("base")  
        result = model.transcribe(self.__audio_temp, fp16=False)
        return result["text"]
    
    @api.post("/video", description="This is the video transcription")
    async def video(self, file:UploadFile=File(media_type="video/*")):
        video_file = await file.read()
        video = VideoFileClip(video_file)
        audio = video.audio
        audio.write_audiofile(self.__audio_temp, codec="mp3")
        model = whisper.load_model("base")
        result = model.transcribe(self.__audio_temp, fp16=False)
        return result["text"]

if __name__ == "__main__":
    FastAPI_Whisper()
