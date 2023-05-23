#!/usr/bin/env python3
# coding: utf-8

import os

import uvicorn
import whisper
from fastapi import FastAPI, File, UploadFile
from moviepy.editor import AudioFileClip, VideoFileClip
from starlette.responses import RedirectResponse

audio_mp3 = "audio.mp3"
host_dev = "127.0.0.1"
port_dev = "8888"
title="FastAPI to access AI Whisper's features for audio and video transcription"
version="Alpha.1.1"

api = FastAPI(title=title, version=version)
    
@api.get("/", description="The root redirect to swagger user interface")
def root():
    return RedirectResponse(url="/docs")
    
@api.post("/audio", description="This is the audio transcription")
async def audio(file:UploadFile=File()):
    if "audio" in file.content_type:
        audio_source = "audio_source." + file.filename[-3:]
        audio_content = await file.read()
        open(audio_source, "wb").write(audio_content)
        audio = AudioFileClip(audio_source)
        audio.write_audiofile(audio_mp3, codec="mp3")
        model = whisper.load_model("base")  
        result = model.transcribe(audio_mp3, fp16=False)
        os.remove(audio_source)
        os.remove(audio_mp3)
        return result["text"]
    else:
        return "This file is not an audio file"
    
@api.post("/video", description="This is the video transcription")
async def video(file:UploadFile=File()):
    if "video" in file.content_type:
        video_source = "video_source." + file.filename[-3:]
        video_content = await file.read()
        open(video_source, "wb").write(video_content)
        video = VideoFileClip(video_source)
        audio = video.audio
        audio.write_audiofile(audio_mp3, codec="mp3")
        model = whisper.load_model("base")
        result = model.transcribe(audio_mp3, fp16=False)
        os.remove(video_source)
        os.remove(audio_mp3)
        return result["text"]
    else:
        return "This file is not an video file"

if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        host = host_dev
        port = int(port_dev)
    elif len(os.sys.argv) == 2:
        host = os.sys.argv[1]
        port = int(port_dev)
    elif len(os.sys.argv) == 3:
        host = os.sys.argv[1]
        port = int(os.sys.argv[2])
    uvicorn.run("fastapi_whisper:api", host=host, port=port, reload=True)
