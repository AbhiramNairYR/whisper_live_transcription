# whisper_live_transcription

This is a demo of real time speech to text with OpenAI's Whisper model. It works by constantly recording audio in a thread and concatenating the raw bytes over multiple recordings.This is created by editing https://github.com/davabase/whisper_real_time.git and making use of docker.

To install dependencies simply run
```
docker run -it --rm     -v /home/abhiram/code/python/live/whisper_real_time:/workspace     --gpus all     --device /dev/snd     --group-add $(getent group audio | cut -d: -f3)     -e PULSE_SERVER=unix:/run/user/1000/pulse/native     -v /run/user/1000/pulse:/run/user/1000/pulse     fin_wislan

```

For more information on Whisper please see https://github.com/openai/whisper

The code in this repository is public domain.
