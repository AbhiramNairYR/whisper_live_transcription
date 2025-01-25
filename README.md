# whisper_live_transcription

This is a demo of real time speech to text with OpenAI's Whisper model. It works by constantly recording audio in a thread and concatenating the raw bytes over multiple recordings.This is created by editing https://github.com/davabase/whisper_real_time.git and making use of docker if you want to only run it on your local env then go to his repository.

First clone this repository:
```
git clone https://github.com/AbhiramNairYR/whisper_live_transcription.git
```


Then run the docker image from docker hub:
```
docker pull abhiramnyr/fin_wislan
```


To run the docker image:
```
docker run -it --rm -v /home/abhiram/code/python/live/whisper_real_time:/workspace --gpus all --device /dev/snd --group-add $(getent group audio | cut -d: -f3) -e PULSE_SERVER=unix:/run/user/1000/pulse/native -v /run/user/1000/pulse:/run/user/1000/pulse fin_wislan

```

To run the pyton file run:
```
python transcribe_demo.py --device_index 9
```
if --device_index 9 dosent work try from 0 to 10

other arguments:
Model to use:
>--model  

default="medium" 
choices=["tiny", "base", "small", "medium", "large"])

Don't use the english model:
>--non_english
>action='store_true'

Energy level for mic to detect:
>--energy_threshold

default=1000

How real time the recording is in seconds
>--record_timeout

default=30

How much empty space between recordings before we consider it a new line in the transcription:
>--phrase_timeout

default=0.5

Specific device index for microphone:
>--device_index

default=None

Path to mock audio file for testing:
>--mock_audio

default=None

List all available microphone devices
>--list-devices

action="store_true"


For more information on Whisper please see https://github.com/openai/whisper

The code in this repository is public domain.
