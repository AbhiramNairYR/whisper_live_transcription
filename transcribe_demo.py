#! python3.7

import argparse
import os
import sys
import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep

def list_microphones():
    available_mics = sr.Microphone.list_microphone_names()
    print("Available microphones:")
    for index, name in enumerate(available_mics):
        print(f"Device {index}: {name}")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=30,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=0.5,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    parser.add_argument("--device_index", type=int, default=None,
                        help="Specific device index for microphone")
    parser.add_argument("--mock_audio", type=str, default=None,
                        help="Path to mock audio file for testing")
    parser.add_argument("--list-devices", action="store_true",
                        help="List all available microphone devices")
    args = parser.parse_args()

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    # Source selection with multiple fallback strategies
    source = None
    
    if args.list_devices:
        list_microphones()

    # Check for mock audio file first
    if args.mock_audio and os.path.exists(args.mock_audio):
        print(f"Using mock audio file: {args.mock_audio}")
        source = sr.AudioFile(args.mock_audio)
    else:
        # Try to find a working microphone
        available_mics = sr.Microphone.list_microphone_names()
        
        if not available_mics:
            print("No microphones detected.")
            sys.exit(1)
        
        print("Available microphones:")
        for index, name in enumerate(available_mics):
            print(f"Device {index}: {name}")
        
        # If device_index is specified, try that first
        if args.device_index is not None:
            try:
                source = sr.Microphone(sample_rate=16000, device_index=args.device_index)
                print(f"Using specified device index: {args.device_index}")
            except Exception as e:
                print(f"Failed to use specified device: {e}")
        
        # If no specific device or it failed, try the first available
        if source is None:
            try:
                source = sr.Microphone(sample_rate=16000, device_index=0)
                print("Using first available microphone")
            except Exception as e:
                print(f"Failed to initialize microphone: {e}")
                sys.exit(1)

    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")

    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now
                
                # Combine audio data from queue
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                
                # Convert in-ram buffer to something the model can use directly without needing a temp file.
                # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Read the transcription.
                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                # If we detected a pause between recordings, add a new item to our transcription.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                # Clear the console to reprint the updated transcription.
                os.system('cls' if os.name=='nt' else 'clear')
                for line in transcription:
                    print(line)
                # Flush stdout.
                print('', end='', flush=True)
            else:
                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)

if __name__ == "__main__":
    main()
