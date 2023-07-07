import argparse
import os
import struct
import wave
import shutil

from utils.gui import *
from threading import Thread
from pvrecorder import PvRecorder

from component.asr import ASR

class Demo(Thread):

    def __init__(
            self,
            input_device_index=None,
            whisper='base'):

        super(Demo, self).__init__()

        self._input_device_index = input_device_index
        self.asr = ASR(model=whisper)

    @classmethod
    def show_audio_devices(cls):
        devices = PvRecorder.get_audio_devices()
        for i in range(len(devices)):
            print('index: %d, device name: %s' % (i, devices[i]))

    def run(self):

        try:
            print(True)
            recorder = PvRecorder(device_index=self._input_device_index, frame_length=512, log_overflow=False)
            recorder.start()

            temp_dir = './tmp'
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.mkdir(temp_dir)


            print('Using device: %s' % recorder.selected_device)
            print('Listening...')

            wav_file = wave.open(f'{temp_dir}{"/audio.wav"}', "w")
            wav_file.setparams((1, 2, 16000, 512, "NONE", "NONE"))

            while True:
                pcm = recorder.read()
                wav_file.writeframes(struct.pack("h" * len(pcm), *pcm))

        except KeyboardInterrupt:

            print('\nASR Processing...')
            wav_file.close()
            self.asr.calc(f'{temp_dir}{"/audio.wav"}',outputJson=False)


        finally:
            if recorder is not None:
                recorder.delete()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--whisper', type=str, default="medium", help="whisper model ie. tiny, base, small, medium, large-v2")
    parser.add_argument('--audio_device_index', help='Index of input audio device.', type=int, default=-1)
    parser.add_argument('--show_audio_devices', action='store_true')

    args = parser.parse_args()
    print(args)

    if args.show_audio_devices:
        Demo.show_audio_devices()

    demo = Demo(
        input_device_index=args.audio_device_index,
        whisper=args.whisper)
    demo.run()

if __name__ == '__main__':
    main()
