import json
import whisper
import torch
import spacy
import datetime

from spacy.matcher import *
from utils.general import Profile

class ASR:

    def __init__(self, model='medium') -> None:
        print("\033[96mLoading Whisper Model..\033[0m", end='', flush=True)
        torch.cuda.empty_cache()
        self.asr_model = whisper.load_model(model)
        self.nlp = spacy.load("en_core_web_sm")

        self.alphabet = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray", "yankee", "zulu"]
        self.drugs = ["txa", "fentanyl", "blood", "ketamine"] # more options?
        self.measurement_unit = ["mg","milligrams", "milligram", "g", "ml"] # need more units/ outputs that whisper might interperet
        print("\033[90m Done.\033[0m")

        self.timer = Profile()

    def calc(self, audio_source, outputJson=True):
        with self.timer:
            print(f"\n\033[90mTranscribing using {self.asr_model.device}...\033[0m")
            transcription = self.asr_model.transcribe(audio_source, word_timestamps=True)
            text = transcription['text'].lower()
            print(transcription)
            print(transcription.keys())

            matcher = Matcher(self.nlp.vocab)
            drug_pattern = [
                {"LIKE_NUM": True},
                {"LOWER": {"IN": self.measurement_unit}},
                {"LOWER": {"IN": ["of"]}, "OP": "?"},
                {"LOWER": {"IN": self.drugs}}
            ]
            matcher.add("DOSAGE_DRUG_PATIENT", [drug_pattern])

            treatment_pattern = [
                {"LOWER": "treating"},
                {"LOWER": "patient"},
                {"LOWER": {"IN": self.alphabet}}
            ]
            matcher.add("TREATING_PATIENT", [treatment_pattern])

            doc = self.nlp(text)
            matches = matcher(doc)
            print("Transcribed Text: "+text)

            if outputJson:
                with open("./tmp/incomplete.json", "r") as json_input:
                    json_dict = json.load(json_input)
                    json_dict['audio'] = text
                    json_input.close()

                with open("./tmp/complete.json", "w") as json_output:
                    for match_id, start, end in matches:

                        # Getting index in order to time stamp in wav
                        span = doc[start: end]
                        words = text.split()
                        index = span.start_char
                        startWord = None
                        for word in words:
                            if index >= text.index(word) and index < text.index(word) + len(word):
                                startWord = word
                        print(f'Start Word: {startWord}')

                        # Create a dictionary where the keys are the words and the values are the dictionaries containing the start index and probability
                        word_dict = {}
                        for segment in transcription['segments']:
                            loopBroken = False
                            for data in segment['words']:
                                word = data['word'].strip()
                                if word not in word_dict:
                                    word_dict[word] = {'start': data['start'], 'probability': data['probability']}
                                else:
                                    if data['start'] < word_dict[word]['start']:
                                        word_dict[word]['start'] = data['start']
                                        loopBroken = True
                                        break
                            if loopBroken:
                                break

                        # Iterate over the word dictionary to print the start index of the first occurrence of each word
                        for word, info in word_dict.items():
                            print(f"First occurrence of '{word}' starts at index {info['start']}.")


                        if self.nlp.vocab.strings[match_id] == "DOSAGE_DRUG_PATIENT":
                            dose = doc[start:start+2].text
                            drug = doc[start+3].text
                            patient = doc[end-1].text
                            print(f"Dosage: {dose}, Drug: {drug}, Patient: {patient}")
                            json_dict.get("patient").get(str(drug)).update({str(datetime.datetime.now()):str(dose)}) # using "patient" rather than patient because we dont have multiple bodies set up

                        # Not doing anything with this
                        elif self.nlp.vocab.strings[match_id] == "TREATING_PATIENT":
                            print(f"Treating patient {doc[start+2].text}, Time: {datetime.datetime.now()}")



                    json_dict['drugs_available'] = True
                    json.dump(json_dict, json_output, indent=4, default=str)
                    json_output.close()
        print(f'ASR Completed in {self.timer.dt} sec')