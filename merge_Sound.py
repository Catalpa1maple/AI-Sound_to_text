from pydub import AudioSegment

sound1 = AudioSegment.from_wav('./2000.wav')
sound2 = AudioSegment.from_wav('./2001.wav')

out_sounds = sound1+sound2
out_sounds.export('./text.wav',format = 'wav')