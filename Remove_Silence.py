# Import required libraries
import numpy as np
from pydub.silence import split_on_silence
from pydub import AudioSegment, effects 
from scipy.io.wavfile import read, write
# Pass audio path
for i in range(100,500):
    path =f'./{i}.wav'
    rate, audio = read(path)
    # make the audio in pydub audio segment format
    aud = AudioSegment(audio.tobytes(),frame_rate = rate,
                        sample_width = audio.dtype.itemsize,channels = 1)
    # use split on sience method to split the audio based on the silence, 
    # here we can pass the min_silence_len as silent length threshold in ms and intensity thershold
    audio_chunks = split_on_silence(
        aud,
        min_silence_len = 200,
        silence_thresh = -45,
        keep_silence = 50,)
    #audio chunks are combined here
    audio_processed = sum(audio_chunks)
    # audio_processed = np.array(audio_processed.get_array_of_samples())
    #Note the processed audio rate is not the same - it would be 1K
    audio_processed = effects.normalize(audio_processed) 
    audio_processed.export(f'./tuned_{i}.wav',format = 'wav')
