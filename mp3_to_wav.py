# from pydub import AudioSegment
import os
import subprocess
folder_path = './Audio_file'
filenames = os.listdir(folder_path)
index = 2000
for filename in filenames:
    old = folder_path+"/{}.mp3".format(index)
    new = folder_path+"/{}.wav".format(index)
    subprocess.call(['ffmpeg','-i',old,new])
    # sound = AudioSegment.from_mp3(old)
    # sound.export(new, format="wav")
    index +=1
