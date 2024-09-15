import os 
from flask import Flask, request

app = Flask(__name__)

os.environ['TRANSFORMERS_CACHE'] = os.environ.get('TRANSFORMERS_CACHE', os.path.join(os.getcwd(), 'hf_cache'))
os.environ['HF_HOME'] = os.environ['TRANSFORMERS_CACHE']

from peft import set_peft_model_state_dict
from model_util import get_model_from_config
import copy, librosa, torch, io
import soundfile as sf


def get_config(config_file):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("module.name", config_file)
    config = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = config
    spec.loader.exec_module(config)
    return config


processor, model = get_model_from_config(get_config('configs/training_set_v1_r32_a64_d0.05_whisperv2_28_8_1_b32_qvonly.py'), is_training=False)
checkpoint_file = "weights/training_set_v1_r32_a64_d0.05_whisperv2_28_8_1_b32_qvonly/epoch_1.pth"
checkpoint = torch.load(checkpoint_file, map_location='cpu')
set_peft_model_state_dict(model, checkpoint['model_state_dict'])

tokenizer = processor.tokenizer
feature_extractor = processor.feature_extractor
task = "transcribe"
forced_decoder_ids = processor.get_decoder_prompt_ids(language='chinese', task=task)

    
@app.route('/tester', methods=['POST'])
def receive_wav():
        file = request.files['files']
        audio = './AI_test_temp/tmp.wav'
        if os.path.isfile(audio):
            os.remove(audio)
        file.save(audio)
        sample, sr  = sf.read(audio)
        #Alter sample size here: Default 16000
        target_sr = 16000
        sample = sample.T
        sample = librosa.to_mono(sample)
        sample = librosa.resample(sample, orig_sr=sr, target_sr=target_sr)
        input_features = processor(sample, sampling_rate=target_sr, return_tensors="pt").input_features
        generation_config = copy.deepcopy(
            model.generation_config
        )

        with torch.cuda.amp.autocast():
            input_features = input_features.to(model.device).half()
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids, generation_config=generation_config)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
        sen = transcription[0]
        print(sen[57:-13])
        return sen[57:-13], 200
    

if __name__ == '__main__':
    app.run(
            debug=True,
            host='0.0.0.0', 
            port=5001,
            )
