import whisper
import numpy as np
from datasets import load_dataset
import re
import time, torchaudio

model = whisper.load_model("tiny.en")
options = whisper.DecodingOptions(language='en')

audio = np.float32(whisper.load_audio('test_en.wav'))
audio = whisper.pad_or_trim(audio)
audio = whisper.log_mel_spectrogram(audio).to(model.device)
stream = []

for i in range(15):
	t = audio[:,i * 200: (i * 200) + 201].unsqueeze(0)
	t = model.encoder(t, stream=True, step=i)
	stream.append(t)

audio = whisper.torch.cat(stream, dim=1).to(audio.dtype).contiguous()


result = whisper.decode(model, audio[0], options)

print(result.text)
