import numpy as np
import pandas as pd
import librosa
import os

import tensorflow as tf
import tensorflow_hub as hub

# Reload the model
current_direction = os.path.dirname(os.path.abspath(__file__))
deepfake_model = tf.saved_model.load(os.path.join(current_direction,"artifact/ann_human_or_bot"))


def load_wav_16k_mono(filename):
    try:
      
      """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
      sound_sample,sr=librosa.load(filename ,sr=16000)
      return sound_sample
    
    except Exception as e:
      print("load_wav_16k_mono")
      return None

    

MY_CLASSES = ['FAKE', 'REAL']


def infa_deepfake_with_scores(audio_path):
  """Return (status, label, {"FAKE": p, "REAL": p}). status 1 = success, 0 = error."""
  try:
      testing_wav_data = load_wav_16k_mono(audio_path)
      if testing_wav_data is None or len(testing_wav_data) == 0:
          raise ValueError("Could not read audio. Please upload a valid WAV/MP3/FLAC/OGG file.")

      infer = deepfake_model.signatures['serving_default']
      input_tensor = tf.convert_to_tensor(testing_wav_data, dtype=tf.float32)
      predictions = infer(input_tensor)['output_0'].numpy()

      label = MY_CLASSES[int(np.argmax(predictions))]
      scores = {c: round(float(p), 4) for c, p in zip(MY_CLASSES, predictions)}
      return 1, label, scores

  except Exception as e:
      return 0, str(e), {}


def infa_deepfake(audio_path):
  """Backwards-compatible wrapper: returns (status, label_or_error_message)."""
  status, message, _ = infa_deepfake_with_scores(audio_path)
  return status, message
