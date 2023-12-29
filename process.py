import torch
import numpy as np
import os
import io
from scipy.io import wavfile
from scipy.signal import istft, stft
from pydub import AudioSegment
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from whisper.audio import *
import librosa
import pandas as pd
import shutil
from tqdm import tqdm

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80

def preprocess(source_adv, source_nat, split_noise, split_nat, split_output, words_dict, split_label, label="",
               tot_max = None, tot_min = None, diff_max = None, diff_min = None, samples=50):
    """Takes in source_adv and source_nat as noised and original wav directories, constructs 
    spectrogram slices and saves them in split_noise and split_nat directories"""
    os.makedirs(os.path.join(split_noise, label), exist_ok=True)
    os.makedirs(os.path.join(split_nat, label), exist_ok=True)
    os.makedirs(os.path.join(split_output, label), exist_ok=True)
    if tot_max is None:
        tot_max, tot_min, diff_max, diff_min = get_max_min(source_adv, source_nat, samples)
    splits_dict = {}
    for file in os.listdir(os.path.join(source_nat, label)):
        if '.wav' not in file:
            continue

        curr = AudioSegment.from_wav(os.path.join(source_nat, label, file))
        curr_sounds = curr.split_to_mono()
        curr_samples = [s.get_array_of_samples() for s in curr_sounds]
        curr_arr = np.array(curr_samples).T.astype(np.float32)
        curr_arr /= np.iinfo(curr_samples[0].typecode).max
        curr_res = librosa.feature.melspectrogram(y = curr_arr.squeeze(-1), sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)[..., :-1]
        curr_res = np.clip(curr_res, 1e-10, 1e5)
        log_res = np.log10(curr_res)

        adv = AudioSegment.from_wav(os.path.join(source_adv, label, file))
        adv_sounds = adv.split_to_mono()
        adv_samples = [s.get_array_of_samples() for s in adv_sounds]
        adv_arr = np.array(adv_samples).T.astype(np.float32)
        adv_arr /= np.iinfo(adv_samples[0].typecode).max
        adv_res = librosa.feature.melspectrogram(y = adv_arr.squeeze(-1), sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)[..., :-1]
        adv_res = np.clip(adv_res, 1e-10, 1e5)

        diff_res = adv_res - curr_res
        log_diff_res = np.log10(np.clip(diff_res, 1e-10, 1e5))
        noise = (log_diff_res - tot_min)/(tot_max - tot_min)
        log_res = (log_res - diff_min)/(diff_max - diff_min)

        word_count = words_dict[file.split('.')[0]]
        splits = get_splits(log_res, word_count)

        for j, split in enumerate(splits):
            curr_img = np.uint8(log_res[:, split[0]:split[1]+1] * 255)
            noise_img = np.uint8(noise[:, split[0]:split[1]+1] * 255)
            Image.fromarray(curr_img, 'L').save(os.path.join(split_nat, file.split('.')[0] + '_' + str(j) + '.png'))
            Image.fromarray(noise_img, 'L').save(os.path.join(split_noise, file.split('.')[0] + '_' + str(j) + '.png'))
        splits_dict[file] = splits

    with open(os.path.join(split_output, f"{split_label}.pickle"), 'wb') as handle:
        pickle.dump(splits_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_max_min(source_adv, source_nat, samples = 50):
    """Gets estimate of max and min values for spectrograms of source_adv and source_nat"""
    tot_max = -10
    tot_min = 0
    diff_max = -10
    diff_min = 0
    for file in os.listdir(source_nat)[:samples]:
        if '.wav' not in file:
            continue

        curr = AudioSegment.from_wav(os.path.join(files_sub, file))
        curr_sounds = curr.split_to_mono()
        curr_samples = [s.get_array_of_samples() for s in curr_sounds]
        curr_arr = np.array(curr_samples).T.astype(np.float32)

        curr_arr /= np.iinfo(curr_samples[0].typecode).max
        curr_res = librosa.feature.melspectrogram(y = curr_arr.squeeze(-1), sr=16000, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=80)[..., :-1]
        curr_res = np.clip(curr_res, 1e-10, 1e5)
        log_res = np.log10(curr_res)

        adv = AudioSegment.from_wav(os.path.join(source_adv, file))
        adv_sounds = adv.split_to_mono()
        adv_samples = [s.get_array_of_samples() for s in adv_sounds]
        adv_arr = np.array(adv_samples).T.astype(np.float32)

        adv_arr /= np.iinfo(adv_samples[0].typecode).max
        adv_res = librosa.feature.melspectrogram(y = adv_arr.squeeze(-1), sr=16000, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=80)[..., :-1]
        adv_res = np.clip(adv_res, 1e-10, 1e5)

        diff_res = adv_res - curr_res
        log_diff_res = np.log10(np.clip(diff_res, 1e-10, 1e5))

        tot_max = max(np.max(log_res), tot_max)
        tot_min = min(np.min(log_res), tot_min)
        diff_max = max(np.max(log_diff_res), diff_max)
        diff_min = min(np.min(log_diff_res), diff_min)
    return tot_max, tot_min, diff_max, diff_min

def get_splits(log_res, word_count):
    """Gets splits for clean log mel spectrogram slices"""
    diffs = np.linalg.norm(log_res[:, :-1] - log_res[:, 1:], axis=0)
    inds = diffs.argsort()[-int(3*word_count):]
    sort_inds = np.sort(inds)
    splits = []
    d = []
    prev = 0
    for ind in sort_inds:
        if ind - prev < 20:
            continue
        if len(d) > 0:
            if ind - d[len(d)-1] >= 40:
                splits.append((d[len(d)-1], ind))
                d.pop()
        
        d.insert(0, prev)
        if len(diffs) - 1 - ind < 40:
            break
        prev = ind
    if len(d) > 0:
        splits.append((d[len(d)-1], len(diffs)-1))
    return splits

def l2_clamp_or_normalize(arr, eps=None):
    """Clamp tensor to eps in L2 norm"""
    xnorm = np.linalg.norm(arr)
    if eps is not None:
        coeff = min(eps / xnorm, 1)
    else:
        coeff = 1.0 / xnorm
    return coeff * arr

def reverse_bound_from_rel_bound(wav, snr):
    """From a relative eps bound, reconstruct the absolute bound for the given batch"""
    rel_eps = np.power(10.0, float(snr) / 20)
    eps = np.linalg.norm(wav) / rel_eps
    return eps

def postprocess(output, source_nat, split_output, final_write, split_label, label=""):
    """Patches generated adversarial spectrogram slices into full spectrograms and writes them to final_write directory"""
    with open(os.path.join(split_output, f"{split_label}.pickle"), 'rb') as handle:
        splits_dict = pickle.load(handle)
    output_dict = {}
    for test in os.listdir(output):
        test_split = test.split('.')
        if test_split[-1] != "png":
            continue
        test_split2 = test_split[0].split('_')
        npy_dir = os.path.join(output, test)
        if test_split2[0] not in output_dict:
            output_dict[test_split2[0]] = {}
        img = Image.open(npy_dir)
        output_dict[test_split2[0]][int(test_split2[1])] = np.array(img.getdata()).reshape(img.size[0], img.size[1]) / 255

    for key in tqdm(output_dict.keys()):
        breaks = splits_dict[key + '.wav']
        val = output_dict[key]
        prev_break = breaks[0]
        full_arr = val[0][:prev_break[1] - prev_break[0] + 1, :]
        for i in range(1, len(val)):
            curr_break = breaks[i]
            new_arr = val[i][:curr_break[1] - curr_break[0], :]
            interp = np.tile(np.linspace(0, 1, prev_break[1]-curr_break[0]), (full_arr.shape[1], 1)).T
            full_arr = np.concatenate([full_arr[:-(prev_break[1]-curr_break[0]), :], (1-interp) * full_arr[-(prev_break[1]-curr_break[0]):, :] 
                                    + (interp) * new_arr[:(prev_break[1]-curr_break[0]), :], new_arr[(prev_break[1]-curr_break[0]):, :]], axis=0)
            prev_break = breaks[i]
        diff_res = np.power(10, 9*((full_arr + 1) / 2) - 10)
        mres = librosa.feature.inverse.mel_to_audio(diff_res.T, sr=SAMPLE_RATE, n_fft=N_FFT, 
                                                    hop_length=HOP_LENGTH, window='hamming', power=2.0)
        _, clean = wavfile.read(os.path.join(source_nat, label, key + '.wav'))
        mres = np.pad(mres, (160, clean.size - mres.size - 160), mode='constant', constant_values=0)
        eps = reverse_bound_from_rel_bound(clean, 35)
        mres = l2_clamp_or_normalize(mres, eps)
        adv = clean + mres
        wavfile.write(os.path.join(final_write, key + ".wav"), 16000, adv)