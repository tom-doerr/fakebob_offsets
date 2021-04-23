
## Copyright (C) 2019, Guangke Chen <gkchen.shanghaitech@gmail.com>.
## This program is licenced under the BSD 2-Clause licence
## contained in the LICENCE file in this directory.


import numpy as np
import os
from ivector_PLDA_OSI import iv_OSI
from ivector_PLDA_CSI import iv_CSI
from ivector_PLDA_SV import iv_SV
from gmm_ubm_OSI import gmm_OSI
from gmm_ubm_CSI import gmm_CSI
from gmm_ubm_SV import gmm_SV
from scipy.io.wavfile import read
import pickle

debug = False
n_jobs = 24

test_dir = "./data/test-set"
adversarial_dir = "./adversarial-audio"
illegal_dir = "./data/illegal-set"

model_dir = "model"
spk_id_list = ["1580", "2830", "4446", "5142", "61"]  # Change to your own spk ids !!!!
iv_model_paths = [os.path.join(model_dir, spk_id + ".iv") for spk_id in spk_id_list]
gmm_model_paths = [os.path.join(model_dir, spk_id + ".gmm") for spk_id in spk_id_list]

iv_model_list = []
gmm_model_list = []
for path in iv_model_paths:
    with open(path, "rb") as reader:
        model = pickle.load(reader)
        iv_model_list.append(model)

for path in gmm_model_paths:
    with open(path, "rb") as reader:
        model = pickle.load(reader)
        gmm_model_list.append(model)

pre_model_dir = "pre-models"
ubm = os.path.join(pre_model_dir, "final.dubm")


def set_threshold(score_target, score_untarget):

    if not isinstance(score_target, np.ndarray):
        score_target = np.array(score_target)
    if not isinstance(score_untarget, np.ndarray):
        score_untarget = np.array(score_untarget)

    n_target = score_target.size
    n_untarget = score_untarget.size

    final_threshold = 0.
    min_difference = np.infty
    final_far = 0.
    final_frr = 0.
    for candidate_threshold in score_target:

        frr = np.argwhere(score_target < candidate_threshold).flatten().size * 100 / n_target
        far = np.argwhere(score_untarget >= candidate_threshold).flatten().size * 100 / n_untarget
        difference = np.abs(frr - far)
        if difference < min_difference:
            final_threshold = candidate_threshold
            final_far = far
            final_frr = frr
            min_difference = difference

    return final_threshold, final_frr, final_far



def gen_audio_offset_list(audio, num_max_offset_sampmles, step=1):
    audio_offset_list = []
    for num_offset_samples in range(0, num_max_offset_sampmles, step):
        offset = np.zeros(num_offset_samples, dtype=np.int16)
        audio_with_offset = np.concatenate([offset, audio])
        audio_offset_list.append(audio_with_offset)
    return audio_offset_list

''' Test for gmm-ubm-based SV
'''
spk_iter = os.listdir(illegal_dir)
illegal_audio_list = []
for spk_id in spk_iter:
    spk_dir = os.path.join(illegal_dir, spk_id)
    audio_iter = os.listdir(spk_dir)
    for i, audio_name in enumerate(audio_iter):
        path = os.path.join(spk_dir, audio_name)
        _, audio = read(path)
        illegal_audio_list.append(audio)




NUM_MODELS = 1
NUM_AUDIOS = 1
NUM_POINTS_TO_PLOT = int(1e3)
MAX_OFFSET = int(1e5)
offset_step = int(MAX_OFFSET / NUM_POINTS_TO_PLOT)

score_target = []
score_untarget = []
offset_scores_dict = {}
for model in gmm_model_list[:NUM_MODELS]:

    spk_id = model[0]
    print("spk_id:", spk_id)
    spk_id_extra = "test-gmm-SV-" + spk_id
    gmm_sv_model = gmm_SV(spk_id_extra, model, ubm)

    # normal audio
    audio_list = []
    spk_dir = os.path.join(test_dir, spk_id)
    audio_iter = os.listdir(spk_dir)
    for i, audio_name in enumerate(audio_iter):
        path = os.path.join(spk_dir, audio_name)
        _, audio = read(path)
        audio_list.append(audio)

    # adversarial audio
    adversarial_audio_list = []
    spk_dir = os.path.join(adversarial_dir, 'gmm-SV-targeted' , spk_id)
    audio_iter = os.listdir(spk_dir)
    for i, audio_name in enumerate(audio_iter):
        path = os.path.join(spk_dir, audio_name)
        path_full = os.path.join(path, [e for e in os.listdir(path) if '.wav' in e and \
            not 'ot_' in e and not 'otrl_' in e][0])
        _, audio = read(path_full)
        adversarial_audio_list.append(audio)

    # adversarial audio ot
    adversarial_audio_list_ot = []
    spk_dir = os.path.join(adversarial_dir, 'gmm-SV-targeted' , spk_id)
    audio_iter = os.listdir(spk_dir)
    for i, audio_name in enumerate(audio_iter):
        path = os.path.join(spk_dir, audio_name)
        path_full = os.path.join(path, [e for e in os.listdir(path) if '.wav' in e and \
            'ot_' in e][0])
        _, audio = read(path_full)
        adversarial_audio_list_ot.append(audio)

    # adversarial audio otrl
    adversarial_audio_list_otrl = []
    spk_dir = os.path.join(adversarial_dir, 'gmm-SV-targeted' , spk_id)
    audio_iter = os.listdir(spk_dir)
    for i, audio_name in enumerate(audio_iter):
        path = os.path.join(spk_dir, audio_name)
        path_full = os.path.join(path, [e for e in os.listdir(path) if '.wav' in e and \
            'otrl_' in e][0])
        _, audio = read(path_full)
        adversarial_audio_list_otrl.append(audio)


    _, scores = gmm_sv_model.make_decisions(audio_list, n_jobs=n_jobs, debug=debug)
    score_target += [score for score in scores]
    print("score_target:", score_target)

    _, scores = gmm_sv_model.make_decisions(illegal_audio_list, n_jobs=n_jobs, debug=debug)
    score_untarget += [score for score in scores]
    print("score_untarget:", score_untarget)

    print("audio_list:", audio_list)
    audio_list_divided_by_num_elements = [e/NUM_AUDIOS for e in audio_list]
    print("audio_list_divided_by_num_elements:", audio_list_divided_by_num_elements)
    max_audio_length = max([len(e) for e in audio_list])
    min_audio_length = min([len(e) for e in audio_list])
    #audio_list_mean = np.zeros((1,max_audio_length))
    audio_list_mean = [np.zeros(min_audio_length)]
    for e in audio_list[:NUM_AUDIOS]:
        #audio_list_mean[0][:len(e)] += e
        audio_list_mean[0] += e[:min_audio_length]

    list_dict = {'normal': audio_list, 'illegal': illegal_audio_list, 'adversarial': adversarial_audio_list, 'normal_mean': audio_list_mean,
            'adversarial_ot': adversarial_audio_list_ot, 'adversarial_otrl': adversarial_audio_list_otrl}
    for type_of_list in ['normal', 'illegal', 'adversarial', 'adversarial_ot', 'adversarial_otrl', 'normal_mean']:
        print("type_of_list:", type_of_list)
        #current_list = audio_list if type_of_list == 'normal' else illegal_audio_list
        current_list = list_dict[type_of_list]
        num_audios = min([NUM_AUDIOS, len(current_list)])
        for audio_num in range(num_audios):
            audio_offset_list = gen_audio_offset_list(current_list[audio_num], MAX_OFFSET, offset_step)
            _, scores = gmm_sv_model.make_decisions(audio_offset_list, n_jobs=n_jobs, debug=debug)
            offset_scores_dict[type_of_list + '_' + spk_id + '_' + str(audio_num)] = [score for score in scores]

print("offset_scores_dict:", offset_scores_dict)


threshold, frr, far = set_threshold(score_target, score_untarget)
print("----- Test of gmm-ubm-based SV, result ---> threshold: %f FRR: %f, FAR: %f" % (threshold, frr, far))

with open('offset_scores/out_dict.py', 'w') as f:
    f.write('offset_dict = ' + str(offset_scores_dict))






