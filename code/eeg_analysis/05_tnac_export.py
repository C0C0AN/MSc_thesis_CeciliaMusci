# --- Jose C. Garcia Alanis
# --- utf-8
# --- Python 3.6.2
#
#--- EEG prepocessing - DPX TT
# --- Version Sep 2018
#
# --- Apply baseline, crop for smaller file, and
# --- Export epochs to .txt

# ==================================================================================================
# ------------------------------ Import relevant extensions ----------------------------------------
import glob
import os

# import numpy as np

import mne
# from mne.time_frequency import tfr_morlet

import pickle

# ========================================================================
# --- GLOBAL SETTINGS
# --- SET PATH to data
base_path = '/media/cmusci/ceccmusci/ga_trial/'
# output_path = '/Users/Josealanis/Documents/Experiments/dpx_tt/eeg/output/'
# results_path = '/Volumes/TOSHIBA/manuscripts_and_data/dpx_tt/'

# list of files
#for all subs:
#files = sorted(glob.glob(os.path.join(base_path + 'mne_tnac_epochs/', '*-epo.fif')))

#for each participant group:
#files = sorted(glob.glob(os.path.join(base_path + 'mne_epochs_musicians/', '*-epo.fif')))
#files = sorted(glob.glob(os.path.join(base_path + 'mne_epochs_non_mus/', '*-epo.fif')))
#files = sorted(glob.glob(os.path.join(base_path + 'mne_epochs_left/', '*-epo.fif')))
files = sorted(glob.glob(os.path.join(base_path + 'mne_epochs_rightnonmus/', '*-epo.fif')))

# ---  Global variables to store results ---
# allEpochs = list()
music = list()
voice = list()
song = list()

# === LOOP THROUGH FILES AND EXPORT EPOCHS =========================================================
# --- No baseline correction for time-frequency analyses
for file in files:
    # Set file names
    filepath, filename = os.path.split(file)
    filename, ext = os.path.splitext(filename)
    #name = str(filename.split('_')[0]) + '_tnac_epochs'
    #name = str(filename.split('_')[0]) + '_epochs_musicians'
    #name = str(filename.split('_')[0]) + '_epochs_non_mus'
    name = str(filename.split('_')[0]) + '_epochs_rightnonmus'

    # Read epochs
    epochs = mne.read_epochs(file, preload=True)
    # # Only keep time window from -2 to 2.5 sec. around motor response
    # epochs = epochs.copy().crop(tmin=-2., tmax=2.5)

    # Sample down to improve computation time? (uncomment if needed)
    # epochs.resample(100., npad='auto')

    # Copy of A-epochs
    music_epochs = epochs['music'].copy()
    # Copy of B-epochs
    voice_epochs = epochs['voice'].copy()
    # Copy of B-epochs
    song_epochs = epochs['song'].copy()

    # Store them in a list for later
    music.append(music_epochs)
    voice.append(voice_epochs)
    song.append(song_epochs)

    # # --- Parameters for wavelet convolution ---
    # # # decimation factor
    # # decim = 3
    #
    # # Frequency space (number of frequencies)
    # freqs = np.logspace(*np.log10([1, 50]), num=30)
    # # Number of cycles (here I'm changing number of cycles per frequency,
    # # i.e., improving time-frequency resolution trade-off)
    # n_cycles = np.logspace(*np.log10([1, 10]), num=len(freqs))
    # # n_cycles = freqs / 2.  # different number of cycle per frequency
    #
    # # === Compute TOTAL power ===
    # # Wavelet convolution for A-epochs
    # power_a, itc_a = tfr_morlet(A_epochs,
    #                             freqs=freqs,
    #                             n_cycles=n_cycles,
    #                             use_fft=True,
    #                             return_itc=True,
    #                             # decim=decim,
    #                             n_jobs=2)
    #
    # # Wavelet convolution for B-Epochs
    # power_b, itc_b = tfr_morlet(B_epochs,
    #                             freqs=freqs,
    #                             n_cycles=n_cycles,
    #                             use_fft=True,
    #                             return_itc=True,
    #                             # decim=decim,
    #                             n_jobs=2)
    #
    # # Save total power
    # Total_A.append(power_a)
    # Total_B.append(power_b)
    #
    # # Save ITC
    # ITC_A.append(itc_a)
    # ITC_B.append(itc_b)
    #
    # # === Compute ERP power ===
    # # Wavelet convolution for A-ERP
    # erp_power_a = tfr_morlet(A_epochs.copy().average(),
    #                          freqs=freqs,
    #                          n_cycles=n_cycles,
    #                          use_fft=True,
    #                          return_itc=False,
    #                          # decim=decim,
    #                          n_jobs=2)
    #
    # # Wavelet convolution for B-ERP
    # erp_power_b = tfr_morlet(B_epochs.copy().average(),
    #                          freqs=freqs,
    #                          n_cycles=n_cycles,
    #                          use_fft=True,
    #                          return_itc=False,
    #                          # decim=decim,
    #                          n_jobs=2)
    #
    # # save ERP power
    # Evoked_A.append(erp_power_a)
    # Evoked_B.append(erp_power_b)
    # # Evoked_A.append(A_epochs.copy())
    # # Evoked_B.append(B_epochs.copy())
    #
    # # === Compute INDUCED power ===
    # # Wavelet convolution for A-ERP
    # induced_power_a = tfr_morlet(A_epochs.copy().subtract_evoked(),
    #                              freqs=freqs,
    #                              n_cycles=n_cycles,
    #                              use_fft=True,
    #                              return_itc=False,
    #                              # decim=decim,
    #                              n_jobs=2)
    #
    # # Wavelet convolution for B-ERP
    # induced_power_b = tfr_morlet(B_epochs.copy().subtract_evoked(),
    #                              freqs=freqs,
    #                              n_cycles=n_cycles,
    #                              use_fft=True,
    #                              return_itc=False,
    #                              # decim=decim,
    #                              n_jobs=2)
    #
    # # save INDUCED power
    # Induced_A.append(induced_power_a)
    # Induced_B.append(induced_power_b)

# ==================================================================================================
# ---------------------------------------- save results --------------------------------------------

# check if output path exists
#if not os.path.exists(base_path + 'mne_tnac_results/'):
    #os.makedirs(base_path + 'mne_tnac_results/')
#if not os.path.exists(base_path + 'mne_tnac_results_musicians/'): #musicians group
    #os.makedirs(base_path + 'mne_tnac_results_musicians/') #musicians
#if not os.path.exists(base_path + 'mne_tnac_results_non_mus/'): #non musicians group
    #os.makedirs(base_path + 'mne_tnac_results_non_mus/')
#if not os.path.exists(base_path + 'mne_tnac_results_left/'): #left
    #os.makedirs(base_path + 'mne_tnac_results_left/')
if not os.path.exists(base_path + 'mne_tnac_results_rightnonmus/'):
    os.makedirs(base_path + 'mne_tnac_results_rightnonmus/')

# path to save results
#results = base_path + 'mne_tnac_results/'
#results = base_path + 'mne_tnac_results_musicians/'
#results = base_path + 'mne_tnac_results_left/'
#results = base_path + 'mne_tnac_results_left/'
results = base_path + 'mne_tnac_results_rightnonmus/'

# --- save results ---
erp_dict = dict()
# --- ERP DATA ---
# tf_dict["all_epochs"] = []
# tf_dict["all_epochs"].append(allEpochs)
erp_dict["music"] = []
erp_dict["music"].append(music)
erp_dict["voice"] = []
erp_dict["voice"].append(voice)
erp_dict["song"] = []
erp_dict["song"].append(song)
# save results
#with open(results + 'erp_data_dict.pkl', 'wb') as erp:
#with open(results + 'erp_musicians_data_dict.pkl', 'wb') as erp: #musicians
    #pickle.dump(erp_dict, erp)

with open(results + 'erp_rightnonmus_data_dict.pkl', 'wb') as erp: #musicians
    pickle.dump(erp_dict, erp)
#
#
# # --- TOTAL TFR ---
# total_tfr_dict = dict()
# # total TFR
# total_tfr_dict["Total_A"] = []
# total_tfr_dict["Total_A"].append(Total_A)
# total_tfr_dict["Total_B"] = []
# total_tfr_dict["Total_B"].append(Total_B)
# # ITC
# total_tfr_dict["ITC_A"] = []
# total_tfr_dict["ITC_A"].append(ITC_A)
# total_tfr_dict["ITC_B"] = []
# total_tfr_dict["ITC_B"].append(ITC_B)
# # save results
# with open(results + 'total_tfr_dict.pkl', 'wb') as total:
#     pickle.dump(total_tfr_dict, total)
#
# # --- ERP TFR ---
# erp_tfr_dict = dict()
# # evoked TFR
# erp_tfr_dict["Evoked_A"] = []
# erp_tfr_dict["Evoked_A"].append(Evoked_A)
# erp_tfr_dict["Evoked_B"] = []
# erp_tfr_dict["Evoked_B"].append(Evoked_B)
# # save results
# with open(results + 'erp_tfr_dict.pkl', 'wb') as evoked:
#     pickle.dump(erp_tfr_dict, evoked)
#
# # --- INDUCED TFR ---
# induced_tfr_dict = dict()
# # induced TFR
# induced_tfr_dict["Induced_A"] = []
# induced_tfr_dict["Induced_A"].append(Induced_A)
# induced_tfr_dict["Induced_B"] = []
# induced_tfr_dict["Induced_B"].append(Induced_B)
# # save results
# with open(results + 'induced_tfr_dict.pkl', 'wb') as induced:
#     pickle.dump(induced_tfr_dict, induced)
