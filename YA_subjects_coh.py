import mne
import numpy as np
from mne.connectivity import spectral_connectivity
from matplotlib import cm
import seaborn as sns
from heatmap import heatmap, corrplot
from mne.viz import circular_layout, plot_connectivity_circle
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

# cwt_freqs = np.arange(4, 40, 1)
freqs = np.arange(4, 40, 1)

def Coherence(sub_number):
    path = os.path.join("YA_subject"+str(sub_number)+"_RH_condition.epo.fif")
    epochs = mne.read_epochs(path)

    # Calculate coh for every channel, time and freq
    freqs = np.arange(4, 40, 1)


    con, freqs, times, _, _ = spectral_connectivity(
        epochs, mode='cwt_morlet', sfreq=250, tmin=1, tmax=7, cwt_freqs = freqs)

    con_b, _, times_b, _, _ = spectral_connectivity(
        epochs, mode='cwt_morlet', sfreq=250, tmin=1, tmax=3, cwt_freqs=freqs)

    # print(con.shape)
    # print(times_b.shape)
    n_rows, n_cols = con_b.shape[:2]

    # Calculating baseline level
    con_base = np.average(con_b, 3)
    # print(con_base.shape)
    #
    for i in range(n_rows):
        for j in range(i + 1):
            for f in range(36):
                for t in range(1501):
                    con[i, j, f, t] = con[i, j, f, t] - con_base[i,j,f]

    return con[:, :, :, :]


con0 = Coherence(0)
con1 = Coherence(1)
con2 = Coherence(2)
con3 = Coherence(3)
con4 = Coherence(4)
con5 = Coherence(5)
con6 = Coherence(6)
con7 = Coherence(7)
con8 = Coherence(8)
con9 = Coherence(9)
# # #
# print(con9.shape)
# np.savez_compressed('persons_with_baseline_11', con0, con1, con2, con3, con4, con5, con6, con7, con8, con9)
# npzfile = np.load('persons_with_baseline_11.npz')
# sorted(npzfile.files)
# con0 = npzfile['arr_0']
# con1 = npzfile['arr_1']
# con2 = npzfile['arr_2']
# con3 = npzfile['arr_3']
# con4 = npzfile['arr_4']
# con5 = npzfile['arr_5']
# con6 = npzfile['arr_6']
# con7 = npzfile['arr_7']
# con8 = npzfile['arr_8']
# con9 = npzfile['arr_9']
#
# # #
av_con = np.average([con0, con1, con2, con3, con4, con5, con6, con7, con8, con9], axis=0)
# # med_con = np.median([con0, con1, con2, con3, con4, con5, con6, con7, con8, con9], axis=0)
# #
# np.save('av_coh_with_baseline111', av_con)
# np.save('median_coh_with_baseline12', med_con)

# av_con = np.load('av_coh_with_baseline.npy')
# av_con = np.load('median_coh_with_baseline.npy')
# # print(av_con.shape)
times = np.arange(0, 1501, 1)
#
cwt_freqs = np.arange(4, 40, 1)
# #
# # # # # #
# first_ch = av_con[20, 15, :, :]
#
# plt.figure(figsize=(12, 9))
# p1, = plt.plot(times, first_ch[0], marker='.')
# p2, = plt.plot(times, first_ch[20], marker='.')
# p3, = plt.plot(times, first_ch[10], marker='.')
# plt.legend([p2, p1, p3], ["24Hz", "4HZ", "14Hz"])
# plt.xlabel('time')
# plt.ylabel('coherence')
# plt.show()
#

all_ch=np.average(av_con, axis=(0,1))
B, D = np.meshgrid(times, cwt_freqs)

fig = plt.figure()
levels = np.arange(-0.035, 0.035, 0.005)
contour = plt.contourf(B/250-2, D, all_ch, levels=levels, cmap=cm.coolwarm)
cb = plt.colorbar(contour)
cb.ax.set_ylabel('coherence')
plt.xlabel('time, s')
plt.ylabel('freqs, Hz')

plt.show()






