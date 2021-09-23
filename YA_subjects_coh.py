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

cwt_freqs = np.arange(4, 40, 1)
def Coherence(sub_number):

    path = os.path.join("YA_subject"+str(sub_number)+"_RH_condition.epo.fif")
    epochs = mne.read_epochs(path)

    cwt_freqs = np.arange(4, 40, 1)

    # Calculate coh for every channel, time and freq
    con, freqs, times, _, _ = spectral_connectivity(
        epochs, mode='cwt_morlet', sfreq=250, tmin=1, tmax=7, cwt_freqs=cwt_freqs)
    # print(con.shape)
    #
    # # Plotting coh matrix for one time period for one freq
    # conmat = con[:, :, 0]
    # plt.figure(figsize=(12, 9))
    # sns.heatmap(conmat[:, :, 0], annot=True, fmt='.1f')
    # plt.show()


    con_b, _, times_b, _, _ = spectral_connectivity(
         epochs, mode='cwt_morlet', sfreq=250, tmin=1, tmax=3, cwt_freqs=cwt_freqs)

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
                for t in range(501):
                    con[i, j, f, t] = con[i, j, f, t] - con_base[i,j,f]

    return con[:, :, :, :]


# con0 = Coherence(0)
# con1 = Coherence(1)
# con2 = Coherence(2)
# con3 = Coherence(3)
# con4 = Coherence(4)
# con5 = Coherence(5)
# con6 = Coherence(6)
# con7 = Coherence(7)
# con8 = Coherence(8)
# con9 = Coherence(9)
#
#
# np.savez('persons_with_baseline', con0, con1, con2, con3, con4, con5, con6, con7, con8, con9)
# npzfile = np.load('persons_with_baseline.npz')
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
# # # #
# av_con = np.average([con0, con1, con2, con3, con4, con5, con6, con7, con8, con9], axis=0)
#
#
# np.save('av_coh_with_baseline', av_con)

av_con = np.load('av_coh_with_baseline.npy')
print(av_con.shape)
times = np.arange(0, 1501, 1)
# # # #
# first_ch = av_con[20, 15, :, :]
# # #
# conmat_b = av_con[:, :, 0]
# plt.figure(figsize=(12, 9))
# sns.heatmap(conmat_b[:, :, 1], annot=True, fmt='.1f')
# plt.show()

# plt.figure(figsize=(12, 9))
# p1, = plt.plot(times, first_ch[0], marker='.')
# p2, = plt.plot(times, first_ch[20], marker='.')
# p3, = plt.plot(times, first_ch[10], marker='.')
# plt.legend([p2, p1, p3], ["24Hz", "4HZ", "14Hz"])
# plt.xlabel('time')
# plt.ylabel('coherence')
# plt.show()


all_ch=np.average(av_con,axis=(0,1))
# print(all_ch.shape)
# print(times.shape)
# print(cwt_freqs.shape)
# plt.figure(figsize=(12, 9))
# plt.plot(times, all_ch[0], color='green', marker='.')
# plt.show()

B, D = np.meshgrid(times, cwt_freqs)
# fig = plt.figure()
# ax = Axes3D(fig)
# surf = ax.plot_surface(B, D, all_ch, cmap=cm.coolwarm)
# plt.xlabel('time')
# plt.ylabel('freqs')
# plt.colorbar(surf)
# plt.show()

import matplotlib.colors as colors
# vmin = -0.3
# vmax = 0.3
# level_boundaries = np.linspace(vmin, vmax)
# fig = plt.figure()
# contour = plt.contourf(B/250-2, D, all_ch, cmap=cm.coolwarm, norm=colors.Normalize(vmin=-0.3, vmax=0.3))
# plt.colorbar(contour)
# # cbar.ax.set_ylabel('coherence')
# # cbar.ax.set_clim(-0.3,0.3)
# plt.xlabel('time, s')
# plt.ylabel('freqs, Hz')
#
# plt.show()

fig, ax = plt.subplots(2, 1)
pcm = ax[0].pcolor(B/250-2, D, all_ch,
                   norm=colors.Normalize(vmin=-0.3, vmax=0.3),
                   cmap='coolwarm', shading='auto')
fig.colorbar(pcm, ax=ax[0], extend='max')
plt.show()





