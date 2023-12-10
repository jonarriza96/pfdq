#%%
%matplotlib tk
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pfdq.utils.quaternion_ph import rotation_to_quaternion
from pfdq.results.utils import (
    visualize_path_frames,
    visualize_path_frames_spatial,
    visualize_path_frames_spatial_top,
    visualize_path_frames_spatial_front,
    visualize_path_frames_spatial_side,
)
from pfdq.utils.pfdq import get_desired_velocity_profile

# -------------------------------- import data ------------------------------- #
data_path = "/home/jonarriza96/pfdq/pfdq/results/data/case_study2/"  # 27night/"
save_path = "/home/jonarriza96/pfdq/pfdq/results/figures/case_study2/"

# load pf "0,1,2,3"
file_name = "pf_f_5"
pf_path = data_path + file_name + ".pickle"
with open(pf_path, "rb") as handle:
    dataset = pickle.load(handle)
thf5 = dataset["X"][:, -2]
thDf5 = dataset["X"][:, -1]
Qf5 = dataset["X"][:, 6:10]
QEf5 = dataset["QE"]
PEf5 = dataset["PE"]
pff5_P = dataset["P"]
pff5_ERF = dataset["ERF"]
pff5_Pd = dataset["PD"]
xif_5 = dataset["SP"][:, 0]
dperpf_5 = np.linalg.norm(dataset["SP"][:, 1:3], axis=1) ** 2
ph_spline = dataset["ph_spline"]

file_name = "pf_s_5"
pf_path = data_path + file_name + ".pickle"
with open(pf_path, "rb") as handle:
    dataset = pickle.load(handle)
ths5 = dataset["X"][:, -2]
thDs5 = dataset["X"][:, -1]
Qs5 = dataset["X"][:, 6:10]
pfs5_P = dataset["P"]
pfs5_ERF = dataset["ERF"]
pfs5_Pd = dataset["PD"]
xis_5 = dataset["SP"][:, 0]
dperps_5 = np.linalg.norm(dataset["SP"][:, 1:3], axis=1) ** 2

file_name = "pf_v_5"
pf_path = data_path + file_name + ".pickle"
with open(pf_path, "rb") as handle:
    dataset = pickle.load(handle)
thv5 = dataset["X"][:, -2]
thDv5 = dataset["X"][:, -1]
Qv5 = dataset["X"][:, 6:10]
pfv5_P = dataset["P"]
pfv5_ERF = dataset["ERF"]
pfv5_Pd = dataset["PD"]
xiv_5 = dataset["SP"][:, 0]
dperpv_5 = np.linalg.norm(dataset["SP"][:, 1:3], axis=1) ** 2

file_name = "pf_i_5"
pf_path = data_path + file_name + ".pickle"
with open(pf_path, "rb") as handle:
    dataset = pickle.load(handle)
thi5 = dataset["X"][:, -2]
thDi5 = dataset["X"][:, -1]
Qi5 = dataset["X"][:, 6:10]
QEi5 = dataset["QE"]
PEi5 = dataset["PE"]
pfi5_P = dataset["P"]
pfi5_ERF = dataset["ERF"]
pfi5_Pd = dataset["PD"]
xii_5 = dataset["SP"][:, 0]
dperpi_5 = np.linalg.norm(dataset["SP"][:, 1:3], axis=1) ** 2


ph_spline_p = ph_spline["gamma"]
ph_spline_erf = ph_spline["erf"]
#%%
######################### front #########################
fig = plt.figure()
ax = fig.add_subplot(111)

stop_ind2 = -1000
ax = visualize_path_frames_spatial_front(
    pd=pfi5_P[:, :100],
    erf=pfi5_ERF[:, :, :100],
    scale=0.17,
    ax=ax,
    secondary=True,
    interval=4,
)

i_start = -4000
ax = visualize_path_frames_spatial_front(
    pd=pfi5_P[:, i_start:],
    erf=pfi5_ERF[:, :, i_start:],
    scale=0.17,
    ax=ax,
    secondary=True,
    interval=20,
)

ax = visualize_path_frames_spatial_front(
    pd=pff5_P[:, :stop_ind2],
    erf=pff5_ERF[:, :, :stop_ind2],
    scale=0.12,
    ax=ax,
    secondary=False,
    interval=27,
)

ax.plot(pfi5_P[0, :100], pfi5_P[2, :100], "--", color="gray", alpha=0.5)
ax.plot(pfi5_P[0, i_start:], pfi5_P[2, i_start:], "--", color="gray", alpha=0.5)
ax.plot(pff5_P[0, :], pff5_P[2, :], color="blueviolet")#, alpha=1)#,linewidth=1.0)
ax.plot(pfs5_P[0, :], pfs5_P[2, :], color="forestgreen", alpha=0.75)
ax.plot(pfv5_P[0, :], pfv5_P[2, :], color="darkorange", alpha=0.75)

ax.plot(ph_spline_p[0, :], ph_spline_p[2, :], color="black", alpha=1.0, linewidth=4.0)
ax = visualize_path_frames_spatial_front(
    pd=ph_spline_p, erf=ph_spline_erf, scale=0.1, ax=ax, secondary=False, interval=40
)
# ax.set_xlim([-1.1,1.1])
plt.axis("off")
plt.axis('equal')
plt.tight_layout()
# fig.set_size_inches(9, 3)
# fig.savefig(save_path + 'front_col1.pdf',dpi=1800)

#%%
######################### top #########################
fig = plt.figure()
ax = fig.add_subplot(111)



ax.plot(pfi5_P[0, :100], pfi5_P[1, :100], "--", color="gray", alpha=0.5)
ax.plot(pfi5_P[0, -4000:], pfi5_P[1, -4000:], "--", color="gray", alpha=0.5)
ax = visualize_path_frames_spatial_top(
    pd=pff5_P[:, :stop_ind2],
    erf=pff5_ERF[:, :, :stop_ind2],
    scale=0.12,
    ax=ax,
    secondary=False,
    interval=30,
)

ax.plot(pff5_P[0, :], pff5_P[1, :], color="blueviolet")#, alpha=1)#,linewidth=1)
ax.plot(pfs5_P[0, :], pfs5_P[1, :], color="forestgreen", alpha=0.75)
ax.plot(pfv5_P[0, :], pfv5_P[1, :], color="darkorange", alpha=0.75)

ax.plot(ph_spline_p[0, :], ph_spline_p[1, :], color="black", alpha=1.0, linewidth=4.0)
ax = visualize_path_frames_spatial_top(
    pd=ph_spline_p, erf=ph_spline_erf, scale=0.1, ax=ax, secondary=False, interval=40
)



plt.axis("off")
plt.axis('equal')
plt.tight_layout()
# fig.set_size_inches(9, 3)
# fig.savefig(save_path + 'front_col2.pdf',dpi=1800)

#%%
######################### thetadot #########################

gf, _ = get_desired_velocity_profile("f")
gs, _ = get_desired_velocity_profile("s")
gv, _ = get_desired_velocity_profile("v")

thf5_modified = np.hstack([np.linspace(0.6,thf5[0],100),thf5])
thf_d = np.squeeze([gf(th) for th in thf5_modified])
ths_d = np.squeeze([gs(th) for th in thf5_modified])
thv_d = np.squeeze([gv(th) for th in thf5_modified])

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(thf5_modified, thf_d, "--",color="magenta")#, alpha=0.75,linewidth=lw)  # thp
ax.plot(thf5, thDf5, color="blueviolet",linewidth=2)#, alpha=0.75,linewidth=lw)  # thp

ax.plot(thf5_modified, ths_d, "--",color="limegreen")#, alpha=0.75,linewidth=lw)  # thp
ax.plot(ths5, thDs5, color="forestgreen",linewidth=2)#, alpha=0.85,linewidth=lw)  # thm

ax.plot(thf5_modified, thv_d, "--",color="goldenrod")#, alpha=0.75,linewidth=lw)  # thp
ax.plot(thv5, thDv5, color="darkorange",linewidth=2)#, alpha=0.75,linewidth=lw)  # thc

ax.set_yticks([0.0,0.018,0.075])
ax.set_yticklabels(['0','0.019','0.075'])
ax.set_xticks([0.65,0.85,1])
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_ylim([0,0.08])
ax.set_xlim([0.62,1.01])
ax.set_ylabel(r"$\dot{\theta}$",fontsize=16,labelpad=-30)
ax.set_xlabel(r"$\theta$",fontsize=16,labelpad=5)

# ax.grid(axis='x')
fig.set_size_inches(3.5, 3.5)
plt.tight_layout()
# fig.savefig(save_path + 'vprofile_col1.pdf',dpi=1800)

#%%
######################### errors #########################
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True)
plt.setp(ax, xticks=[0.65, 0.85, 1.0], xlim = [0.62,1.01])#, xticklabels=['a', 'b', 'c']
# for AX in ax:
#     AX.grid(axis='x')
ax1,ax2 = ax
ax11 = ax1[0]
ax12 = ax1[1]
ax21 = ax2[0]
ax22 = ax2[1]

ax11.plot(thf5, PEf5.T[:,0],'r')
ax11.plot(thf5, PEf5.T[:,1],'g')
ax11.plot(thf5, PEf5.T[:,2],'b')
ax11.tick_params(axis='y', which='major', labelsize=14)
ax11.tick_params(axis='x', which='major', labelsize=14,length=0)
ax11.set_ylabel(r"$p_e$ [m]",fontsize=16,labelpad=1)

ax21.plot(thf5, QEf5.T[:,0],'r')
ax21.plot(thf5, QEf5.T[:,1],'g')
ax21.plot(thf5, QEf5.T[:,2],'b')
ax21.plot(thf5, QEf5.T[:,3],'k')
ax21.tick_params(axis='both', which='major', labelsize=14)
ax21.set_ylabel(r"$q_e$",fontsize=16,labelpad=1)
ax21.set_xlabel(r"$\theta$",fontsize=16,labelpad=1)

ax22.plot(thi5, QEi5.T[:,0],'r')
ax22.plot(thi5, QEi5.T[:,1],'g')
ax22.plot(thi5, QEi5.T[:,2],'b')
ax22.plot(thi5, QEi5.T[:,3],'k')
ax22.tick_params(axis='both', which='major', labelsize=14)
ax22.set_ylabel(r"$q_e$",fontsize=16,labelpad=1)
ax22.set_xlabel(r"$\theta$",fontsize=16,labelpad=1)
# ax12.grid(axis='x')


ax12.plot(thi5, PEi5.T[:,0],'r')
ax12.plot(thi5, PEi5.T[:,1],'g')
ax12.plot(thi5, PEi5.T[:,2],'b')
ax22.tick_params(axis='x', which='major', labelsize=14, length=0)
ax12.tick_params(axis='y', which='major', labelsize=14)
ax12.set_ylabel(r"$p_e$ [m]",fontsize=16,labelpad=1)
# ax22.grid(axis='x')

fig.set_size_inches(6.5, 3.5)
plt.tight_layout()

fig.savefig(save_path + 'errors_col1.pdf',dpi=1800)

# fig.subplots_adjust(hspace=.1,wspace=.1)
# ax21.plot(thf5_modified, thv_d, "--",color="goldenrod")