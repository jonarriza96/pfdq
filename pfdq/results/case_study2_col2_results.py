# %%
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
file_name = "pf_f_0"
pf_path = data_path + file_name + ".pickle"
with open(pf_path, "rb") as handle:
    dataset = pickle.load(handle)
thf0 = dataset["X"][:, -2]
thDf0 = dataset["X"][:, -1]
Qf0 = dataset["X"][:, 6:10]
pff0_P = dataset["P"]
pff0_ERF = dataset["ERF"]
pff0_Pd = dataset["PD"]
xif_0 = dataset["SP"][:, 0]
dperpf_0 = np.linalg.norm(dataset["SP"][:, 1:3], axis=1) ** 2
ph_spline = dataset["ph_spline"]

file_name = "pf_s_0"
pf_path = data_path + file_name + ".pickle"
with open(pf_path, "rb") as handle:
    dataset = pickle.load(handle)
ths0 = dataset["X"][:, -2]
thDs0 = dataset["X"][:, -1]
Qs0 = dataset["X"][:, 6:10]
pfs0_P = dataset["P"]
pfs0_ERF = dataset["ERF"]
pfs0_Pd = dataset["PD"]
xis_0 = dataset["SP"][:, 0]
dperps_0 = np.linalg.norm(dataset["SP"][:, 1:3], axis=1) ** 2

file_name = "pf_f_1"
pf_path = data_path + file_name + ".pickle"
with open(pf_path, "rb") as handle:
    dataset = pickle.load(handle)
thf1 = dataset["X"][:, -2]
thDf1 = dataset["X"][:, -1]
Qf1 = dataset["X"][:, 6:10]
pff1_P = dataset["P"]
pff1_ERF = dataset["ERF"]
pff1_Pd = dataset["PD"]
xif_1 = dataset["SP"][:, 0]
dperpf_1 = np.linalg.norm(dataset["SP"][:, 1:3], axis=1) ** 2

file_name = "pf_s_1"
pf_path = data_path + file_name + ".pickle"
with open(pf_path, "rb") as handle:
    dataset = pickle.load(handle)
ths1 = dataset["X"][:, -2]
thDs1 = dataset["X"][:, -1]
Qs1 = dataset["X"][:, 6:10]
pfs1_P = dataset["P"]
pfs1_ERF = dataset["ERF"]
pfs1_Pd = dataset["PD"]
xis_1 = dataset["SP"][:, 0]
dperps_1 = np.linalg.norm(dataset["SP"][:, 1:3], axis=1) ** 2


file_name = "pf_f_2"
pf_path = data_path + file_name + ".pickle"
with open(pf_path, "rb") as handle:
    dataset = pickle.load(handle)
thf2 = dataset["X"][:, -2]
thDf2 = dataset["X"][:, -1]
Qf2 = dataset["X"][:, 6:10]
pff2_P = dataset["P"]
pff2_ERF = dataset["ERF"]
pff2_Pd = dataset["PD"]
xif_2 = dataset["SP"][:, 0]
dperpf_2 = np.linalg.norm(dataset["SP"][:, 1:3], axis=1) ** 2

file_name = "pf_s_2"
pf_path = data_path + file_name + ".pickle"
with open(pf_path, "rb") as handle:
    dataset = pickle.load(handle)
ths2 = dataset["X"][:, -2]
thDs2 = dataset["X"][:, -1]
Qs2 = dataset["X"][:, 6:10]
pfs2_P = dataset["P"]
pfs2_ERF = dataset["ERF"]
pfs2_Pd = dataset["PD"]
xis_2 = dataset["SP"][:, 0]
dperps_2 = np.linalg.norm(dataset["SP"][:, 1:3], axis=1) ** 2

file_name = "pf_f_3"
pf_path = data_path + file_name + ".pickle"
with open(pf_path, "rb") as handle:
    dataset = pickle.load(handle)
thf3 = dataset["X"][:, -2]
thDf3 = dataset["X"][:, -1]
Qf3 = dataset["X"][:, 6:10]
pff3_P = dataset["P"]
pff3_ERF = dataset["ERF"]
pff3_Pd = dataset["PD"]
xif_3 = dataset["SP"][:, 0]
dperpf_3 = np.linalg.norm(dataset["SP"][:, 1:3], axis=1) ** 2

file_name = "pf_s_3"
pf_path = data_path + file_name + ".pickle"
with open(pf_path, "rb") as handle:
    dataset = pickle.load(handle)
ths3 = dataset["X"][:, -2]
thDs3 = dataset["X"][:, -1]
Qs3 = dataset["X"][:, 6:10]
pfs3_P = dataset["P"]
pfs3_ERF = dataset["ERF"]
pfs3_Pd = dataset["PD"]
xis_3 = dataset["SP"][:, 0]
dperps_3 = np.linalg.norm(dataset["SP"][:, 1:3], axis=1) ** 2

file_name = "pf_f_4"
pf_path = data_path + file_name + ".pickle"
with open(pf_path, "rb") as handle:
    dataset = pickle.load(handle)
thf4 = dataset["X"][:, -2]
thDf4 = dataset["X"][:, -1]
Qf4 = dataset["X"][:, 6:10]
pff4_P = dataset["P"]
pff4_ERF = dataset["ERF"]
pff4_Pd = dataset["PD"]
xif_4 = dataset["SP"][:, 0]
dperpf_4 = np.linalg.norm(dataset["SP"][:, 1:3], axis=1) ** 2

file_name = "pf_s_4"
pf_path = data_path + file_name + ".pickle"
with open(pf_path, "rb") as handle:
    dataset = pickle.load(handle)
ths4 = dataset["X"][:, -2]
thDs4 = dataset["X"][:, -1]
Qs4 = dataset["X"][:, 6:10]
pfs4_P = dataset["P"]
pfs4_ERF = dataset["ERF"]
pfs4_Pd = dataset["PD"]
xis_4 = dataset["SP"][:, 0]
dperps_4 = np.linalg.norm(dataset["SP"][:, 1:3], axis=1) ** 2


# obtain quaternion of ph spline
ph_spline_q = []
for k in range(ph_spline["erf"].shape[2]):
    erf = ph_spline["erf"][:, :, k]
    ph_spline_q += [rotation_to_quaternion(erf, swap=True)]
ph_spline_q = np.squeeze(ph_spline_q)
ph_spline_p = ph_spline["gamma"]
ph_spline_erf = ph_spline["erf"]

# %%
# --------------------------------- visualize -------------------------------- #


######################### isometric #########################
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

stop_ind1 = 4000
ax = visualize_path_frames_spatial(
    pd=pff1_P[:, :stop_ind1],
    erf=pff1_ERF[:, :, :stop_ind1],
    scale=0.15,
    ax=ax,
    secondary=True,
    interval=40,
)
ax.plot(pfs1_P[0, :], pfs1_P[1, :], pfs1_P[2, :], "--", color="peru", alpha=0.75)
ax.plot(pff1_P[0, :], pff1_P[1, :], pff1_P[2, :], color="peru", alpha=1)

stop_ind2 = 4600
ax = visualize_path_frames_spatial(
    pd=pff2_P[:, :stop_ind2],
    erf=pff2_ERF[:, :, :stop_ind2],
    scale=0.15,
    ax=ax,
    secondary=True,
    interval=40,
)
ax.plot(pfs2_P[0, :], pfs2_P[1, :], pfs2_P[2, :], "--", color="orange", alpha=0.75)
ax.plot(pff2_P[0, :], pff2_P[1, :], pff2_P[2, :], color="orange", alpha=1)

stop_ind3 = 3500
ax = visualize_path_frames_spatial(
    pd=pff3_P[:, :stop_ind3],
    erf=pff3_ERF[:, :, :stop_ind3],
    scale=0.15,
    ax=ax,
    secondary=True,
    interval=40,
)
ax.plot(pfs3_P[0, :], pfs3_P[1, :], pfs3_P[2, :], "--", color="blueviolet", alpha=0.75)
ax.plot(pff3_P[0, :], pff3_P[1, :], pff3_P[2, :], color="blueviolet", alpha=1)

stop_ind4 = 3000
ax = visualize_path_frames_spatial(
    pd=pff4_P[:, :stop_ind4],
    erf=pff4_ERF[:, :, :stop_ind4],
    scale=0.15,
    ax=ax,
    secondary=True,
    interval=20,
)
ax.plot(
    pfs4_P[0, :], pfs4_P[1, :], pfs4_P[2, :], "--", color="darkturquoise", alpha=0.75
)
ax.plot(pff4_P[0, :], pff4_P[1, :], pff4_P[2, :], color="darkturquoise", alpha=1)

ax.plot(
    ph_spline_p[0, :],
    ph_spline_p[1, :],
    ph_spline_p[2, :],
    color="black",
    alpha=1.0,
    linewidth=4.0,
)
ax = visualize_path_frames_spatial(
    pd=ph_spline_p, erf=ph_spline_erf, scale=0.15, ax=ax, secondary=False, interval=40
)


plt.axis("off")
plt.tight_layout()
fig.set_size_inches(9, 6)
# fig.savefig(save_path + "isometric_col2.pdf", dpi=1800)

# %%
######################### top #########################
fig = plt.figure()
ax = fig.add_subplot(111)

stop_ind1 = 4000
ax = visualize_path_frames_spatial_top(
    pd=pff1_P[:, :stop_ind1],
    erf=pff1_ERF[:, :, :stop_ind1],
    scale=0.15,
    ax=ax,
    secondary=True,
    interval=40,
)
ax.plot(pfs1_P[0, :], pfs1_P[1, :], "--", color="peru", alpha=0.75)
ax.plot(pff1_P[0, :], pff1_P[1, :], color="peru", alpha=1)

stop_ind2 = 4600
ax = visualize_path_frames_spatial_top(
    pd=pff2_P[:, :stop_ind2],
    erf=pff2_ERF[:, :, :stop_ind2],
    scale=0.15,
    ax=ax,
    secondary=True,
    interval=40,
)
ax.plot(pfs2_P[0, :], pfs2_P[1, :], "--", color="orange", alpha=0.75)
ax.plot(pff2_P[0, :], pff2_P[1, :], color="orange", alpha=1)

stop_ind3 = 3500
ax = visualize_path_frames_spatial_top(
    pd=pff3_P[:, :stop_ind3],
    erf=pff3_ERF[:, :, :stop_ind3],
    scale=0.15,
    ax=ax,
    secondary=True,
    interval=40,
)
ax.plot(pfs3_P[0, :], pfs3_P[1, :], "--", color="blueviolet", alpha=0.75)
ax.plot(pff3_P[0, :], pff3_P[1, :], color="blueviolet", alpha=1)

stop_ind4 = 3000
ax = visualize_path_frames_spatial_top(
    pd=pff4_P[:, :stop_ind4],
    erf=pff4_ERF[:, :, :stop_ind4],
    scale=0.15,
    ax=ax,
    secondary=True,
    interval=20,
)
ax.plot(pfs4_P[0, :], pfs4_P[1, :], "--", color="darkturquoise", alpha=0.75)
ax.plot(pff4_P[0, :], pff4_P[1, :], color="darkturquoise", alpha=1)

ax.plot(ph_spline_p[0, :], ph_spline_p[1, :], color="black", alpha=1.0, linewidth=4.0)
ax = visualize_path_frames_spatial_top(
    pd=ph_spline_p, erf=ph_spline_erf, scale=0.15, ax=ax, secondary=False, interval=40
)

plt.axis("off")
plt.axis("equal")
plt.tight_layout()
# fig.set_size_inches(9, 3)
# fig.savefig(save_path + "top_col2.pdf", dpi=1800)

# %%
######################### front #########################
fig = plt.figure()
ax = fig.add_subplot(111)

stop_ind1 = 4000
ax = visualize_path_frames_spatial_front(
    pd=pff1_P[:, :stop_ind1],
    erf=pff1_ERF[:, :, :stop_ind1],
    scale=0.15,
    ax=ax,
    secondary=True,
    interval=40,
)
ax.plot(pfs1_P[0, :], pfs1_P[2, :], "--", color="peru", alpha=0.75)
ax.plot(pff1_P[0, :], pff1_P[2, :], color="peru", alpha=1)

stop_ind2 = 4600
ax = visualize_path_frames_spatial_front(
    pd=pff2_P[:, :stop_ind2],
    erf=pff2_ERF[:, :, :stop_ind2],
    scale=0.15,
    ax=ax,
    secondary=True,
    interval=40,
)
ax.plot(pfs2_P[0, :], pfs2_P[2, :], "--", color="orange", alpha=0.75)
ax.plot(pff2_P[0, :], pff2_P[2, :], color="orange", alpha=1)

stop_ind3 = 3500
ax = visualize_path_frames_spatial_front(
    pd=pff3_P[:, :stop_ind3],
    erf=pff3_ERF[:, :, :stop_ind3],
    scale=0.15,
    ax=ax,
    secondary=True,
    interval=40,
)
ax.plot(pfs3_P[0, :], pfs3_P[2, :], "--", color="blueviolet", alpha=0.75)
ax.plot(pff3_P[0, :], pff3_P[2, :], color="blueviolet", alpha=1)

stop_ind4 = 3000
ax = visualize_path_frames_spatial_front(
    pd=pff4_P[:, :stop_ind4],
    erf=pff4_ERF[:, :, :stop_ind4],
    scale=0.15,
    ax=ax,
    secondary=True,
    interval=20,
)
ax.plot(pfs4_P[0, :], pfs4_P[2, :], "--", color="darkturquoise", alpha=0.75)
ax.plot(pff4_P[0, :], pff4_P[2, :], color="darkturquoise", alpha=1)

ax.plot(ph_spline_p[0, :], ph_spline_p[2, :], color="black", alpha=1.0, linewidth=4.0)
ax = visualize_path_frames_spatial_front(
    pd=ph_spline_p, erf=ph_spline_erf, scale=0.15, ax=ax, secondary=False, interval=40
)

plt.axis("off")
plt.axis("equal")
plt.tight_layout()
# fig.set_size_inches(9, 3)
# fig.savefig(save_path + "front_col2.pdf", dpi=1800)

# %%
######################### side #########################
fig = plt.figure()
ax = fig.add_subplot(111)

stop_ind1 = 4000
ax = visualize_path_frames_spatial_side(
    pd=pff1_P[:, :stop_ind1],
    erf=pff1_ERF[:, :, :stop_ind1],
    scale=0.15,
    ax=ax,
    secondary=True,
    interval=40,
)
ax.plot(pfs1_P[1, :], pfs1_P[2, :], "--", color="peru", alpha=0.75)
ax.plot(pff1_P[1, :], pff1_P[2, :], color="peru", alpha=1)

stop_ind2 = 4600
ax = visualize_path_frames_spatial_side(
    pd=pff2_P[:, :stop_ind2],
    erf=pff2_ERF[:, :, :stop_ind2],
    scale=0.15,
    ax=ax,
    secondary=True,
    interval=40,
)
ax.plot(pfs2_P[1, :], pfs2_P[2, :], "--", color="orange", alpha=0.75)
ax.plot(pff2_P[1, :], pff2_P[2, :], color="orange", alpha=1)

stop_ind3 = 3500
ax = visualize_path_frames_spatial_side(
    pd=pff3_P[:, :stop_ind3],
    erf=pff3_ERF[:, :, :stop_ind3],
    scale=0.15,
    ax=ax,
    secondary=True,
    interval=40,
)
ax.plot(pfs3_P[1, :], pfs3_P[2, :], "--", color="blueviolet", alpha=0.75)
ax.plot(pff3_P[1, :], pff3_P[2, :], color="blueviolet", alpha=1)

stop_ind4 = 3000
ax = visualize_path_frames_spatial_side(
    pd=pff4_P[:, :stop_ind4],
    erf=pff4_ERF[:, :, :stop_ind4],
    scale=0.15,
    ax=ax,
    secondary=True,
    interval=20,
)
ax.plot(pfs4_P[1, :], pfs4_P[2, :], "--", color="darkturquoise", alpha=0.75)
ax.plot(pff4_P[1, :], pff4_P[2, :], color="darkturquoise", alpha=1)

ax.plot(ph_spline_p[1, :], ph_spline_p[2, :], color="black", alpha=1.0, linewidth=4.0)
ax = visualize_path_frames_spatial_side(
    pd=ph_spline_p, erf=ph_spline_erf, scale=0.15, ax=ax, secondary=False, interval=40
)

plt.axis("off")
plt.axis("equal")
plt.tight_layout()
# fig.savefig(save_path + "side_col2.pdf", dpi=1800)

plt.show()
