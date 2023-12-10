# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pfdq.utils.quaternion_ph import rotation_to_quaternion
from pfdq.results.utils import visualize_path_frames
from pfdq.utils.pfdq import get_desired_velocity_profile

# -------------------------------- import data ------------------------------- #
data_path = "/home/jonarriza96/pfdq/pfdq/results/data/case_study1/"
save_path = "/home/jonarriza96/pfdq/pfdq/results/figures/case_study1"

# load pf "conservative, medium, progressive"
file_name = "pf_c"
pf_path = data_path + file_name + ".pickle"
with open(pf_path, "rb") as handle:
    dataset = pickle.load(handle)
thc = dataset["X"][:, -2]
thDc = dataset["X"][:, -1]
Qc = dataset["X"][:, 6:10]
pfc_P = dataset["P"]
pfc_ERF = dataset["ERF"]
pfc_Pd = dataset["PD"]
xi_c = dataset["SP"][:, 0]
dperp_c = np.linalg.norm(dataset["SP"][:, 1:3], axis=1) ** 2

file_name = "pf_m"
pf_path = data_path + file_name + ".pickle"
with open(pf_path, "rb") as handle:
    dataset = pickle.load(handle)
thm = dataset["X"][:, -2]
thDm = dataset["X"][:, -1]
Qm = dataset["X"][:, 6:10]
pfm_P = dataset["P"]
pfm_Pd = dataset["PD"]
pfm_ERF = dataset["ERF"]
xi_m = dataset["SP"][:, 0]
dperp_m = np.linalg.norm(dataset["SP"][:, 1:3], axis=1) ** 2

file_name = "pf_p"
pf_path = data_path + file_name + ".pickle"
with open(pf_path, "rb") as handle:
    dataset = pickle.load(handle)
thp = dataset["X"][:, -2]
thDp = dataset["X"][:, -1]
Qp = dataset["X"][:, 6:10]
pfp_P = dataset["P"]
pfp_Pd = dataset["PD"]
pfp_ERF = dataset["ERF"]
xi_p = dataset["SP"][:, 0]
dperp_p = np.linalg.norm(dataset["SP"][:, 1:3], axis=1) ** 2
Qdt = dataset["QD"].T

# load pt
file_name = "pt"
pt_path = data_path + file_name + ".pickle"
with open(pt_path, "rb") as handle:
    dataset = pickle.load(handle)
pt_P = dataset["P"]
pt_ERF = dataset["ERF"]
pt_Pd = dataset["PD"]
pt_ERFd = dataset["ERFD"]
Qt = dataset["X"][:, 6:10]
xi_t = dataset["SP"][:, 0]
dperp_t = np.linalg.norm(dataset["SP"][:, 1:3], axis=1) ** 2
ph_spline = dataset["ph_spline"]
L = np.cumsum(np.linalg.norm(np.diff(ph_spline["gamma"]), axis=0))[-1]

# obtain quaternion of ph spline
ph_spline_q = []
for k in range(ph_spline["erf"].shape[2]):
    erf = ph_spline["erf"][:, :, k]
    ph_spline_q += [rotation_to_quaternion(erf, swap=True)]
ph_spline_q = np.squeeze(ph_spline_q)


# %%
# --------------------------------- visualize -------------------------------- #
erf_scale = 0.05
# figure top
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(pt_Pd[0, :], pt_Pd[1, :], color="black", linewidth=0.25)
ax.plot(pt_P[0, :], pt_P[1, :], color="magenta", alpha=0.75)  # , linewidth=0.9)
ax.plot(pfp_P[0, :], pfp_P[1, :], color="red", alpha=0.75)
ax.plot(pfm_P[0, :], pfm_P[1, :], color="orange", alpha=0.85)
ax.plot(pfc_P[0, :], pfc_P[1, :], color="blue", alpha=0.75)
visualize_path_frames(pt_Pd, pt_ERFd, erf_scale, ax, secondary=True, hide_e1=False)
ax.axis("equal")
plt.axis("off")
plt.tight_layout()
fig.set_size_inches(9, 3)
# fig.savefig(save_path + 'top_view.pdf',dpi=1800)

# %%
# figure zoom
fig = plt.figure()
# fig.set_size_inches(6, 3.5)
ax = fig.add_subplot(111)

visualize_path_frames(pt_P, pt_ERF, erf_scale, ax, secondary=False)
visualize_path_frames(pfp_P, pfp_ERF, erf_scale, ax, secondary=False)
visualize_path_frames(pfm_P, pfm_ERF, erf_scale, ax, secondary=False)
visualize_path_frames(pfc_P, pfc_ERF, erf_scale, ax, secondary=False)

# ax.plot(pt_Pd[0, :], pt_Pd[1, :], "--", color="black")
ax.plot(pt_Pd[0, :], pt_Pd[1, :], "-", color="black", linewidth=0.25)  # , alpha=0.75)
visualize_path_frames(
    pt_Pd, pt_ERFd, erf_scale, ax, secondary=True, hide_e1=False, alpha=0.6
)
ax.plot(pt_P[0, :], pt_P[1, :], color="magenta", alpha=0.75)  # , linewidth=0.9)
ax.plot(pfp_P[0, :], pfp_P[1, :], color="red", alpha=0.75)
ax.plot(pfm_P[0, :], pfm_P[1, :], color="orange", alpha=0.85)
ax.plot(pfc_P[0, :], pfc_P[1, :], color="blue", alpha=0.75)
ax.axis("equal")
ax.set_xlim([3.2, 3.25])
plt.tight_layout()
plt.axis("off")
# fig.savefig(save_path + 'zoom.pdf',dpi=1800)

# %%
# figure data
# fig = plt.figure()

fig, ax = plt.subplots(nrows=3, sharex=True)
plt.setp(ax, xticks=[0, 2, 4, 6, 8], xlim=[-0.1, 8])  # , xticklabels=['a', 'b', 'c']
for AX in ax:
    AX.grid(axis="x")

ax1, ax2, ax3 = ax


# ax = fig.add_subplot(311)
lw = 1.2
ax2.plot(L * xi_t, dperp_t, color="magenta", alpha=0.75, linewidth=lw)  # , linewidth=1)
ax2.plot(L * xi_p, dperp_p, color="red", alpha=0.75, linewidth=lw)  # thp[1:]
ax2.plot(L * xi_m, dperp_m, color="orange", alpha=0.85, linewidth=lw)  # thm[1:]
ax2.plot(L * xi_c, dperp_c, color="blue", alpha=0.75, linewidth=lw)  # thc[1:]
ax2.tick_params(axis="x", which="major", labelsize=14, length=0)
ax2.tick_params(axis="y", which="major", labelsize=14)  # , length = 0)
ax2.set_yticks([0, 0.35])
ax2.set_ylabel(r"$d_{e\perp}$[m]", fontsize=16, labelpad=-16)

# ax = fig.add_subplot(312)
ax3.plot(L * ph_spline["xi"], ph_spline_q[:, 2:], "--", linewidth=1.25, color="black")
ax3.plot(
    L * xi_t, Qt[1:, 2:], color="magenta", alpha=0.75, linewidth=lw
)  # , linewidth=0.9)
ax3.plot(L * xi_p, Qp[1:, 2:], color="red", alpha=0.75, linewidth=lw)  # thp
ax3.plot(L * xi_m, Qm[1:, 2:], color="orange", alpha=0.85, linewidth=lw)  # thm
ax3.plot(L * xi_c, Qc[1:, 2:], color="blue", alpha=0.75, linewidth=lw)  # thc
ax3.tick_params(axis="x", which="major", labelsize=14, length=0)
ax3.tick_params(axis="y", which="major", labelsize=14)  # , length = 1)
ax3.set_ylabel(r"$q$", fontsize=16, labelpad=-18)
ax3.set_yticks([-0.5, 1])
ax3.set_xlabel(r"L [m]", fontsize=16, labelpad=0)

# ax = fig.add_subplot(313)
ax1.plot(L * xi_p, thDp[1:], color="red", alpha=0.75, linewidth=lw)  # thp
ax1.plot(L * xi_m, thDm[1:], color="orange", alpha=0.85, linewidth=lw)  # thm
ax1.plot(L * xi_c, thDc[1:], color="blue", alpha=0.75, linewidth=lw)  # thc
ax1.set_ylabel(r"$\dot{\theta}$", fontsize=16, labelpad=-30)
ax1.set_yticks([0.01, 0.075])
ax1.set_yticklabels(["0.01", "0.075"])
ax1.tick_params(axis="x", which="major", labelsize=14, length=0)
ax1.tick_params(axis="y", which="major", labelsize=14)  # , length = 0)
# ax1.grid(axis='y')
plt.tight_layout()

fig.set_size_inches(3.3, 6.3)
fig.subplots_adjust(hspace=0.2, wspace=0)
# fig.savefig(save_path + 'data.pdf',dpi=1800)

# %%
# figure velocity profiles
gp, _ = get_desired_velocity_profile(profile="p")
gm, _ = get_desired_velocity_profile(profile="m")
gc, _ = get_desired_velocity_profile(profile="c")


deltas = np.linspace(0, 2 * (0.4**2), 10000)
theta_dots_p = np.squeeze([gp(dd) for dd in deltas])
theta_dots_m = np.squeeze([gm(dd) for dd in deltas])
theta_dots_c = np.squeeze([gc(dd) for dd in deltas])

fig = plt.figure()
ax = fig.add_subplot(111)
# ax.plot([deltas[0], deltas[-1]],[min(theta_dots_c),min(theta_dots_c)],"--k",alpha=0.5)
# ax.plot([deltas[0], deltas[-1]],[max(theta_dots_c),max(theta_dots_c)],"--k",alpha=0.5)
ax.plot(deltas, theta_dots_p, color="red", alpha=0.75)
ax.plot(deltas, theta_dots_m, color="orange", alpha=0.85)
ax.plot(deltas, theta_dots_c, color="blue", alpha=0.75)
ax.set_xscale("log")
ax.set_xlabel(r"$||d_{e\perp}||^2$", fontsize=16, labelpad=0)
ax.set_ylabel(r"$\theta_{vd}$", fontsize=16, labelpad=-30)
ax.set_xlim([deltas[1], deltas[-1]])
ax.set_yticks([0.01, 0.075])
ax.set_yticklabels(["0.01", "0.075"])
ax.grid(axis="y")
plt.tight_layout()
fig.set_size_inches(3, 3)
ax.tick_params(axis="both", which="major", labelsize=14)  # , length = 0)

# fig.savefig(save_path + 'v_profiles.pdf',dpi=1800)
plt.show()
