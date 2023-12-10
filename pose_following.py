import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

from tqdm import tqdm

from pfdq.utils.pfdq import *
from pfdq.utils.path_generation import (
    get_parametric_function,
    convert_path_to_ph,
    evaluate_ph_spline,
    generate_interpolators,
    time_interpolators,
    visualize_path_with_frames,
)
from pfdq.utils.quaternion_ph import quaternion_to_rotation, rotation_to_quaternion
from pfdq.results.utils import save_pickle

# -------------------------------- user input -------------------------------- #
# overall settings
case_study = 2
save = False

# pf settings
pf = True

# case study settings
if case_study == 1:
    pf_profile = "p"  # Options: p (progressive), m (medium), c (conservative)

    start_deviated = False
    disturbe = True

    dist_start = 0.38
    dist_end = 0.43

elif case_study == 2:
    pf_profile = "f"  # Options: f (fast), s (slow), v (varying)
    starting_pt = 5
    start_deviated = False
    disturbe = False

    p0 = [
        np.array([0, -1.5, 2]),
        np.array([1.5, 0, 5.5]),
        np.array([1.5, -3, 1.5]),
        np.array([-2.5, 1, 4.5]),
        np.array([0.1, -3, 3]),
        # np.array([0, 1, 9.5]),
        np.array([-2.5, 1, 4.5]),
    ]
    ypr = [
        np.array([np.pi / 4, -np.pi / 4, 0]),
        np.array([0, np.pi / 3, -np.pi / 3]),
        np.array([-2 * np.pi / 3, 0, np.pi / 4]),
        np.array([np.pi / 3, -np.pi / 3, 0]),
        np.array([np.pi / 4, -np.pi / 4, 0]),
        # np.array([-np.pi / 2, 0, 0]),
        np.array([np.pi / 3, -np.pi / 3, 0]),
    ]
    p0 = p0[starting_pt]
    ypr = ypr[starting_pt]

    R_ypr = Rotation.from_euler("zyx", ypr, degrees=False)

# simulation settings
t_total = 14.445
if case_study == 1:
    d_time = 30 / 10000
elif case_study == 2:
    d_time = 10 / 10000

# rigid body properties
m = 1
J = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) / 100

# controller settings
kp = 3 * DualQuaternion.from_vector([1, 1, 1, 0, 1, 1, 1, 0])
kv = 3 * DualQuaternion.from_vector([1, 1, 1, 0, 1, 1, 1, 0])
ktheta = 1

# ph settings
if case_study == 1:
    n = 4
elif case_study == 2:
    n = 10
n_spline = 1000

# ------------------------------- get ph curve ------------------------------- #
print("Generating PH path ...")
# define path
if case_study == 1:
    xi = cs.SX.sym("xi")
    r = cs.vertcat(6 * xi, 0.5 * cs.sin(15 * xi), 1e-8 * cs.sin(xi))
    # r = cs.vertcat(6 * xi, cs.sin(15 * xi), cs.cos(10 * xi))  # * cs.sin(5 * xi))
    # r = cs.vertcat(6 * xi, cs.sin(15 * xi), cs.cos(15 * xi))  # * cs.sin(5 * xi))
elif case_study == 2:
    xi, r = get_spatial_path(visualize=False)
path = {"xi": xi, "r": r}

# convert path to ph
ph_func = get_parametric_function(n_grid=2**n + 1)  # declare conversion function
ph_path = convert_path_to_ph(path=path, n_segments=2**n)
ph_spline = evaluate_ph_spline(
    xi_eval=np.linspace(1e-8, ph_path["xi_grid"][-1] - 1e-8, n_spline),
    ph_func=ph_func,
    ph_path=ph_path,
    visualize=False,
    planar=False,
)

# generate interpolators from ph
ph_interp = generate_interpolators(ph_spline)
ph_time_interp = time_interpolators(ph_interp, t_total)
# test_interpolators(ph_interp, ph_spline, variable="xi")
# test_interpolators(ph_time_interp, ph_spline, variable="time")
print("Done")

# ---------------------------- pf velocity profile --------------------------- #
g, g_d = get_desired_velocity_profile(profile=pf_profile, visualize=False)

# ------------------------- initial and desired state ------------------------ #
# initial desired state
T0 = ph_time_interp["time"][0] + 1e-6  # t_total / 3  # 1e-6

if case_study == 1:
    theta0 = 1e-6
elif case_study == 2:
    ind_closest = np.argmin(np.linalg.norm(ph_spline["gamma"].T - p0, axis=1))
    theta0 = ph_spline["xi"][ind_closest]
    q0 = np.squeeze(ph_interp["q"](theta0))

    R_rotated = Rotation.from_quat(q0).as_matrix() @ R_ypr.as_matrix()
    q0 = Rotation.from_matrix(R_rotated).as_quat()

if pf:
    pd0 = np.squeeze(ph_interp["p"](theta0))  # np.array([0, 0, 0])
    qd0 = np.squeeze(ph_interp["q"](theta0))
    vd0 = np.squeeze(ph_interp["v"](theta0))
    wd0 = np.squeeze(ph_interp["w"](theta0))
    rd0 = np.squeeze(ph_interp["r"](theta0))
    ad0 = np.squeeze(ph_interp["a"](theta0))

    # theta_vd, theta_vd_D = get_desired_velocity_profile(
    #     theta=np.squeeze(theta0), case=theta_vd_profile
    # )
else:
    pd0 = np.squeeze(ph_time_interp["p"](T0))  # np.array([0, 0, 0])
    qd0 = np.squeeze(ph_time_interp["q"](T0))
    vd0 = np.squeeze(ph_time_interp["v"](T0))
    wd0 = np.squeeze(ph_time_interp["w"](T0))
    rd0 = np.squeeze(ph_time_interp["r"](T0))
    ad0 = np.squeeze(ph_time_interp["a"](T0))


dqd, dtd = euclidean_to_dq(pd0, vd0, qd0, wd0)
dtd_D = DualQuaternion.from_vector(
    np.hstack([rd0, 0, ad0 + np.cross(rd0, pd0) + np.cross(wd0, vd0), 0])
)

# initial system state
if start_deviated:
    erfd0 = np.squeeze(ph_interp["erf"](theta0))
    e1d = erfd0[:, 0]
    e2d = erfd0[:, 1]
    e3d = erfd0[:, 2]

    e10 = (e1d + e2d) / np.linalg.norm((e1d + e2d))
    e30 = e3d.copy()
    e20 = np.cross(e30, e10) / np.linalg.norm(np.cross(e30, e10))
    erf0 = np.vstack([e10, e20, e30]).T

    dev = 2 * 1.2
    p0 = pd0 + e2d * dev
    v0 = np.zeros(3)
    # q0 = qd0.copy()
    q0 = rotation_to_quaternion(erf0, swap=True)  # qd0.copy()
    w0 = np.zeros(3)
    a0 = np.zeros(3)
else:
    if case_study == 1:
        p0 = pd0.copy()
        q0 = qd0.copy()

    v0 = np.zeros(3)  # vd0.copy()
    w0 = np.zeros(3)  # wd0.copy()
    a0 = np.zeros(3)  # ad0.copy()

dq0, dt0 = euclidean_to_dq(p0, v0, q0, w0)
x0 = np.hstack([p0, v0, q0, w0])
# x_sp = euclidean_to_spatial(xi=theta0, x=x0, ph_interp=ph_interp)

if pf:
    x0 = np.hstack([x0, [np.squeeze(theta0), 0]])
    # x0 = np.hstack([x0, [x_sp[0], x_sp[3]]])

# convert to error state
dqe, dte = dq_to_error(dq=dq0, dt=dt0, dqd=dqd, dtd=dtd, theta_dot=x0[-1], pf=pf)

# -------------------------------- simulation -------------------------------- #

# simulation loop
T = [T0]
DQE = [dqe]
DTE = [dte]
DQD = [dqd]
DTD = [dtd]
X = [x0]
U = []
SP = []
if pf:
    TH_VD = [0]

k = 0
time = T0
while True:
    ind_closest = np.argmin(np.linalg.norm(ph_spline["gamma"].T - X[-1][:3], axis=1))
    xi_approx = ph_spline["xi"][ind_closest]
    x_sp = euclidean_to_spatial(xi=xi_approx, x=X[-1], ph_interp=ph_interp)
    SP += [x_sp]

    # compute control law
    if pf:
        if case_study == 1:
            # pd, _, _, _ = dq_to_euclidean(dqd, dtd)
            # p = X[-1][:3]
            # pe_n2 = np.linalg.norm(pd - p) ** 2
            # x_sp = euclidean_to_spatial(xi=X[-1][-2], x=X[-1], ph_interp=ph_interp)

            x_sp = euclidean_to_spatial(xi=X[-1][-2], x=X[-1], ph_interp=ph_interp)
            w1 = x_sp[1]
            w2 = x_sp[2]
            # w1_dot = x_sp[4]
            # w2_dot = x_sp[5]
            # delta = w1**2 + w2**2
            # delta_dot = 2 * w1 * w1_dot + 2 * w2 * w2_dot

            delta = w1**2 + w2**2
            # delta = np.linalg.norm(dqe.to_pose()[:3])  # pe
            delta_dot = 0  # np.linalg.norm((1 / 2 * dte * dqe).dq[4:7])  # pedot

        elif case_study == 2:
            delta = X[-1][-2]  # theta
            delta_dot = X[-1][-1]  # theta_dot

        u = PD_control_pf(
            dqe=dqe,
            dte=dte,
            dqd=dqd,
            dtd=dtd,
            dtd_D=dtd_D,
            theta_states=X[-1][-2:],
            J=J,
            kp=kp,
            kv=kv,
            ktheta=ktheta,
            g=g,
            g_d=g_d,
            delta=delta,
            delta_dot=delta_dot,
        )
    else:
        u = PD_control(dqe, dte, dqd, dtd, dtd_D, J, kp, kv)

    # generate disturbance
    disturbance = None
    if disturbe:
        if pf:
            xi = X[-1][-2]
        else:
            xi = ph_time_interp["xi"](T[-1])
        if dist_start < xi and xi < dist_end:
            # longitudinal disturbance
            erf = np.squeeze(ph_interp["erf"](ph_time_interp["xi"](T[-1])))
            e1 = erf[:, 0]
            e2 = erf[:, 1]
            d_mag = 4
            d_dir = e1 + e2
            dist_f = d_mag * (d_dir) / np.linalg.norm(d_dir)

            # yaw disturbance
            dist_tau = np.array([0, 0, 0.01])
            disturbance = np.hstack([dist_f, dist_tau])

    # integrate forward
    x_next, body_forces, body_acc = f_dynamic(
        d_time=d_time, x=X[-1], u=u, m=m, J=J, disturbance=disturbance
    )
    if pf:
        theta_states_next, theta_dotdot_next = f_kinematic_theta(
            d_time=d_time,
            theta_states=X[-1][-2:],
            ktheta=ktheta,
            g=g,
            g_d=g_d,
            delta=delta,
            delta_dot=delta_dot,
        )

        # ind_closest = np.argmin(
        #     np.linalg.norm(ph_spline["gamma"].T - x_next[:3], axis=1)
        # )
        # xi_approx = ph_spline["xi"][ind_closest]
        # x_sp = euclidean_to_spatial(xi=xi_approx, x=x_next, ph_interp=ph_interp)
        # theta_states_next = np.array([x_sp[0], x_sp[3]])

        # theta = X[-1][-2]
        # theta_dot = X[-1][-1]
        # theta_next = theta + theta_dot * d_time
        # x_sp = euclidean_to_spatial(xi=theta_next, x=x_next, ph_interp=ph_interp)
        # theta_states_next = np.array([x_sp[0], x_sp[3]])

        x_next = np.hstack([x_next, theta_states_next])

    # convert euclidean state to dq error
    dq, dt = euclidean_to_dq(
        p=x_next[:3], v=x_next[3:6], q=x_next[6:10], w=x_next[10:13]
    )
    dqe, dte = dq_to_error(dq=dq, dt=dt, dqd=dqd, dtd=dtd, theta_dot=x_next[-1], pf=pf)

    # get desired state
    if pf:
        theta = X[-1][-2]
        theta_next = x_next[-2]

        dqd, dtd, dtd_D = get_desired_state_variable(
            dqd=dqd,
            dtd=dtd,
            d_variable=theta_next - theta,
            variable=theta,
            ph_variable_interp=ph_interp,
        )

        # theta_vd, theta_vd_D = get_desired_velocity_profile(
        #     theta=theta, case=theta_vd_profile
        # )

    else:
        dqd, dtd, dtd_D = get_desired_state_variable(
            dqd=dqd,
            dtd=dtd,
            d_variable=d_time,
            variable=time,
            ph_variable_interp=ph_time_interp,
        )

    # store results
    DQE += [dqe]
    DTE += [dte]
    DQD += [dqd]
    DTD += [dtd]
    X += [x_next]
    U += [body_forces]
    T += [time + d_time]
    if pf:
        TH_VD += [0]

    if pf and X[-1][-2] > ph_interp["xi"][-1] - 1e-3:  # end of pf
        break
    elif not pf and T[-1] >= t_total:  # end of trackin
        break

    time += d_time
    if pf:
        print("theta:", X[-1][-2])
    else:
        print("time:", T[-1])


# post process data
T = np.squeeze(T)
X = np.squeeze(X)
U = np.squeeze(U)
SP = np.squeeze(SP)

PE = np.zeros((3, T.shape[0]))
QE = np.zeros((4, T.shape[0]))

P = np.zeros((3, T.shape[0]))
V = np.zeros((3, T.shape[0]))
Q = np.zeros((4, T.shape[0]))
W = np.zeros((3, T.shape[0]))
ERF = np.zeros((3, 3, T.shape[0]))

PD = np.zeros((3, T.shape[0]))
VD = np.zeros((3, T.shape[0]))
QD = np.zeros((4, T.shape[0]))
WD = np.zeros((3, T.shape[0]))
ERFD = np.zeros((3, 3, T.shape[0]))

trans_coord = np.zeros((2, T.shape[0]))
for k, (dqe, dte, dqd, dtd) in enumerate(zip(DQE, DTE, DQD, DTD)):
    pe, _, qe, _ = dq_to_euclidean(dqe, dte)
    PE[:, k] = pe
    QE[:, k] = qe

    P[:, k] = X[k, :3]
    V[:, k] = X[k, 3:6]
    Q[:, k] = X[k, 6:10]
    W[:, k] = X[k, 10:13]
    ERF[:, :, k] = quaternion_to_rotation(Q[:, k], swap=True)

    pd, vd, qd, wd = dq_to_euclidean(dqd, dtd)
    PD[:, k] = pd
    QD[:, k] = qd
    VD[:, k] = vd
    WD[:, k] = wd
    ERFD[:, :, k] = quaternion_to_rotation(qd, swap=True)

    x_sp = euclidean_to_spatial(xi=X[k, -2], x=X[k, :], ph_interp=ph_interp)
    trans_coord[:, k] = x_sp[1:3]

# ------------------------- benchmark the simulation -------------------------
pe_n2_t = np.sum(np.linalg.norm(PE, axis=0) * d_time)
qe_n2_1 = np.sum(np.linalg.norm(QE.T - np.array([0, 0, 0, 1]), axis=1) * d_time)
qe_n2_2 = np.sum(np.linalg.norm(QE.T - np.array([0, 0, 0, -1]), axis=1) * d_time)
qe_n2_t = min(qe_n2_1, qe_n2_2)
print("#### Results ####")
print("pe_n2_t:", round(pe_n2_t, 4))
print("qe_n2_t:", round(qe_n2_t, 4), "\n")

# ------------------------------- visualization ------------------------------ #
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
scale = 0.1

if disturbance:
    ind_dist_start = np.argmin(np.abs(X[:, -2] - dist_start))
    ind_dist_end = np.argmin(np.abs(X[:, -2] - dist_end))

POS = P.copy()
ax = plt.figure().add_subplot(131, projection="3d")
ax = visualize_path_with_frames(pd=PD, erf=ERFD, scale=scale, ax=ax, secondary=True)
ax = visualize_path_with_frames(pd=POS, erf=ERF, scale=scale, ax=ax)
ax.plot(POS[0, 0], POS[1, 0], POS[2, 0], "og")
ax.plot(POS[0, -1], POS[1, -1], POS[2, -1], "or")

plt.subplot(432)
plt.plot(T, PE.T)
# plt.plot(T, (POS - PD).T, "--")
plt.ylabel(r"$\hat{q}_e\, [p_e]$")
plt.subplot(435)
plt.plot(T, QE.T)
plt.ylabel(r"$\hat{q}_e\,[q_e]$")
plt.subplot(438)
plt.plot(T[:-1], U[:, :3])
plt.ylabel(r"$f\,[N]$")
plt.subplot(4, 3, 11)
plt.plot(T[:-1], U[:, 3:])
plt.ylabel(r"$\tau\,[Nm]$")
plt.xlabel("t [s]")

plt.subplot(433)
for k in range(3):
    plt.plot(T, P[k, :], color=colors[k])
    plt.plot(T, PD[k, :], "--", color=colors[k])
plt.ylabel(r"$p\,[m]$")
plt.subplot(436)
for k in range(4):
    plt.plot(T, Q[k, :], color=colors[k])
    plt.plot(T, QD[k, :], "--", color=colors[k])
plt.ylabel(r"$q$")
plt.subplot(439)
plt.plot(T, V.T)
plt.ylabel(r"$v\,[m/s]$")
plt.subplot(4, 3, 12)
plt.plot(T, W.T)
plt.ylabel(r"$w\,[rad/s]$")
plt.xlabel("t [s]")

# if pf:
#     TH_VD = np.squeeze(TH_VD)
#     plt.figure()
#     plt.subplot(211)
#     plt.plot(T, X[:, -2])
#     if disturbance:
#         plt.plot([T[ind_dist_start], T[ind_dist_start]], [0, np.max(X[:, -2])], "--k")
#         plt.plot([T[ind_dist_end], T[ind_dist_end]], [0, np.max(X[:, -2])], "--k")
#     plt.ylabel(r"$\theta$")
#     plt.subplot(212)
#     plt.plot(T, X[:, -1])
#     if disturbance:
#         plt.plot([T[ind_dist_start], T[ind_dist_start]], [0, np.max(X[:, -1])], "--k")
#         plt.plot([T[ind_dist_end], T[ind_dist_end]], [0, np.max(X[:, -1])], "--k")
#     # plt.plot(T, TH_VD, "--k")
#     plt.ylabel(r"$\dot{\theta}$")
#     plt.xlabel(r"t [s]")

# plt.figure()
# plt.plot(T, trans_coord.T)
# plt.plot([T[ind_dist_start], T[ind_dist_start]], [0, np.max(trans_coord)], "--k")
# plt.plot([T[ind_dist_end], T[ind_dist_end]], [0, np.max(trans_coord)], "--k")
# plt.ylabel(r"Trans. coord")
# plt.xlabel(r"t [s]")
plt.show()


# ------------------------------- save results ------------------------------- #
data = {
    "T": T,
    "X": X,
    "U": U,
    "SP": SP,
    "P": POS,
    "V": V,
    "Q": Q,
    "W": W,
    "ERF": ERF,
    "PD": PD,
    "VD": VD,
    "QD": QD,
    "WD": WD,
    "ERFD": ERFD,
    "PE": PE,
    "QE": QE,
    "ph_spline": ph_spline,
}
if pf:
    if case_study == 1:
        file_name = "pf_" + pf_profile
    elif case_study == 2:
        # file_name = "pf_" + pf_profile + "_" + str(starting_pt)
        file_name = "pf_i_" + str(starting_pt)
else:
    file_name = "pt"

path = "/home/jonarriza96/pfdq/pfdq/results/data/case_study" + str(case_study)
if save:
    save_pickle(path=path, file_name=file_name, data=data)
