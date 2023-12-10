# %%

import numpy as np
import matplotlib.pyplot as plt

from pfdq.utils.dual_quaternion import DualQuaternion

"""
Replicates the results from 'Unit dual quaternion-based feedback linearization
tracking problem for attitude and position dynamics' Wang et al., 2012
"""


# ----------------------------- function library ----------------------------- #
def get_F(dqe, dte, dqd, dtd):
    dq = dqd.conjugate().inverse() * dqe
    dt = dte + dqe.conjugate() * dtd * dqe.conjugate().inverse()

    pos = dq.to_pose()[:3]
    ang_vel = np.hstack([dt.r_x, dt.r_y, dt.r_z])
    vel = np.hstack([dt.t_x, dt.t_y, dt.t_z]) - np.cross(ang_vel, pos)
    a = -np.cross(np.linalg.inv(J) @ ang_vel, J @ ang_vel)

    F_rot = np.hstack([a, 0])
    F_dual = np.hstack([np.cross(a, pos) + np.cross(ang_vel, vel), 0])
    F = DualQuaternion.from_vector(np.concatenate([F_rot, F_dual]))

    return F


def ln_dq(dq):
    """Calculates the logarithmic of the dual quaternion dq."""
    angle_axis_rot = dq.q_rot.angle_axis()
    angle_rot = angle_axis_rot[3]
    axis_rot = angle_axis_rot[:3]

    pos = dq.to_pose()[:3]

    lndq = DualQuaternion.from_vector(
        1 / 2 * np.hstack([angle_rot * axis_rot, 0, pos, 0])
    )

    return lndq


def f_RK4(d_time, dqe, dte, dqd, dtd, dtd_dot, U):
    # Calculate derivatives
    F = get_F(dqe, dte, dqd, dtd)

    dqe_dot = 1 / 2 * (dqe * dte)
    dte_dot = (
        U
        + F
        - (
            dqe_dot.conjugate() * dtd * dqe
            + dqe.conjugate() * dtd_dot * dqe
            + dqe.conjugate() * dtd * dqe_dot
        )
    )

    # Integration step
    dqe_next = dqe + dqe_dot * d_time
    dte_next = dte + dte_dot * d_time

    return dqe_next, dte_next


def PD_control(dqe, dte, dqd, dtd, dtd_dot):
    lambda_switch = 1
    if dqe.r_w < 0:
        lambda_switch *= -1

    ln_dqe = ln_dq(lambda_switch * dqe)
    F = get_F(dqe, dte, dqd, dtd)
    dqe_dot = 1 / 2 * (dqe * dte)

    U = (
        -2 * kp.vector_dot_product(ln_dqe)
        - kv.vector_dot_product(dte)
        - F
        + (
            dqe_dot.conjugate() * dtd * dqe
            + dqe.conjugate() * dtd_dot * dqe
            + dqe.conjugate() * dtd * dqe_dot
        )
    )
    return U


def get_desired_state(dqd=None, dtd=None, time=0, case="r", d_time=0.001):
    if case == "r":
        pd = np.array([0, 0, 0])
        qd = np.array([0, 0, 0, 1])
        td = np.zeros(8)

        dqd = DualQuaternion.from_pose(
            pd[0],
            pd[1],
            pd[2],
            qd[0],
            qd[1],
            qd[2],
            qd[3],
        )
        dtd = DualQuaternion.from_vector(td)
        dtd_dot = DualQuaternion.from_vector(np.zeros(8))

    elif case == "t":
        if dqd == None and dtd == None:
            dqd = DualQuaternion.from_vector(np.array([0, 0, 0, 1, 0, 0, 0, 0]))
            dtd = DualQuaternion.from_vector(np.zeros(8))
        a_d = 0.1 * np.pi * np.sin(0.1 * np.pi * time) * np.ones(3)
        r_d = 0.2 * np.pi * np.sin(0.2 * np.pi * time) * np.ones(3)

        # get desired state derivatives
        p_d = dqd.to_pose()[:3]
        w_d = np.hstack([dtd.r_x, dtd.r_y, dtd.r_z])
        v_d = np.hstack([dtd.t_x, dtd.t_y, dtd.t_z]) - np.cross(w_d, p_d)

        dqd_dot = 1 / 2 * (dqd * dtd)
        dtd_dot = DualQuaternion.from_vector(
            np.hstack([r_d, 0, a_d + np.cross(r_d, p_d) + np.cross(w_d, v_d), 0])
        )

        # Integration step
        dqd = dqd + dqd_dot * d_time
        dtd = dtd + dtd_dot * d_time

    return dqd, dtd, dtd_dot


# -------------------------------- user input -------------------------------- #
case = "t"  # Options: "r" (regulation) or "t"  (tracking)
m = 100
J = np.array([[1, 0, 0], [0, 0.63, 0], [0, 0, 0.85]])

kp = DualQuaternion.from_vector([1, 1, 1, 0, 1, 1, 1, 0])
kv = DualQuaternion.from_vector([1, 1, 1, 0, 1, 1, 1, 0])

# ------------------------- initial and desired state ------------------------ #
# initial state
T0 = 0
p0 = np.array([2, 2, 1])
theta = 3.8134
n = np.array([0.4896, 0.2032, 0.8480])
t0 = np.zeros(8)
q0 = np.hstack([np.sin(theta / 2) * np.array(n), np.cos(theta / 2)])

dq0 = DualQuaternion.from_pose(
    p0[0],
    p0[1],
    p0[2],
    q0[0],
    q0[1],
    q0[2],
    q0[3],
)
dt0 = DualQuaternion.from_vector(t0)

# desired state
dqd, dtd, dtd_dot = get_desired_state(case=case)

# convert to error state
dqe0 = dqd.conjugate() * dq0
dte0 = dt0 - dqe0.conjugate() * dtd * dqe0.conjugate().inverse()

# -------------------------------- simulation -------------------------------- #
d_time = 0.01

T = [T0]
DQ = [(dqd.conjugate().inverse() * dqe0).to_pose()]
DQD = [dqd.to_pose()]
DTD = [dtd.dq]
DQE = [dqe0.to_pose()]
DTE = [dte0.dq]
DU = []
while T[-1] < 15:
    # get desired state
    dqd, dtd, dtd_dot = get_desired_state(dqd=dqd, dtd=dtd, time=T[-1], case=case)

    # compute control law
    U = PD_control(dqe0, dte0, dqd, dtd, dtd_dot)

    # integrate forward
    dqe_next, dte_next = f_RK4(d_time, dqe0, dte0, dqd, dtd, dtd_dot, U)

    # store results
    DQ += [(dqd.conjugate().inverse() * dqe_next).to_pose()]
    DQD += [dqd.to_pose()]
    DTD += [dtd.dq]
    DQE += [dqe_next.to_pose()]
    DTE += [dte_next.dq]
    DU += [U.dq]
    T += [T[-1] + d_time]

    # reset initial state
    dqe0 = dqe_next.copy()
    dte0 = dte_next.copy()

    # print(T[-1])

DQ = np.squeeze(DQ)
DQE = np.squeeze(DQE)
DTE = np.squeeze(DTE)
DQD = np.squeeze(DQD)
DTD = np.squeeze(DTD)
DU = np.squeeze(DU)
T = np.squeeze(T)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.figure()
plt.subplot(221)
plt.plot(T, DQE[:, 3:])
plt.ylabel(r"$\hat{q}_e\, [q_e]$")
# plt.xlabel('t [s]')
plt.subplot(223)
plt.plot(T, DQE[:, :3])
plt.ylabel(r"$\hat{q}_e\,[p_e]$")
plt.subplot(222)
for k in range(4):
    plt.plot(T, DQ[:, 3 + k], color=colors[k])
    plt.plot(T, DQD[:, 3 + k], "--", color=colors[k])
plt.ylabel(r"$\hat{q}\, [q]$")
# plt.xlabel('t [s]')
plt.subplot(224)
for k in range(3):
    plt.plot(T, DQ[:, k], color=colors[k])
    plt.plot(T, DQD[:, k], "--", color=colors[k])
plt.ylabel(r"$\hat{q}\,[p]$")
plt.xlabel("t [s]")

plt.show()
