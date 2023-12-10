import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

from pfdq.utils.quaternion import Quaternion
from pfdq.utils.dual_quaternion import DualQuaternion
from pfdq.utils.quaternion_ph import skew_symmetric

# from pfdq.utils.path_generation import axis_equal


def dq_to_euclidean(dq, dt):
    p = dq.to_pose()[:3]
    q = dq.to_pose()[3:]
    w = np.hstack([dt.r_x, dt.r_y, dt.r_z])
    # v = np.hstack([dt.t_x, dt.t_y, dt.t_z]) - np.cross(w, p)
    v = np.hstack([dt.t_x, dt.t_y, dt.t_z]) - np.cross(p, w)

    return p, v, q, w


def euclidean_to_dq(p, v, q, w):
    dq = DualQuaternion.from_pose_vector(np.hstack([p, q]))
    # dt = DualQuaternion.from_vector(np.hstack([w, 0, v + np.cross(w, p), 0]))
    dt = DualQuaternion.from_vector(np.hstack([w, 0, v + np.cross(p, w), 0]))
    return dq, dt


def dq_to_error(dq, dt, dqd, dtd, theta_dot, pf):
    # dqe = dqd.conjugate() * dq
    # dte = dt - dqe.conjugate() * dtd * dqe.conjugate().inverse()
    dqe = dq * dqd.conjugate()
    if pf:
        dte = dt + theta_dot * dqe * dtd.conjugate() * dqe.conjugate()
    else:
        dte = dt + dqe * dtd.conjugate() * dqe.conjugate()
    return dqe, dte


def dq_from_error(dqe, dte, dqd, dtd, theta_dot, pf):
    # dqe = dqd.conjugate() * dq
    # dte = dt - dqe.conjugate() * dtd * dqe.conjugate().inverse()
    dq = dqe * dqd.conjugate().inverse()
    if pf:
        dt = dte - theta_dot * dqe * dtd.conjugate() * dqe.conjugate()
    else:
        dt = dte - dqe * dtd.conjugate() * dqe.conjugate()

    return dq, dt


def f_kinematic_dq(d_time, dq, dt, a, r):

    # get desired state derivatives
    p, v, q, w = dq_to_euclidean(dq, dt)

    # dq_dot = 1 / 2 * (dq * dt)
    dq_dot = 1 / 2 * (dt * dq)
    dt_dot = DualQuaternion.from_vector(
        # np.hstack([r, 0, a + np.cross(r, p) + np.cross(w, v), 0])
        np.hstack([r, 0, a + np.cross(p, r) + np.cross(v, w), 0])
    )

    # Integration step
    dq_next = dq + dq_dot * d_time
    dt_next = dt + dt_dot * d_time

    # w_next = np.hstack([dt_next.r_x, dt_next.r_y, dt_next.r_z])
    # v_next = np.hstack([dt_next.t_x, dt_next.t_y, dt_next.t_z]) - np.cross(w, p)
    # p_next = p + (v_next - v) * d_time
    # q_next = dq_next.to_pose()[3:]
    # dq_next = DualQuaternion.from_pose_vector(np.hstack([p_next, q_next]))

    return dq_next, dt_next, dt_dot


def f_kinematic_theta(d_time, theta_states, ktheta, g, g_d, delta, delta_dot):

    # get states and inputs
    theta = theta_states[0]
    theta_dot = theta_states[1]
    theta_dotdot = theta_dotdot_feedback_law(
        theta=theta,
        theta_dot=theta_dot,
        ktheta=ktheta,
        g=g,
        g_d=g_d,
        delta=delta,
        delta_dot=delta_dot,
    )

    theta = theta + theta_dot * d_time
    theta_dot = theta_dot + theta_dotdot * d_time

    return np.hstack([theta, theta_dot]), theta_dotdot


# ------------------------------ euclidean ODEs ------------------------------ #
def f_kinematic(d_time, x, u):

    # get states and inputs
    p = x[:3]
    v = x[3:6]
    q = x[6:10]
    w = x[10:13]
    a = u[:3]
    r = u[3:6]

    # ODEs
    # q_swapped = np.hstack([q[3], q[0], q[1], q[2]])
    # q_swapped_dot = np.squeeze(1 / 2 * cs.mtimes(skew_symmetric(w), q_swapped))
    # q_dot = np.hstack(
    #     [q_swapped_dot[1], q_swapped_dot[2], q_swapped_dot[3], q_swapped_dot[0]]
    # )

    p_dot = v
    v_dot = a
    q_dot = 1 / 2 * (Quaternion(q=np.hstack([w, 0])) * Quaternion(q=q)).q
    w_dot = r

    # integrate forward
    p_next = p + p_dot * d_time
    v_next = v + v_dot * d_time
    q_next = q + q_dot * d_time
    w_next = w + w_dot * d_time

    x_next = np.hstack([p_next, v_next, q_next / np.linalg.norm(q_next), w_next])

    return x_next


def f_dynamic(d_time, x, u, m, J, disturbance=None):
    # get current state
    p = x[:3]
    v = x[3:6]
    q = x[6:10]
    w = x[10:13]

    # compute wrench
    tau = J @ u.dq[:3]
    f = m * (u.dq[4:-1] - np.cross(p, np.linalg.inv(J) @ tau))
    body_forces = np.hstack([f, tau])

    # compute long. and ang. accel
    if disturbance is None:
        dist_f = 0
        dist_tau = 0
    else:
        dist_f = disturbance[:3]
        dist_tau = disturbance[3:]

    a = (f + dist_f) / m
    r = np.linalg.inv(J) @ ((tau + dist_tau) - np.cross(w, J @ w))
    body_acc = np.hstack([a, r])

    # forward step
    x0 = np.hstack([p, v, q, w])
    u = np.hstack([a, r])
    x_next = f_kinematic(d_time=d_time, x=x0, u=u)

    return x_next, body_forces, body_acc


# ------------------------------- spatial ODEs ------------------------------- #
def spatial_to_euclidean(x, ph_interp):
    xi = x[0]
    w1 = x[1]
    w2 = x[2]
    xi_dot = x[3]
    w1_dot = x[4]
    w2_dot = x[5]

    gamma = ph_interp["gamma"](xi)
    erf = ph_interp["erf"](xi)
    gamma_d = ph_interp["gamma_d"](xi)
    erf_d = ph_interp["erf_d"](xi)
    dist = np.hstack([0, w1, w2])
    dist_dot = np.hstack([0, w1_dot, w2_dot])

    p = np.squeeze(gamma + erf @ dist)
    v = np.squeeze(xi_dot * (gamma_d + erf_d @ dist) + erf @ dist_dot)

    return p, v


def euclidean_to_spatial(xi, x, ph_interp):

    p = x[:3]
    v = x[3:6]

    # parametric variables
    gamma = ph_interp["gamma"](xi)
    sigma = ph_interp["sigma"](xi)
    X_ph = ph_interp["X"](xi)
    erf = ph_interp["erf"](xi)
    sigma_d = ph_interp["sigma_d"](xi)

    gamma_d = ph_interp["gamma_d"](xi)
    erf_d = ph_interp["erf_d"](xi)
    X_ph_d = ph_interp["X_d"](xi)

    e1, e2, e3 = cs.horzsplit(erf)
    X1, X2, X3 = cs.vertsplit(X_ph)
    e1_d, e2_d, e3_d = cs.horzsplit(erf_d)
    X1_d, X2_d, X3_d = cs.vertsplit(X_ph_d)

    # spatial states
    w1 = cs.dot(e2, p - gamma)
    w2 = cs.dot(e3, p - gamma)

    xi_dot = cs.dot(e1, v) / (sigma - X3 * w1 + X2 * w2)
    w1_dot = cs.dot(e2, v) + xi_dot * X1 * w2
    w2_dot = cs.dot(e3, v) - xi_dot * X1 * w1

    # xi_dotdot = (xi_dot * cs.dot(e1_d, v) + cs.dot(e1, a)) / (
    #     sigma - X3 * w1 + X2 * w2
    # ) - (cs.dot(e1, v)) * (
    #     (xi_dot * (sigma_d - X3_d * w1 + X2_d * w2) - X3 * w1_dot + X2 * w2_dot)
    #     / (sigma - X3 * w1 + X2 * w2) ** 2
    # )

    xi = np.squeeze(xi)
    w1 = np.squeeze(w1)
    w2 = np.squeeze(w2)
    xi_dot = np.squeeze(xi_dot)
    w1_dot = np.squeeze(w1_dot)
    w2_dot = np.squeeze(w2_dot)
    # xi_dotdot = np.squeeze(xi_dotdot)
    x_spatial = np.hstack([xi, w1, w2, xi_dot, w1_dot, w2_dot, x[6:]])

    return x_spatial  # , np.squeeze(xi_dotdot)


def f_kinematic_spatial(d_time, x, u, ph_interp):

    # get states and inputs
    xi = x[0]
    w1 = x[1]
    w2 = x[2]
    xi_dot = x[3]
    w1_dot = x[4]
    w2_dot = x[5]
    q = x[6:10]
    w = x[10:13]
    a = u[:3]
    r = u[3:6]

    # parametric variables
    gamma = ph_interp["gamma"](xi)
    sigma = ph_interp["sigma"](xi)
    X_ph = ph_interp["X"](xi)
    erf = ph_interp["erf"](xi)
    sigma_d = ph_interp["sigma_d"](xi)

    gamma_d = ph_interp["gamma_d"](xi)
    erf_d = ph_interp["erf_d"](xi)
    X_ph_d = ph_interp["X_d"](xi)

    e1, e2, e3 = cs.horzsplit(erf)
    X1, X2, X3 = cs.vertsplit(X_ph)
    e1_d, e2_d, e3_d = cs.horzsplit(erf_d)
    X1_d, X2_d, X3_d = cs.vertsplit(X_ph_d)

    # longitudinal ODEs
    dist = np.hstack([0, w1, w2])
    dist_dot = np.hstack([0, w1_dot, w2_dot])
    v = xi_dot * (gamma_d + erf_d @ dist) + erf @ dist_dot

    xi_dotdot = (xi_dot * cs.dot(e1_d, v) + cs.dot(e1, a)) / (
        sigma - X3 * w1 + X2 * w2
    ) - (cs.dot(e1, v)) * (
        (xi_dot * (sigma_d - X3_d * w1 + X2_d * w2) - X3 * w1_dot + X2 * w2_dot)
        / (sigma - X3 * w1 + X2 * w2) ** 2
    )
    w1_dotdot = (
        xi_dot * cs.dot(e2_d, v)
        + cs.dot(e2, a)
        + xi_dotdot * X1 * w2
        + xi_dot**2 * X1_d * w2
        + xi_dot * X1 * w2_dot
    )
    w2_dotdot = (
        xi_dot * cs.dot(e3_d, v)
        + cs.dot(e3, a)
        - xi_dotdot * X1 * w1
        - xi_dot**2 * X1_d * w1
        - xi_dot * X1 * w1_dot
    )

    # rotational ODEs
    q_swapped = np.hstack([q[3], q[0], q[1], q[2]])
    q_swapped_dot = np.squeeze(1 / 2 * cs.mtimes(skew_symmetric(w), q_swapped))
    q_dot = np.hstack(
        [q_swapped_dot[1], q_swapped_dot[2], q_swapped_dot[3], q_swapped_dot[0]]
    )
    w_dot = r

    # integrate forward
    xi_next = xi + d_time * xi_dot
    w1_next = w1 + d_time * w1_dot
    w2_next = w2 + d_time * w2_dot
    xi_dot_next = xi_dot + d_time * xi_dotdot
    w1_dot_next = w1_dot + d_time * w1_dotdot
    w2_dot_next = w2_dot + d_time * w2_dotdot
    q_next = q + d_time * q_dot
    w_next = w + d_time * w_dot

    x_next = np.hstack(
        [
            np.squeeze(xi_next),
            np.squeeze(w1_next),
            np.squeeze(w2_next),
            np.squeeze(xi_dot_next),
            np.squeeze(w1_dot_next),
            np.squeeze(w2_dot_next),
            np.squeeze(q_next / np.linalg.norm(q_next)),
            np.squeeze(w_next),
        ]
    )

    return x_next, np.squeeze(xi_dotdot)


def f_dynamic_spatial(d_time, x, u, m, J, ph_interp, disturbance=None):
    # get current state
    p, v = spatial_to_euclidean(x, ph_interp)
    q = x[6:10]
    w = x[10:13]

    # compute wrench
    tau = J @ u.dq[:3]
    f = m * (u.dq[4:-1] - np.cross(p, np.linalg.inv(J) @ tau))
    body_forces = np.hstack([f, tau])

    # compute long. and ang. accel
    if disturbance is None:
        dist_f = 0
        dist_tau = 0
    else:
        dist_f = disturbance[:3]
        dist_tau = disturbance[3:]

    a = (f + dist_f) / m
    r = np.linalg.inv(J) @ ((tau + dist_tau) - np.cross(w, J @ w))
    body_acc = np.hstack([a, r])

    # body_forces = np.zeros(6)
    # body_acc = u.copy()

    # forward step
    u = np.hstack([a, r])
    x_next, xi_dotdot = f_kinematic_spatial(
        d_time=d_time, x=x, u=u, ph_interp=ph_interp
    )

    return x_next, body_forces, body_acc, xi_dotdot


# -------------------------------- controller -------------------------------- #
def get_F(dqe, dte, dqd, dtd, J, theta_dot):
    # dq = dqd.conjugate().inverse() * dqe
    # dt = dte + dqe.conjugate() * dtd * dqe.conjugate().inverse()
    pf = True
    if theta_dot is None:
        pf = False
    dq, dt = dq_from_error(dqe, dte, dqd, dtd, theta_dot, pf)

    # pos = dq.to_pose()[:3]
    # ang_vel = np.hstack([dt.r_x, dt.r_y, dt.r_z])
    # vel = np.hstack([dt.t_x, dt.t_y, dt.t_z]) - np.cross(ang_vel, pos)
    # a = -np.cross(np.linalg.inv(J) @ w, J @ w)
    # F_dual = np.hstack([np.cross(a, p) + np.cross(w, v), 0])

    p, v, q, w = dq_to_euclidean(dq, dt)
    a = -np.cross(np.linalg.inv(J) @ w, J @ w)

    F_rot = np.hstack([a, 0])
    F_dual = np.hstack([np.cross(p, a) + np.cross(v, w), 0])
    F = DualQuaternion.from_vector(np.concatenate([F_rot, F_dual]))

    return F


def PD_control(dqe, dte, dqd, dtd, dtd_dot, J, kp, kv):  # TODO-->Done

    lambda_switch = 1
    if dqe.r_w < 0:
        print("Switching equilibrium")
        lambda_switch *= -1

    ln_dqe = ln_dq(lambda_switch * dqe)
    if np.isnan(ln_dqe.dq).any():
        print("Nan in ln")
    F = get_F(dqe, dte, dqd, dtd, J, None)
    # dqe_dot = 1 / 2 * (dqe * dte)
    dqe_dot = 1 / 2 * (dte * dqe)

    U = (
        -2 * kp.vector_dot_product(ln_dqe)
        - kv.vector_dot_product(dte)
        - F
        - (
            dqe_dot * dtd.conjugate() * dqe.conjugate()
            + dqe * dtd_dot.conjugate() * dqe.conjugate()
            + dqe * dtd.conjugate() * dqe_dot.conjugate()
        )
    )
    return U


def PD_control_pf(
    dqe,
    dte,
    dqd,
    dtd,
    dtd_D,
    theta_states,
    J,
    kp,
    kv,
    ktheta,
    g,
    g_d,
    delta,
    delta_dot,
):

    # get states and inputs related to virtual to timing law
    theta = theta_states[0]
    theta_dot = theta_states[1]
    theta_dotdot = theta_dotdot_feedback_law(
        theta=theta,
        theta_dot=theta_dot,
        ktheta=ktheta,
        g=g,
        g_d=g_d,
        delta=delta,
        delta_dot=delta_dot,
    )

    # get the logarithm of dqe
    lambda_switch = 1
    if dqe.r_w < 0:
        print("Switching equilibrium")
        lambda_switch *= -1
    ln_dqe = ln_dq(lambda_switch * dqe)
    if np.isnan(ln_dqe.dq).any():
        print("Nan in ln")

    # calculate dqe_dot
    dqe_dot = 1 / 2 * (dte * dqe)

    # calculate error in theta_dot
    # theta_vde = DualQuaternion.from_vector(
    #     np.hstack([theta_dot - theta_vd, np.zeros(7)])
    # )

    # calculate F
    F = get_F(dqe, dte, dqd, dtd, J, theta_dot)

    # calculate nonlinearities
    h = theta_dotdot * (dqe * dtd.conjugate() * dqe.conjugate()) + theta_dot * (
        dqe_dot * dtd.conjugate() * dqe.conjugate()
        + dqe * theta_dot * dtd_D.conjugate() * dqe.conjugate()
        + dqe * dtd.conjugate() * dqe_dot.conjugate()
    )

    # get control law
    U = (
        -2 * kp.vector_dot_product(ln_dqe)
        - kv.vector_dot_product(dte)
        # - ktheta * theta_vde
        - F
        - h
    )

    return U


def theta_dotdot_feedback_law(
    theta,
    theta_dot,
    ktheta,
    g,
    g_d,
    delta,
    delta_dot,
):

    # theta_dotdot = -ktheta * (theta_dot - theta_vd) + theta_dot * theta_vd_D
    g_delta = np.squeeze(g(delta))
    gD_delta = np.squeeze(g_d(delta))
    theta_dotdot = -ktheta * (theta_dot - g_delta) + delta_dot * gD_delta

    return theta_dotdot


def f_error(d_time, dqe, dte, dqd, dtd, dtd_dot, U, J):  # TODO --> Done

    # Calculate state derivatives
    F = get_F(dqe, dte, dqd, dtd, J)

    # dqe_dot = 1 / 2 * (dqe * dte)
    dqe_dot = 1 / 2 * (dte * dqe)
    dte_dot = (
        U
        + F
        + (
            dqe_dot * dtd.conjugate() * dqe.conjugate()
            + dqe * dtd_dot.conjugate() * dqe.conjugate()
            + dqe * dtd.conjugate() * dqe_dot.conjugate()
        )
    )

    # Integration step
    dqe_next = dqe + dqe_dot * d_time
    dte_next = dte + dte_dot * d_time

    dqe_next.normalize()

    return dqe_next, dte_next


def ln_dq(dq, no_switch=False):
    """Calculates the logarithmic of the dual quaternion dq."""
    # dq.normalize()  # to avoid nan-s
    # if no_switch and dq.q_rot.w < 0:  # in standard usage ignore these conditions
    #     dq = -1 * dq
    angle_axis_rot = dq.q_rot.angle_axis()
    angle_rot = angle_axis_rot[3]
    axis_rot = angle_axis_rot[:3]
    # if no_switch and dq.q_rot.w < 0:  # in standard usage ignore these conditions
    #     angle_rot = -1 * angle_rot
    #     axis_rot = -1 * axis_rot

    pos = dq.to_pose()[:3]

    lndq = DualQuaternion.from_vector(
        1 / 2 * np.hstack([angle_rot * axis_rot, 0, pos, 0])
    )

    return lndq


# def get_desired_state(dqd=None, dtd=None, time=0, d_time=0.001, ph_time_interp=None):
#     # TODO: When using this to create the dual quaternions with an integration,
#     # attention must be paid: Either use euler with very small step size or RK4
#     # (whose coding is not trivial) --> make sure d_time is small enough or the
#     # computed dual quaternion will "drift" from the real ph spline

#     a = np.squeeze(ph_time_interp["a"](time))
#     r = np.squeeze(ph_time_interp["r"](time))
#     dqd, dtd, dtd_dot = f_kinematic_dq(d_time, dqd, dtd, a, r)

#     dqd.normalize()

#     return dqd, dtd, dtd_dot


def get_desired_state_variable(
    dqd=None, dtd=None, variable=0, d_variable=0.001, ph_variable_interp=None
):
    # ----------------------------- with integration ----------------------------- #
    # TODO: When using this to create the dual quaternions with an integration,
    # attention must be paid: Either use euler with very small step size or RK4
    # (whose coding is not trivial) --> make sure d_time is small enough or the
    # computed dual quaternion will "drift" from the real ph spline

    a = np.squeeze(ph_variable_interp["a"](variable))
    r = np.squeeze(ph_variable_interp["r"](variable))
    dqd, dtd, dtd_D = f_kinematic_dq(d_variable, dqd, dtd, a, r)

    # ---------------------------- with interpolation ---------------------------- #
    # variable = 0.5
    # p = np.squeeze(ph_variable_interp["p"](variable))
    # v = np.squeeze(ph_variable_interp["v"](variable))
    # a = np.squeeze(ph_variable_interp["a"](variable))
    # q = np.squeeze(ph_variable_interp["q"](variable))
    # w = np.squeeze(ph_variable_interp["w"](variable))
    # r = np.squeeze(ph_variable_interp["r"](variable))
    # dqd, dtd = euclidean_to_dq(p, v, q, w)
    # dtd_D = DualQuaternion.from_vector(
    #     np.hstack([r, 0, a + np.cross(p, r) + np.cross(v, w), 0])
    # )

    # dtd = DualQuaternion.from_vector(np.zeros(8))
    # dtd_D = DualQuaternion.from_vector(np.zeros(8))

    # p = np.array([1, 1, 0])
    # v = np.array([0, 0, 0])
    # a = np.array([0, 0, 0])
    # q = np.array([0, 0, 0, 1])
    # w = np.array([0, 0, 0])
    # r = np.array([0, 0, 0])
    # dqd, dtd = euclidean_to_dq(p, v, q, w)
    # dtd_D = DualQuaternion.from_vector(
    #     np.hstack([r, 0, a + np.cross(p, r) + np.cross(v, w), 0])
    # )

    dqd.normalize()

    return dqd, dtd, dtd_D


# def get_desired_velocity_profile(theta, case):
#     if case == "constant":
#         theta_vd = 0.05
#         theta_vd_D = 0
#     elif case == "sin":
#         amplitud = 0.02
#         frequency = 2
#         offset = 0.05
#         theta_vd = amplitud * np.sin(2 * np.pi * frequency * theta) + offset
#         theta_vd_D = amplitud * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * theta)

#     return theta_vd, theta_vd_D


def get_desired_velocity_profile(profile, visualize=False):
    theta_dot_max = 0.075
    theta_dot_min = 0.01
    delta_min = 0.01**2
    if profile == "p":  # progress
        delta_max = 0.4**2
    elif profile == "m":  # medium
        delta_max = 0.2**2
    elif profile == "c":  # conserv
        delta_max = 0.05**2

    delta = cs.SX.sym("delta")
    if profile == "p" or profile == "m" or profile == "c":
        slope = (theta_dot_min - theta_dot_max) / (delta_max - delta_min)
        offset = theta_dot_min - slope * delta_max
        theta_dot = slope * delta + offset

        theta_dot = cs.if_else(delta >= delta_max, theta_dot_min, theta_dot)
        theta_dot = cs.if_else(delta <= delta_min, theta_dot_max, theta_dot)

    elif profile == "f":
        theta_dot = theta_dot_max

    elif profile == "s":
        theta_dot = theta_dot_max / 4

    elif profile == "v":
        offset = theta_dot_max / 2
        amplitude = theta_dot_max / 4
        frequency = 4
        theta_dot = offset + amplitude * cs.sin(frequency * 2 * np.pi * delta)

    g = cs.Function("g", [delta], [theta_dot])
    g_d = cs.Function("g", [delta], [cs.jacobian(g(delta), delta)])

    if visualize:
        deltas = np.linspace(0, 2 * delta_max, 100)
        theta_dots = np.squeeze([g(dd) for dd in deltas])
        g_ds = np.squeeze([g_d(dd) for dd in deltas])
        plt.subplot(211)
        plt.plot(deltas, theta_dots)
        plt.subplot(212)
        plt.plot(deltas, g_ds)

    return g, g_d


def get_spatial_path(visualize=False):
    x = np.array([0, 1, 0, 1, 0, -1, 0, -1, 0])
    y = np.array([-3, -2, 0, 1, 1, 1, 0, -2, -3])
    z = np.array([2, 2, 3, 5, 7, 5, 3, 2, 2])

    # x = np.array([0, 1, 1, 1, 0, -1, -1, -1, 0])
    # y = np.array([-3, -2, 0, 2, 3, 2, 0, -2, -3])
    # z = np.array([1.9, 2, 3, 5, 7, 5, 3, 2, 1.9])

    pts = np.vstack([x, y, z])
    progress = np.hstack([0, np.cumsum(np.linalg.norm(np.diff(pts), axis=0))])
    f_x = cs.interpolant("f_x", "bspline", [progress / progress[-1]], x)
    f_y = cs.interpolant("f_y", "bspline", [progress / progress[-1]], y)
    f_z = cs.interpolant("f_z", "bspline", [progress / progress[-1]], z)
    xi = cs.MX.sym("xi")
    r = cs.vertcat(f_x(xi), f_y(xi), f_z(xi))
    f = cs.Function("f", [xi], [r])
    if visualize:
        progress_eval = np.linspace(0, 1, 100)
        pts_eval = np.squeeze([f(dd) for dd in progress_eval])
        ax = plt.figure().add_subplot(111, projection="3d")
        ax.plot(pts_eval[:, 0], pts_eval[:, 1], pts_eval[:, 2])
        plt.plot(x, y, z, ".r")
        # axis_equal(pts_eval[:, 0], pts_eval[:, 1], pts_eval[:, 2], ax=ax)

    return xi, r


#%%

# import numpy as np
# import casadi as cs

# import matplotlib.pyplot as plt

# theta_dot_max = 0.075
# theta_dot_min = 0.01
# delta_min = 0.01**2
# delta_max = 0.2**2

# delta_diff = delta_max - delta_min
# theta_dot_diff = theta_dot_max - theta_dot_min

# p1 = np.array([])
# # knots
# x = np.array(
#     [delta_min, delta_min + delta_diff / 4, delta_min + delta_diff * 3 / 4, delta_max]
# )
# y = np.array(
#     [
#         theta_dot_max,
#         theta_dot_min + theta_dot_diff * 3 / 4,
#         theta_dot_min + theta_dot_diff / 4,
#         theta_dot_min,
#     ]
# )

# # append start and end
# x = np.hstack(
#     [
#         np.linspace(0, delta_min * 0.9, 10),
#         x,
#         np.linspace(delta_max * 1.1, 2 * delta_max, 10),
#     ]
# )
# y = np.hstack([np.ones(10) * theta_dot_max, y, np.ones(10) * theta_dot_min])
# f = cs.interpolant("f", "bspline", [x], y)

# # visualize
# delta_eval = np.linspace(0, delta_max * 1.5, 100)
# f_eval = np.squeeze([f(dd) for dd in delta_eval])
# plt.plot(delta_eval, f_eval)
# plt.plot(x, y, "or")
