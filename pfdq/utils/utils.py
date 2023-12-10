import numpy as np
from pfdq.utils.quaternion import Quaternion
from pfdq.utils.dual_quaternion import DualQuaternion


def RK4(x, u, dt, f):
    k1 = f(x, u)
    k2 = f(x+dt/2*k1, u)
    k3 = f(x+dt/2*k2, u)
    k4 = f(x+dt*k3, u)
    x_next = x + dt/6*(k1+2*k2+2*k3+k4)

    return x_next


def f_rk4(x, u, dt):
    """ 2D model for omnidirectional robot: xdot = f(x,u), where x=[x,y,theta] 
    and u = [vx,vy,omega]. The ODEs are integrated with a RK4 routine. """

    def f_ode(x, u):
        x_dot = u
        return x_dot

    x_next = RK4(x, u, dt, f_ode)

    return x_next


def x_to_dq(x):
    """ Converts x to a dual quaternion. """

    # px = x[0]
    # py = x[1]
    # theta = x[2]
    # qrw = np.cos(theta/2)
    # qrx = 0
    # qry = 0
    # qrz = np.sin(theta/2)
    # qtw = 0
    # qtx = (px*np.cos(theta/2) + py*np.sin(theta/2))  # 1/2*
    # qty = (py*np.cos(theta/2) - px*np.sin(theta/2))  # 1/2*
    # qtz = 0
    # dq = DualQuaternion.from_vector([qrx, qry, qrz, qrw, qtx, qty, qtz, qtw])

    px = x[0]
    py = x[1]
    pz = 0
    theta = x[2]
    dq = DualQuaternion.from_pose_vector([px, py, pz,
                                          0, 0, np.sin(theta/2), np.cos(theta/2)])
    return dq


def log_x(x):
    return np.array([0, 0, x[2]/2, x[0]/2, x[1]/2, 0])


def log_dq(dq):
    """ Calculates the logarithmic of the dual quaternion dq. """
    angle_axis_rot = dq.q_rot.angle_axis()
    angle_rot = angle_axis_rot[3]
    axis_rot = angle_axis_rot[:3]

    pos = dq.to_pose()[:3]  # dq.q_dual.angle_axis()
    # angle_axis_dual = dq.q_dual.angle_axis()
    # angle_dual = angle_axis_dual[3]
    # axis_dual = angle_axis_dual[:3]

    return np.hstack([1/2*angle_rot*axis_rot, 1/2*pos])
    # return np.hstack([1/2*angle_rot*axis_rot, 1/2*angle_dual*axis_dual])


def ws_to_u(x, ws):
    # ws = ws[1:-1]
    return np.array([ws[3] - x[1]*ws[2], ws[4] + x[0]*ws[2], ws[2]])


def trajectory_reference(t, name='circle'):
    if name == 'circle':
        x_ref = np.array([5*np.cos(t), 5*np.sin(t), t-np.pi])

        wz = 1
        vx = -5*np.sin(t)
        vy = 5*np.cos(t)

    elif name == 'line':
        x_ref = np.array([t, t, 3/4*np.pi])

        wz = 0
        vx = 1
        vy = 1

    w_ref = DualQuaternion.from_vector(
        [0, 0, wz, 0, vx+x_ref[1]*wz, vy-x_ref[0]*wz, 0, 0])

    return x_ref, w_ref
