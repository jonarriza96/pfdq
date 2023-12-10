import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

from sympy import var
from sympy.utilities.lambdify import lambdify
from sympy.parsing.mathematica import mathematica

from pyquaternion import Quaternion

from pfdq.utils.quaternion_ph import *
from pfdq.utils.dual_quaternion import DualQuaternion
from pfdq.utils.pfdq import (
    euclidean_to_dq,
    dq_to_euclidean,
    f_kinematic,
    f_kinematic_dq,
)


def Bezier(t, p, d=9):
    bz = p[:, 0]
    if d == 1:
        bz = (1 - t) * p[0, :] + t * p[1, :]
    elif d == 2:
        bz = p[0, :] * (1 - t) ** 2 + 2 * p[1, :] * (1 - t) * t + p[2, :] * t**2
    elif d == 4:
        bz = (
            (1 - t) ** 4 * p[0, :]
            + 4 * (1 - t) ** 3 * t * p[1, :]
            + 6 * (1 - t) ** 2 * t**2 * p[2, :]
            + 4 * (1 - t) * t**3 * p[3, :]
            + t**4 * p[4, :]
        )
    elif d == 5:
        bz = (
            (1 - t) ** 5 * p[0, :]
            + 5 * t * (1 - t) ** 4 * p[1, :]
            + 10 * t**2 * (1 - t) ** 3 * p[2, :]
            + 10 * t**3 * (1 - t) ** 2 * p[3, :]
            + 5 * t**4 * (1 - t) * p[4, :]
            + t**5 * p[5, :]
        )
    elif d == 8:
        bz = (
            p[0, :] * (1 - t) ** 8
            + 8 * p[1, :] * (1 - t) ** 7 * t
            + 28 * p[2, :] * (1 - t) ** 6 * t**2
            + 56 * p[3, :] * (1 - t) ** 5 * t**3
            + 70 * p[4, :] * (1 - t) ** 4 * t**4
            + 56 * p[5, :] * (1 - t) ** 3 * t**5
            + 28 * p[6, :] * (1 - t) ** 2 * t**6
            + 8 * p[7, :] * (1 - t) * t**7
            + p[8, :] * t**8
        )
    elif d == 9:
        bz = (
            (1 - t) ** 9 * p[0, :]
            + 9 * (1 - t) ** 8 * t * p[1, :]
            + 36 * (1 - t) ** 7 * t**2 * p[2, :]
            + 84 * (1 - t) ** 6 * t**3 * p[3, :]
            + 126 * (1 - t) ** 5 * t**4 * p[4, :]
            + 126 * (1 - t) ** 4 * t**5 * p[5, :]
            + 84 * (1 - t) ** 3 * t**6 * p[6, :]
            + 36 * (1 - t) ** 2 * t**7 * p[7, :]
            + 9 * (1 - t) * t**8 * p[8, :]
            + t**9 * p[9, :]
        )
    elif d == 17:
        bz = (
            p[0, :] * (1 - t) ** 17
            + 17 * p[1, :] * (1 - t) ** 16 * t
            + 136 * p[2, :] * (1 - t) ** 15 * t**2
            + 680 * p[3, :] * (1 - t) ** 14 * t**3
            + 2380 * p[4, :] * (1 - t) ** 13 * t**4
            + 6188 * p[5, :] * (1 - t) ** 12 * t**5
            + 12376 * p[6, :] * (1 - t) ** 11 * t**6
            + 19448 * p[7, :] * (1 - t) ** 10 * t**7
            + 24310 * p[8, :] * (1 - t) ** 9 * t**8
            + 24310 * p[9, :] * (1 - t) ** 8 * t**9
            + 19448 * p[10, :] * (1 - t) ** 7 * t**10
            + 12376 * p[11, :] * (1 - t) ** 6 * t**11
            + 6188 * p[12, :] * (1 - t) ** 5 * t**12
            + 2380 * p[13, :] * (1 - t) ** 4 * t**13
            + 680 * p[14, :] * (1 - t) ** 3 * t**14
            + 136 * p[15, :] * (1 - t) ** 2 * t**15
            + 17 * p[16, :] * (1 - t) * t**16
            + p[17, :] * t**17
        )
    return bz


def Z_to_erf(t, Z):

    # ERF
    sigma = Z[0] ** 2 + Z[1] ** 2 + Z[2] ** 2 + Z[3] ** 2
    e1 = (quat_AiconjA(Z, "i") / sigma)[1:]
    e2 = (quat_AiconjA(Z, "j") / sigma)[1:]
    e3 = (quat_AiconjA(Z, "k") / sigma)[1:]
    erf = cs.horzcat(e1, e2, e3)

    # Darboux vector
    e1_d = cs.jacobian(e1, t)  # .T
    e2_d = cs.jacobian(e2, t)  # .T
    e3_d = cs.jacobian(e3, t)  # .T

    X1 = cs.dot(e2_d, e3)
    X2 = cs.dot(e3_d, e1)
    X3 = cs.dot(e1_d, e2)
    # omega =  X1*e1+X2*e2+X3*e3
    X = cs.horzcat(X1, X2, X3)

    return erf, X  # ,omega


def ph_curve(interp_order):
    """Generates symbolic functions for C2 or C4 hermite interpolation of PH curves
        interp_order: Order of hermite interpolation (2 or 4)
    Returns:
        dict: Dictionary with hodograph, curve and PH construction functions
    """
    if interp_order == 2:
        # symbolic variables and constants
        pb = cs.SX.sym("pb", 3)
        pe = cs.SX.sym("pe", 3)
        vb = cs.SX.sym("vb", 3)
        ve = cs.SX.sym("ve", 3)
        ab = cs.SX.sym("ab", 3)
        ae = cs.SX.sym("ae", 3)

        o0 = cs.SX.sym("o0")
        o4 = cs.SX.sym("o4")
        o2 = cs.SX.sym("o2")
        t1 = cs.SX.sym("t1")
        t3 = cs.SX.sym("t3")

        i = cs.DM([0, 1, 0, 0])

        # ------------------- function for C2 hermite interpolation ------------------ #
        # Step 1
        h0 = cs.vertcat(0, vb)
        h8 = cs.vertcat(0, ve)
        h1 = cs.vertcat(0, ab / 8) + h0
        h7 = -(cs.vertcat(0, ae / 8) - h8)

        # Step 2
        h0_norm = cs.norm_2(h0)
        numden = (h0 / h0_norm) + i
        Qo0 = cs.vertcat(cs.cos(o0), cs.sin(o0), 0, 0)
        A0 = quat_mult(cs.sqrt(h0_norm) * (numden / cs.norm_2(numden)), Qo0)

        h8_norm = cs.norm_2(h8)
        numden = (h8 / h8_norm) + i
        Qo4 = cs.vertcat(cs.cos(o4), cs.sin(o4), 0, 0)
        A4 = quat_mult(cs.sqrt(h8_norm) * (numden / cs.norm_2(numden)), Qo4)

        # Step 3
        A1 = quat_mult(quat_mult(-(cs.vertcat(t1, 0, 0, 0) + h1), A0), i) / (
            cs.norm_2(A0) ** 2
        )
        A3 = quat_mult(quat_mult(-(cs.vertcat(t3, 0, 0, 0) + h7), A4), i) / (
            cs.norm_2(A4) ** 2
        )

        # Step 4
        delta_p = cs.vertcat(0, pe - pb)
        delta_v = cs.vertcat(0, ve + vb)
        delta_a = cs.vertcat(0, ae - ab)

        def AstB(A, B):
            return 1 / 2 * (quat_AiB(A, quat_conj(B)) + quat_AiB(B, quat_conj(A)))

        # alpha = 2520 * delta_p - 435 * delta_v + 45 / 2 * delta_a - (
        #     60 * AstB(A1, A1) - 60 * AstB(A0, A3) - 60 * AstB(A1, A4) + 60 *
        #     AstB(A3, A3) - 42 * AstB(A0, A4) - 72 * AstB(A1, A3))

        # alpha_norm = cs.norm_2(alpha)
        # numden = (alpha/alpha_norm) + i
        # Qo2 = cs.vertcat(cs.cos(o2), cs.sin(o2), 0, 0)
        # A2 = 1/12*(quat_mult(cs.sqrt(alpha_norm)*(numden/cs.norm_2(numden)), Qo2) -
        #            (10*A1 + 5*A0 + 5*A4 + 10*A3))

        def ZiZ(A, B):
            return quat_AiB(A, quat_conj(B)).T

        alpha2 = 27 * (
            85 * ZiZ(A0, A0)
            + 30 * ZiZ(A0, A1)
            - 10 * ZiZ(A0, A3)
            - 7 * ZiZ(A0, A4)
            + 30 * ZiZ(A1, A0)
            + 20 * ZiZ(A1, A1)
            - 12 * ZiZ(A1, A3)
            - 10 * ZiZ(A1, A4)
            - 10 * ZiZ(A3, A0)
            - 12 * ZiZ(A3, A1)
            + 20 * ZiZ(A3, A3)
            + 30 * ZiZ(A3, A4)
            - 7 * ZiZ(A4, A0)
            - 10 * ZiZ(A4, A1)
            + 30 * ZiZ(A4, A3)
            + 85 * ZiZ(A4, A4)
        )
        alpha = 22680 * delta_p - alpha2.T
        alpha_norm = cs.norm_2(alpha)
        numden = (alpha / alpha_norm) + i
        Qo2 = cs.vertcat(cs.cos(o2), cs.sin(o2), 0, 0)
        A2 = (
            1
            / 36
            * (
                quat_mult(cs.sqrt(alpha_norm) * (numden / cs.norm_2(numden)), Qo2)
                - (15 * A0 + 30 * A1 + 30 * A3 + 15 * A4)
            )
        )

        f_A = cs.Function(
            "f_A",
            [pb, pe, vb, ve, ab, ae, o0, o2, o4, t1, t3],
            [A0, A1, A2, A3, A4],
            ["pb", "pe", "vb", "ve", "ab", "ae", "o0", "o2", "o3", "t1", "t3"],
            ["A0", "A1", "A2", "A3", "A4"],
        )

        # ------------- quaternion coefficients to quaternion polynomial ------------- #

        xi = cs.SX.sym("xi")  # path parameter
        A0 = cs.SX.sym("A0", 4)  # quat. coeff 0
        A1 = cs.SX.sym("A1", 4)  # quat. coeff 1
        A2 = cs.SX.sym("A2", 4)  # quat. coeff 2
        A3 = cs.SX.sym("A3", 4)  # quat. coeff 3
        A4 = cs.SX.sym("A4", 4)  # quat. coeff 4
        Zcoeff = cs.horzcat(A0, A1, A2, A3, A4).T

        U = Bezier(xi, Zcoeff[:, 0], 4)
        V = Bezier(xi, Zcoeff[:, 1], 4)
        G = Bezier(xi, Zcoeff[:, 2], 4)
        H = Bezier(xi, Zcoeff[:, 3], 4)
        Z = cs.vertcat(U, V, G, H)
        Z_d = cs.jacobian(Z, xi)
        Z_dd = cs.jacobian(Z_d, xi)
        Z_ddd = cs.jacobian(Z_dd, xi)
        f_Z = cs.Function(
            "f_Z",
            [xi, A0, A1, A2, A3, A4],
            [Z],
            ["xi", "A0", "A1", "A2", "A3", "A4"],
            ["Z"],
        )
        f_Zd = cs.Function(
            "f_Zd",
            [xi, A0, A1, A2, A3, A4],
            [Z_d],
            ["xi", "A0", "A1", "A2", "A3", "A4"],
            ["Zd"],
        )
        f_Zdd = cs.Function(
            "f_Zdd",
            [xi, A0, A1, A2, A3, A4],
            [Z_dd],
            ["xi", "A0", "A1", "A2", "A3", "A4"],
            ["Zdd"],
        )
        f_Zddd = cs.Function(
            "f_Zddd",
            [xi, A0, A1, A2, A3, A4],
            [Z_ddd],
            ["xi", "A0", "A1", "A2", "A3", "A4"],
            ["Zddd"],
        )

        # ------------------- quaternion control points to position ------------------ #

        # hodograph controlpoints
        h0 = AstB(A0, A0)
        h1 = AstB(A0, A1)
        h2 = 1 / 7 * (4 * AstB(A1, A1) + 3 * AstB(A0, A2))
        h3 = 1 / 7 * (AstB(A0, A3) + 6 * AstB(A1, A2))
        h4 = 1 / 35 * (18 * AstB(A2, A2) + AstB(A0, A4) + 16 * AstB(A1, A3))
        h5 = 1 / 7 * (AstB(A1, A4) + 6 * AstB(A2, A3))
        h6 = 1 / 7 * (4 * AstB(A3, A3) + 3 * AstB(A2, A4))
        h7 = AstB(A3, A4)
        h8 = AstB(A4, A4)
        hodograph = [h0, h1, h2, h3, h4, h5, h6, h7, h8]
        f_h = cs.Function(
            "f_h",
            [A0, A1, A2, A3, A4],
            hodograph,
            ["A0", "A1", "A2", "A3", "A4"],
            ["h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8"],
        )

        # curve controlpoints
        p0 = cs.SX.sym("p0", 3)
        p = cs.SX.zeros(10, 3)
        p[0, :] = p0
        for j in range(1, 10):
            # sum_hodograph = cs.SX.zeros(1,3)
            # for i in range(j):
            #    sum_hodograph += hodograph[i][1:,0].T
            p[j, :] = p[j - 1, :] + 1 / 9 * hodograph[j - 1][1:, 0].T

        # f_h = cs.Function('f_h', [p0, A0, A1, A2, A3, A4], hodograph)
        # f_p = cs.Function('f_p', [p0, A0, A1, A2, A3, A4], [p], [
        #    'p0', 'A0', 'A1', 'A2', 'A3', 'A4'], ['p'])

        # curve
        gamma = Bezier(xi, p).T
        gamma_d = cs.jacobian(gamma, xi)
        gamma_dd = cs.jacobian(gamma_d, xi)
        gamma_ddd = cs.jacobian(gamma_dd, xi)
        gamma_dddd = cs.jacobian(gamma_ddd, xi)
        f_gamma = cs.Function(
            "f_gamma",
            [xi, p0, A0, A1, A2, A3, A4],
            [gamma],
            ["xi", "p0", "A0", "A1", "A2", "A3", "A4"],
            ["gamma"],
        )
        f_gammad = cs.Function(
            "f_gammad",
            [xi, p0, A0, A1, A2, A3, A4],
            [gamma_d],
            ["xi", "p0", "A0", "A1", "A2", "A3", "A4"],
            ["gamma_d"],
        )
        f_gammadd = cs.Function(
            "f_gammadd",
            [xi, p0, A0, A1, A2, A3, A4],
            [gamma_dd],
            ["xi", "p0", "A0", "A1", "A2", "A3", "A4"],
            ["gamma_dd"],
        )
        f_gammaddd = cs.Function(
            "f_gammaddd",
            [xi, p0, A0, A1, A2, A3, A4],
            [gamma_ddd],
            ["xi", "p0", "A0", "A1", "A2", "A3", "A4"],
            ["gamma_ddd"],
        )
        f_gammadddd = cs.Function(
            "f_gammadddd",
            [xi, p0, A0, A1, A2, A3, A4],
            [gamma_dddd],
            ["xi", "p0", "A0", "A1", "A2", "A3", "A4"],
            ["gamma_dddd"],
        )

        # ----------------------------- parametric speed ----------------------------- #
        # sigma = Z[0]**2 + Z[1]**2 + Z[2]**2 + Z[3]**2
        sigma = cs.sqrt(gamma[0] ** 2 + gamma[1] ** 2 + gamma[2] ** 2)
        sigma_d = cs.jacobian(sigma, xi)
        sigma_dd = cs.jacobian(sigma_d, xi)

        f_sigma = cs.Function(
            "f_sigma",
            [xi, A0, A1, A2, A3, A4],
            [sigma],
            ["xi", "A0", "A1", "A2", "A3", "A4"],
            ["sigma"],
        )
        f_sigmad = cs.Function(
            "f_sigmad",
            [xi, A0, A1, A2, A3, A4],
            [sigma_d],
            ["xi", "A0", "A1", "A2", "A3", "A4"],
            ["sigmad"],
        )
        f_sigmadd = cs.Function(
            "f_sigmadd",
            [xi, A0, A1, A2, A3, A4],
            [sigma_dd],
            ["xi", "A0", "A1", "A2", "A3", "A4"],
            ["sigmadd"],
        )

        # ------------------------------- adapted frame ------------------------------ #
        erf, X = Z_to_erf(xi, Z)
        erf_d = cs.horzcat(
            cs.jacobian(erf[:, 0], xi),
            cs.jacobian(erf[:, 1], xi),
            cs.jacobian(erf[:, 2], xi),
        )
        erf_dd = cs.horzcat(
            cs.jacobian(erf_d[:, 0], xi),
            cs.jacobian(erf_d[:, 1], xi),
            cs.jacobian(erf_d[:, 2], xi),
        )
        # erf_ddd = cs.jacobian(erf_dd, xi)

        X_d = cs.jacobian(X, xi)
        X_dd = cs.jacobian(X_d, xi)

        f_erf = cs.Function(
            "f_erf",
            [xi, A0, A1, A2, A3, A4],
            [erf],
            ["xi", "A0", "A1", "A2", "A3", "A4"],
            ["erf"],
        )
        f_erfd = cs.Function(
            "f_erfd",
            [xi, A0, A1, A2, A3, A4],
            [erf_d],
            ["xi", "A0", "A1", "A2", "A3", "A4"],
            ["erfd"],
        )
        f_erfdd = cs.Function(
            "f_erfdd",
            [xi, A0, A1, A2, A3, A4],
            [erf_dd],
            ["xi", "A0", "A1", "A2", "A3", "A4"],
            ["erfdd"],
        )
        # f_erfddd = cs.Function('f_erfdd', [xi, A0, A1, A2, A3, A4], [erf_ddd], [
        #    'xi', 'A0', 'A1', 'A2', 'A3', 'A4'], ['erfddd'])

        f_X = cs.Function(
            "f_X",
            [xi, A0, A1, A2, A3, A4],
            [X],
            ["xi", "A0", "A1", "A2", "A3", "A4"],
            ["X"],
        )
        f_Xd = cs.Function(
            "f_X",
            [xi, A0, A1, A2, A3, A4],
            [X_d],
            ["xi", "A0", "A1", "A2", "A3", "A4"],
            ["X_d"],
        )
        f_Xdd = cs.Function(
            "f_X",
            [xi, A0, A1, A2, A3, A4],
            [X_dd],
            ["xi", "A0", "A1", "A2", "A3", "A4"],
            ["X_dd"],
        )

        # -------------------------- dictionary of functions ------------------------- #
        func_dict = {
            "Z": f_Z,
            "Zd": f_Zd,
            "Zdd": f_Zdd,
            "Zddd": f_Zddd,
            "A": f_A,
            "gamma": f_gamma,
            "gamma_d": f_gammad,
            "gamma_dd": f_gammadd,
            "gamma_ddd": f_gammaddd,
            "gamma_dddd": f_gammadddd,
            "hodograph": f_h,
            "sigma": f_sigma,
            "sigma_d": f_sigmad,
            "sigma_dd": f_sigmadd,
            "erf": f_erf,
            "erf_d": f_erfd,
            "erf_dd": f_erfdd,
            "X": f_X,
            "X_d": f_Xd,
            "X_dd": f_Xdd,
        }

    elif interp_order == 4:

        def AstB(A, B):
            return 1 / 2 * (quat_AiB(A, quat_conj(B)) + quat_AiB(B, quat_conj(A)))

        def ZiZ(A, B):
            return quat_AiB(A, quat_conj(B)).T

        # symbolic variables and constants
        pb = cs.SX.sym("pb", 3)
        pe = cs.SX.sym("pe", 3)
        vb = cs.SX.sym("vb", 3)
        ve = cs.SX.sym("ve", 3)
        ab = cs.SX.sym("ab", 3)
        ae = cs.SX.sym("ae", 3)
        jb = cs.SX.sym("jb", 3)
        je = cs.SX.sym("je", 3)
        sb = cs.SX.sym("sb", 3)
        se = cs.SX.sym("se", 3)

        o0 = cs.SX.sym("o0")
        t1 = cs.SX.sym("t1")
        t2 = cs.SX.sym("t2")
        t3 = cs.SX.sym("t3")
        o4 = cs.SX.sym("o4")
        t5 = cs.SX.sym("t5")
        t6 = cs.SX.sym("t6")
        t7 = cs.SX.sym("t7")
        o8 = cs.SX.sym("o8")

        i = cs.DM([0, 1, 0, 0])

        # ------------------- function for C2 hermite interpolation ------------------ #
        # Step 1
        h0 = cs.vertcat(0, vb)
        h16 = cs.vertcat(0, ve)
        h1 = 1 / 16 * (cs.vertcat(0, ab) - (-16 * h0))
        h15 = 1 / -16 * (cs.vertcat(0, ae) - (16 * h16))
        h2 = 1 / 240 * (cs.vertcat(0, jb) - (240 * h0 - 480 * h1))
        h14 = 1 / 240 * (cs.vertcat(0, je) - (-480 * h15 + 240 * h16))
        h3 = 1 / 3360 * (cs.vertcat(0, sb) - (-3360 * h0 + 10080 * h1 - 10080 * h2))
        h13 = 1 / -3360 * (cs.vertcat(0, se) - (10080 * h14 - 10080 * h15 + 3360 * h16))

        # Step 2
        h0_norm = cs.norm_2(h0)
        numden = (h0 / h0_norm) + i
        Qo0 = cs.vertcat(cs.cos(o0), cs.sin(o0), 0, 0)
        A0 = quat_mult(cs.sqrt(h0_norm) * (numden / cs.norm_2(numden)), Qo0)

        h16_norm = cs.norm_2(h16)
        numden = (h16 / h16_norm) + i
        Qo8 = cs.vertcat(cs.cos(o8), cs.sin(o8), 0, 0)
        A8 = quat_mult(cs.sqrt(h16_norm) * (numden / cs.norm_2(numden)), Qo8)

        # Step 3
        A1 = quat_mult(quat_mult(-(cs.vertcat(t1, 0, 0, 0) + h1), A0), i) / (
            cs.norm_2(A0) ** 2
        )
        A7 = quat_mult(quat_mult(-(cs.vertcat(t7, 0, 0, 0) + h15), A8), i) / (
            cs.norm_2(A8) ** 2
        )
        # Step 4
        h2_2 = 1 / 7 * (15 * h2 - 8 * AstB(A1, A1))
        h14_2 = 1 / 7 * (15 * h14 - 8 * AstB(A7, A7))
        A2 = quat_mult(quat_mult(-(cs.vertcat(t2, 0, 0, 0) + h2_2), A0), i) / (
            cs.norm_2(A0) ** 2
        )
        A6 = quat_mult(quat_mult(-(cs.vertcat(t6, 0, 0, 0) + h14_2), A8), i) / (
            cs.norm_2(A8) ** 2
        )

        # Step 5
        h3_2 = 1 / 2 * (10 * h3 - 8 * AstB(A1, A2))
        h13_2 = 1 / 2 * (10 * h13 - 8 * AstB(A6, A7))
        A3 = quat_mult(quat_mult(-(cs.vertcat(t3, 0, 0, 0) + h3_2), A0), i) / (
            cs.norm_2(A0) ** 2
        )
        A5 = quat_mult(quat_mult(-(cs.vertcat(t5, 0, 0, 0) + h13_2), A8), i) / (
            cs.norm_2(A8) ** 2
        )

        # Step 6
        delta_p = cs.vertcat(0, pe - pb)

        alpha2 = (1 / 112633092) * (
            147807 * ZiZ(A0, A0)
            + 72270 * ZiZ(A0, A1)
            + 30954 * ZiZ(A0, A2)
            + 9702 * ZiZ(A0, A3)
            - 3234 * ZiZ(A0, A5)
            - 3150 * ZiZ(A0, A6)
            - 1818 * ZiZ(A0, A7)
            - 565 * ZiZ(A0, A8)
            + 72270 * ZiZ(A1, A0)
            + 72732 * ZiZ(A1, A1)
            + 47124 * ZiZ(A1, A2)
            + 19404 * ZiZ(A1, A3)
            - 8820 * ZiZ(A1, A5)
            - 9324 * ZiZ(A1, A6)
            - 5668 * ZiZ(A1, A7)
            - 1818 * ZiZ(A1, A8)
            + 30954 * ZiZ(A2, A0)
            + 47124 * ZiZ(A2, A1)
            + 40572 * ZiZ(A2, A2)
            + 20580 * ZiZ(A2, A3)
            - 12348 * ZiZ(A2, A5)
            - 14308 * ZiZ(A2, A6)
            - 9324 * ZiZ(A2, A7)
            - 3150 * ZiZ(A2, A8)
            + 9702 * ZiZ(A3, A0)
            + 19404 * ZiZ(A3, A1)
            + 20580 * ZiZ(A3, A2)
            + 12348 * ZiZ(A3, A3)
            - 9604 * ZiZ(A3, A5)
            - 12348 * ZiZ(A3, A6)
            - 8820 * ZiZ(A3, A7)
            - 3234 * ZiZ(A3, A8)
            - 3234 * ZiZ(A5, A0)
            - 8820 * ZiZ(A5, A1)
            - 12348 * ZiZ(A5, A2)
            - 9604 * ZiZ(A5, A3)
            + 12348 * ZiZ(A5, A5)
            + 20580 * ZiZ(A5, A6)
            + 19404 * ZiZ(A5, A7)
            + 9702 * ZiZ(A5, A8)
            - 3150 * ZiZ(A6, A0)
            - 9324 * ZiZ(A6, A1)
            - 14308 * ZiZ(A6, A2)
            - 12348 * ZiZ(A6, A3)
            + 20580 * ZiZ(A6, A5)
            + 40572 * ZiZ(A6, A6)
            + 47124 * ZiZ(A6, A7)
            + 30954 * ZiZ(A6, A8)
            - 1818 * ZiZ(A7, A0)
            - 5668 * ZiZ(A7, A1)
            - 9324 * ZiZ(A7, A2)
            - 8820 * ZiZ(A7, A3)
            + 19404 * ZiZ(A7, A5)
            + 47124 * ZiZ(A7, A6)
            + 72732 * ZiZ(A7, A7)
            + 72270 * ZiZ(A7, A8)
            - 565 * ZiZ(A8, A0)
            - 1818 * ZiZ(A8, A1)
            - 3150 * ZiZ(A8, A2)
            - 3234 * ZiZ(A8, A3)
            + 9702 * ZiZ(A8, A5)
            + 30954 * ZiZ(A8, A6)
            + 72270 * ZiZ(A8, A7)
            + 147807 * ZiZ(A8, A8)
        )

        alpha = 490 / 21879 * delta_p - alpha2.T
        alpha_norm = cs.norm_2(alpha)
        numden = (alpha / alpha_norm) + i
        Qo4 = cs.vertcat(cs.cos(o4), cs.sin(o4), 0, 0)
        A4 = (
            21879
            / 490
            * (
                quat_mult(cs.sqrt(alpha_norm) * (numden / cs.norm_2(numden)), Qo4)
                - (
                    (1 / 442) * A0
                    + (5 / 663) * A1
                    + (35 / 2431) * A2
                    + (49 / 2431) * A3
                    + (49 / 2431) * A5
                    + (35 / 2431) * A6
                    + (5 / 663) * A7
                    + (1 / 442) * A8
                )
            )
        )

        f_A = cs.Function(
            "f_A",
            [
                pb,
                pe,
                vb,
                ve,
                ab,
                ae,
                jb,
                je,
                sb,
                se,
                o0,
                t1,
                t2,
                t3,
                o4,
                t5,
                t6,
                t7,
                o8,
            ],
            [A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [
                "pb",
                "pe",
                "vb",
                "ve",
                "ab",
                "ae",
                "jb",
                "je",
                "sb",
                "se",
                "o0",
                "t1",
                "t2",
                "t3",
                "o4",
                "t5",
                "t6",
                "t7",
                "o8",
            ],
            ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
        )

        # ------------- quaternion coefficients to quaternion polynomial ------------- #

        xi = cs.SX.sym("xi")  # path parameter
        A0 = cs.SX.sym("A0", 4)  # quat. coeff 0
        A1 = cs.SX.sym("A1", 4)  # quat. coeff 1
        A2 = cs.SX.sym("A2", 4)  # quat. coeff 2
        A3 = cs.SX.sym("A3", 4)  # quat. coeff 3
        A4 = cs.SX.sym("A4", 4)  # quat. coeff 4
        A5 = cs.SX.sym("A5", 4)  # quat. coeff 5
        A6 = cs.SX.sym("A6", 4)  # quat. coeff 6
        A7 = cs.SX.sym("A7", 4)  # quat. coeff 7
        A8 = cs.SX.sym("A8", 4)  # quat. coeff 8
        Zcoeff = cs.horzcat(A0, A1, A2, A3, A4, A5, A6, A7, A8).T

        U = Bezier(xi, Zcoeff[:, 0], 8)
        V = Bezier(xi, Zcoeff[:, 1], 8)
        G = Bezier(xi, Zcoeff[:, 2], 8)
        H = Bezier(xi, Zcoeff[:, 3], 8)

        Z = cs.vertcat(U, V, G, H)
        Z_d = cs.jacobian(Z, xi)
        Z_dd = cs.jacobian(Z_d, xi)
        Z_ddd = cs.jacobian(Z_dd, xi)

        f_Z = cs.Function(
            "f_Z",
            [xi, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [Z],
            ["xi", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["Z"],
        )
        f_Zd = cs.Function(
            "f_Zd",
            [xi, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [Z_d],
            ["xi", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["Zd"],
        )
        f_Zdd = cs.Function(
            "f_Zdd",
            [xi, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [Z_dd],
            ["xi", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["Zdd"],
        )
        f_Zddd = cs.Function(
            "f_Zddd",
            [xi, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [Z_ddd],
            ["xi", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["Zddd"],
        )

        # ----------------------------- parametric speed ----------------------------- #
        sigma = Z[0] ** 2 + Z[1] ** 2 + Z[2] ** 2 + Z[3] ** 2
        sigma_d = cs.jacobian(sigma, xi)
        sigma_dd = cs.jacobian(sigma_d, xi)

        f_sigma = cs.Function(
            "f_sigma",
            [xi, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [sigma],
            ["xi", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["sigma"],
        )
        f_sigmad = cs.Function(
            "f_sigmad",
            [xi, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [sigma_d],
            ["xi", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["sigmad"],
        )
        f_sigmadd = cs.Function(
            "f_sigmadd",
            [xi, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [sigma_dd],
            ["xi", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["sigmadd"],
        )
        # -------------------------------- arc-length -------------------------------- #
        def arclength():
            expr_math = "g0^2*t + h0^2*t + 8*g0*(-g0 + g1)*t^2 + 8*h0*(-h0 + h1)*t^2 + (8/3)*(15*g0^2 - 30*g0*g1 + 8*g1^2 + 7*g0*g2)*t^3 + (8/3)*(15*h0^2 - 30*h0*h1 + 8*h1^2 + 7*h0*h2)*t^3 - 28*(5*g0^2 + 4*g1*(2*g1 - g2) - g0*(15*g1 - 7*g2 + g3))*t^4 - 28*(5*h0^2 + 4*h1*(2*h1 - h2) - h0*(15*h1 - 7*h2 + h3))*t^4 + (28/5)*(65*g0^2 + 4*(52*g1^2 - 52*g1*g2 + 7*g2^2 + 8*g1*g3) + g0*(-260*g1 + 182*g2 - 52*g3 + 5*g4))*t^5 + (28/5)*(65*h0^2 + 4*(52*h1^2 - 52*h1*h2 + 7*h2^2 + 8*h1*h3) + h0*(-260*h1 + 182*h2 - 52*h3 + 5*h4))*t^5 - (56/3)*(39*g0^2 + 208*g1^2 + 28*g2*(3*g2 - g3) - 2*g1*(156*g2 - 48*g3 + 5*g4) - g0*(195*g1 - 182*g2 + 78*g3 - 15*g4 + g5))*t^6 - (56/3)*(39*h0^2 + 208*h1^2 + 28*h2*(3*h2 - h3) - 2*h1*(156*h2 - 48*h3 + 5*h4) - h0*(195*h1 - 182*h2 + 78*h3 - 15*h4 + h5))*t^6 + 8*(143*g0^2 + 1144*g1^2 + 14*(66*g2^2 - 44*g2*g3 + 4*g3^2 + 5*g2*g4) - 11*g0*(78*g1 - 91*g2 + 52*g3 - 15*g4 + 2*g5) + 4*g1*(-572*g2 + 264*g3 - 55*g4 + 4*g5) + g0*g6)*t^7 + 8*(143*h0^2 + 1144*h1^2 + 14*(66*h2^2 - 44*h2*h3 + 4*h3^2 + 5*h2*h4) - 11*h0*(78*h1 - 91*h2 + 52*h3 - 15*h4 + 2*h5) + 4*h1*(-572*h2 + 264*h3 - 55*h4 + 4*h5) + h0*h6)*t^7 - 2*(715*g0^2 + 14*(572*g1^2 + 35*(22*g2^2 - 22*g2*g3 + 4*g3^2 + 5*g2*g4 - g3*g4) - 14*g2*g5 + g1*(-1430*g2 + 880*g3 - 275*g4 + 40*g5 - 2*g6)) - g0*(77*(65*g1 - 91*g2 + 65*g3 - 25*g4 + 5*g5) - 35*g6 + g7))*t^8 - 2*(715*h0^2 + 14*(572*h1^2 + 35*(22*h2^2 - 22*h2*h3 + 4*h3^2 + 5*h2*h4 - h3*h4) - 14*h2*h5 + h1*(-1430*h2 + 880*h3 - 275*h4 + 40*h5 - 2*h6)) - h0*(77*(65*h1 - 91*h2 + 65*h3 - 25*h4 + 5*h5) - 35*h6 + h7))*t^8 + (2/9)*(6435*g0^2 + 96096*g1^2 - 336*g1*(858*g2 - 660*g3 + 275*g4 - 60*g5 + 6*g6) + 98*(1980*g2^2 + 720*g3^2 - 360*g3*g4 + 25*g4^2 + 32*g3*g5 + 4*g2*(-660*g3 + 225*g4 - 36*g5 + 2*g6)) + 64*g1*g7 + g0*(-51480*g1 + 84084*g2 - 72072*g3 + 34650*g4 - 9240*g5 + 1260*g6 - 72*g7 + g8))*t^9 + (2/9)*(6435*h0^2 + 96096*h1^2 - 336*h1*(858*h2 - 660*h3 + 275*h4 - 60*h5 + 6*h6) + 98*(1980*h2^2 + 720*h3^2 - 360*h3*h4 + 25*h4^2 + 32*h3*h5 + 4*h2*(-660*h3 + 225*h4 - 36*h5 + 2*h6)) + 64*h1*h7 + h0*(-51480*h1 + 84084*h2 - 72072*h3 + 34650*h4 - 9240*h5 + 1260*h6 - 72*h7 + h8))*t^9 - (8/5)*(715*g0^2 + 13728*g1^2 - 84*g1*(572*g2 - 528*g3 + 275*g4 - 80*g5 + 12*g6) + 98*(396*g2^2 + 240*g3^2 + 5*g4*(5*g4 - g5) - 2*g3*(90*g4 - 16*g5 + g6) + g2*(-660*g3 + 300*g4 - 72*g5 + 8*g6)) + 64*g1*g7 - 28*g2*g7 - g1*g8 + g0*(-6435*g1 + 6*(77*(26*g2 - 26*g3 + 15*g4 - 5*g5) + 70*g6 - 6*g7) + g8))*t^10 - (8/5)*(715*h0^2 + 13728*h1^2 - 84*h1*(572*h2 - 528*h3 + 275*h4 - 80*h5 + 12*h6) + 98*(396*h2^2 + 240*h3^2 + 5*h4*(5*h4 - h5) - 2*h3*(90*h4 - 16*h5 + h6) + h2*(-660*h3 + 300*h4 - 72*h5 + 8*h6)) + 64*h1*h7 - 28*h2*h7 - h1*h8 + h0*(-6435*h1 + 6*(77*(26*h2 - 26*h3 + 15*h4 - 5*h5) + 70*h6 - 6*h7) + h8))*t^10 + (56/11)*(143*g0^2 + 3432*g1^2 + 14*(924*g2^2 + 840*g3^2 + 175*g4^2 - 70*g4*g5 + 4*g5^2 - 14*g2*(132*g3 - 75*g4 + 24*g5 - 4*g6) + 5*g4*g6 - 28*g3*(30*g4 - 8*g5 + g6)) + 8*(-7*g2 + 2*g3)*g7 + g2*g8 - 2*g1*(6864*g2 - 7392*g3 + 4620*g4 - 1680*g5 + 336*g6 - 32*g7 + g8) + g0*(-1430*g1 + 3003*g2 - 6*(572*g3 - 385*g4 + 154*g5 - 35*g6 + 4*g7) + g8))*t^11 + (56/11)*(143*h0^2 + 3432*h1^2 + 14*(924*h2^2 + 840*h3^2 + 175*h4^2 - 70*h4*h5 + 4*h5^2 - 14*h2*(132*h3 - 75*h4 + 24*h5 - 4*h6) + 5*h4*h6 - 28*h3*(30*h4 - 8*h5 + h6)) + 8*(-7*h2 + 2*h3)*h7 + h2*h8 - 2*h1*(6864*h2 - 7392*h3 + 4620*h4 - 1680*h5 + 336*h6 - 32*h7 + h8) + h0*(-1430*h1 + 3003*h2 - 6*(572*h3 - 385*h4 + 154*h5 - 35*h6 + 4*h7) + h8))*t^11 - (28/3)*(39*g0^2 + 2*(572*g1^2 - 6*g1*(429*g2 - 528*g3 + 385*g4 + 42*(-4*g5 + g6)) + 7*(396*g2^2 + 504*g3^2 + 175*g4^2 - 105*g4*g5 + 12*g5^2 + 15*g4*g6 - 2*g5*g6 - 14*g3*(45*g4 - 16*g5 + 3*g6) + 14*g2*(-66*g3 + 45*g4 - 18*g5 + 4*g6)) + 32*g1*g7 + (-42*g2 + 24*g3 - 5*g4)*g7) - (3*g1 - 3*g2 + g3)*g8 + g0*(-429*g1 + 1001*g2 - 1287*g3 + 990*g4 - 462*g5 + 126*g6 - 18*g7 + g8))*t^12 - (28/3)*(39*h0^2 + 2*(572*h1^2 - 6*h1*(429*h2 - 528*h3 + 385*h4 + 42*(-4*h5 + h6)) + 7*(396*h2^2 + 504*h3^2 + 175*h4^2 - 105*h4*h5 + 12*h5^2 + 15*h4*h6 - 2*h5*h6 - 14*h3*(45*h4 - 16*h5 + 3*h6) + 14*h2*(-66*h3 + 45*h4 - 18*h5 + 4*h6)) + 32*h1*h7 + (-42*h2 + 24*h3 - 5*h4)*h7) - (3*h1 - 3*h2 + h3)*h8 + h0*(-429*h1 + 1001*h2 - 1287*h3 + 990*h4 - 462*h5 + 126*h6 - 18*h7 + h8))*t^12 + (28/13)*(65*g0^2 + 2288*g1^2 + 14*(990*g2^2 + 1680*g3^2 + 875*g4^2 - 700*g4*g5 + 120*g5^2 - 4*g2*(660*g3 - 525*g4 + 252*g5 - 70*g6) + 150*g4*g6 - 40*g5*g6 + 2*g6^2 - 280*g3*(9*g4 - 4*g5 + g6)) - 16*g1*(715*g2 - 990*g3 + 825*g4 - 420*g5 + 126*g6 - 20*g7) - 8*(70*g2 - 60*g3 + 25*g4 - 4*g5)*g7 + 5*(-4*g1 + 6*g2 - 4*g3 + g4)*g8 + g0*(-780*g1 + 2002*g2 - 2860*g3 + 2475*g4 - 1320*g5 + 420*g6 - 72*g7 + 5*g8))*t^13 + (28/13)*(65*h0^2 + 2288*h1^2 + 14*(990*h2^2 + 1680*h3^2 + 875*h4^2 - 700*h4*h5 + 120*h5^2 - 4*h2*(660*h3 - 525*h4 + 252*h5 - 70*h6) + 150*h4*h6 - 40*h5*h6 + 2*h6^2 - 280*h3*(9*h4 - 4*h5 + h6)) - 16*h1*(715*h2 - 990*h3 + 825*h4 - 420*h5 + 126*h6 - 20*h7) - 8*(70*h2 - 60*h3 + 25*h4 - 4*h5)*h7 + 5*(-4*h1 + 6*h2 - 4*h3 + h4)*h8 + h0*(-780*h1 + 2002*h2 - 2860*h3 + 2475*h4 - 1320*h5 + 420*h6 - 72*h7 + 5*h8))*t^13 - 8*(5*g0^2 + 2*(104*g1^2 + 7*(110*g2^2 + 240*g3^2 - 420*g3*g4 + 175*g4^2 + 224*g3*g5 - 175*g4*g5 + 40*g5^2 - 10*(7*g3 - 5*g4 + 2*g5)*g6 + 2*g6^2 + g2*(-330*g3 + 300*g4 + 56*(-3*g5 + g6))) - 2*(35*g2 - 40*g3 + 25*g4 - 8*g5 + g6)*g7 + g1*(-572*g2 + 880*g3 - 825*g4 + 480*g5 - 168*g6 + 32*g7)) - (5*g1 - 5*(2*g2 - 2*g3 + g4) + g5)*g8 + g0*(-65*g1 + 182*g2 - 286*g3 + 275*g4 - 165*g5 + 60*g6 - 12*g7 + g8))*t^14 - 8*(5*h0^2 + 2*(104*h1^2 + 7*(110*h2^2 + 240*h3^2 - 420*h3*h4 + 175*h4^2 + 224*h3*h5 - 175*h4*h5 + 40*h5^2 - 10*(7*h3 - 5*h4 + 2*h5)*h6 + 2*h6^2 + h2*(-330*h3 + 300*h4 + 56*(-3*h5 + h6))) - 2*(35*h2 - 40*h3 + 25*h4 - 8*h5 + h6)*h7 + h1*(-572*h2 + 880*h3 - 825*h4 + 480*h5 - 168*h6 + 32*h7)) - (5*h1 - 5*(2*h2 - 2*h3 + h4) + h5)*h8 + h0*(-65*h1 + 182*h2 - 286*h3 + 275*h4 - 165*h5 + 60*h6 - 12*h7 + h8))*t^14 + (8/15)*(15*g0^2 + 728*g1^2 + 98*(66*g2^2 + 180*g3^2 + 175*g4^2 - 210*g4*g5 + 60*g5^2 + 75*g4*g6 - 40*g5*g6 + 6*g6^2 - 4*g3*(90*g4 - 56*g5 + 21*g6) + g2*(-220*g3 + 225*g4 - 144*g5 + 56*g6)) - 56*(21*g2 - 30*g3 + 25*g4 - 12*g5 + 3*g6)*g7 + 8*g7^2 + 7*(15*g2 - 20*g3 + 15*g4 - 6*g5 + g6)*g8 - 14*g1*(312*g2 - 528*g3 + 550*g4 - 360*g5 + 144*g6 - 32*g7 + 3*g8) + g0*(-210*g1 + 637*g2 - 1092*g3 + 1155*g4 - 770*g5 + 315*g6 - 72*g7 + 7*g8))*t^15 + (8/15)*(15*h0^2 + 728*h1^2 + 98*(66*h2^2 + 180*h3^2 + 175*h4^2 - 210*h4*h5 + 60*h5^2 + 75*h4*h6 - 40*h5*h6 + 6*h6^2 - 4*h3*(90*h4 - 56*h5 + 21*h6) + h2*(-220*h3 + 225*h4 - 144*h5 + 56*h6)) - 56*(21*h2 - 30*h3 + 25*h4 - 12*h5 + 3*h6)*h7 + 8*h7^2 + 7*(15*h2 - 20*h3 + 15*h4 - 6*h5 + h6)*h8 - 14*h1*(312*h2 - 528*h3 + 550*h4 - 360*h5 + 144*h6 - 32*h7 + 3*h8) + h0*(-210*h1 + 637*h2 - 1092*h3 + 1155*h4 - 770*h5 + 315*h6 - 72*h7 + 7*h8))*t^15 - (g0 - 7*g1 + 7*(3*g2 - 5*g3 + 5*g4 - 3*g5 + g6) - g7)*(g0 - 8*g1 + 28*g2 - 56*g3 + 70*g4 - 56*g5 + 28*g6 - 8*g7 + g8)*t^16 - (h0 - 7*h1 + 7*(3*h2 - 5*h3 + 5*h4 - 3*h5 + h6) - h7)*(h0 - 8*h1 + 28*h2 - 56*h3 + 70*h4 - 56*h5 + 28*h6 - 8*h7 + h8)*t^16 + (1/17)*(g0 - 8*g1 + 28*g2 - 56*g3 + 70*g4 - 56*g5 + 28*g6 - 8*g7 + g8)^2*t^17 + (1/17)*(h0 - 8*h1 + 28*h2 - 56*h3 + 70*h4 - 56*h5 + 28*h6 - 8*h7 + h8)^2*t^17 + t*u0^2 + 8*t^2*u0*(-u0 + u1) + (8/3)*t^3*(15*u0^2 - 30*u0*u1 + 8*u1^2 + 7*u0*u2) - 28*t^4*(5*u0^2 + 4*u1*(2*u1 - u2) - u0*(15*u1 - 7*u2 + u3)) + (28/5)*t^5*(65*u0^2 + 4*(52*u1^2 - 52*u1*u2 + 7*u2^2 + 8*u1*u3) + u0*(-260*u1 + 182*u2 - 52*u3 + 5*u4)) - (56/3)*t^6*(39*u0^2 + 208*u1^2 + 28*u2*(3*u2 - u3) - 2*u1*(156*u2 - 48*u3 + 5*u4) - u0*(195*u1 - 182*u2 + 78*u3 - 15*u4 + u5)) + 8*t^7*(143*u0^2 + 1144*u1^2 + 14*(66*u2^2 - 44*u2*u3 + 4*u3^2 + 5*u2*u4) - 11*u0*(78*u1 - 91*u2 + 52*u3 - 15*u4 + 2*u5) + 4*u1*(-572*u2 + 264*u3 - 55*u4 + 4*u5) + u0*u6) - 2*t^8*(715*u0^2 + 14*(572*u1^2 + 35*(22*u2^2 - 22*u2*u3 + 4*u3^2 + 5*u2*u4 - u3*u4) - 14*u2*u5 + u1*(-1430*u2 + 880*u3 - 275*u4 + 40*u5 - 2*u6)) - u0*(77*(65*u1 - 91*u2 + 65*u3 - 25*u4 + 5*u5) - 35*u6 + u7)) - t^16*(u0 - 7*u1 + 7*(3*u2 - 5*u3 + 5*u4 - 3*u5 + u6) - u7)*(u0 - 8*u1 + 28*u2 - 56*u3 + 70*u4 - 56*u5 + 28*u6 - 8*u7 + u8) + (1/17)*t^17*(u0 - 8*u1 + 28*u2 - 56*u3 + 70*u4 - 56*u5 + 28*u6 - 8*u7 + u8)^2 - (8/5)*t^10*(715*u0^2 + 13728*u1^2 - 84*u1*(572*u2 - 528*u3 + 275*u4 - 80*u5 + 12*u6) + 98*(396*u2^2 + 240*u3^2 + 5*u4*(5*u4 - u5) - 2*u3*(90*u4 - 16*u5 + u6) + u2*(-660*u3 + 300*u4 - 72*u5 + 8*u6)) + 64*u1*u7 - 28*u2*u7 - u1*u8 + u0*(-6435*u1 + 6*(77*(26*u2 - 26*u3 + 15*u4 - 5*u5) + 70*u6 - 6*u7) + u8)) + (2/9)*t^9*(6435*u0^2 + 96096*u1^2 - 336*u1*(858*u2 - 660*u3 + 275*u4 - 60*u5 + 6*u6) + 98*(1980*u2^2 + 720*u3^2 - 360*u3*u4 + 25*u4^2 + 32*u3*u5 + 4*u2*(-660*u3 + 225*u4 - 36*u5 + 2*u6)) + 64*u1*u7 + u0*(-51480*u1 + 84084*u2 - 72072*u3 + 34650*u4 - 9240*u5 + 1260*u6 - 72*u7 + u8)) - (28/3)*t^12*(39*u0^2 + 2*(572*u1^2 - 6*u1*(429*u2 - 528*u3 + 385*u4 + 42*(-4*u5 + u6)) + 7*(396*u2^2 + 504*u3^2 + 175*u4^2 - 105*u4*u5 + 12*u5^2 + 15*u4*u6 - 2*u5*u6 - 14*u3*(45*u4 - 16*u5 + 3*u6) + 14*u2*(-66*u3 + 45*u4 - 18*u5 + 4*u6)) + 32*u1*u7 + (-42*u2 + 24*u3 - 5*u4)*u7) - (3*u1 - 3*u2 + u3)*u8 + u0*(-429*u1 + 1001*u2 - 1287*u3 + 990*u4 - 462*u5 + 126*u6 - 18*u7 + u8)) - 8*t^14*(5*u0^2 + 2*(104*u1^2 + 7*(110*u2^2 + 240*u3^2 - 420*u3*u4 + 175*u4^2 + 224*u3*u5 - 175*u4*u5 + 40*u5^2 - 10*(7*u3 - 5*u4 + 2*u5)*u6 + 2*u6^2 + u2*(-330*u3 + 300*u4 + 56*(-3*u5 + u6))) - 2*(35*u2 - 40*u3 + 25*u4 - 8*u5 + u6)*u7 + u1*(-572*u2 + 880*u3 - 825*u4 + 480*u5 - 168*u6 + 32*u7)) - (5*u1 - 5*(2*u2 - 2*u3 + u4) + u5)*u8 + u0*(-65*u1 + 182*u2 - 286*u3 + 275*u4 - 165*u5 + 60*u6 - 12*u7 + u8)) + (56/11)*t^11*(143*u0^2 + 3432*u1^2 + 14*(924*u2^2 + 840*u3^2 + 175*u4^2 - 70*u4*u5 + 4*u5^2 - 14*u2*(132*u3 - 75*u4 + 24*u5 - 4*u6) + 5*u4*u6 - 28*u3*(30*u4 - 8*u5 + u6)) + 8*(-7*u2 + 2*u3)*u7 + u2*u8 - 2*u1*(6864*u2 - 7392*u3 + 4620*u4 - 1680*u5 + 336*u6 - 32*u7 + u8) + u0*(-1430*u1 + 3003*u2 - 6*(572*u3 - 385*u4 + 154*u5 - 35*u6 + 4*u7) + u8)) + (28/13)*t^13*(65*u0^2 + 2288*u1^2 + 14*(990*u2^2 + 1680*u3^2 + 875*u4^2 - 700*u4*u5 + 120*u5^2 - 4*u2*(660*u3 - 525*u4 + 252*u5 - 70*u6) + 150*u4*u6 - 40*u5*u6 + 2*u6^2 - 280*u3*(9*u4 - 4*u5 + u6)) - 16*u1*(715*u2 - 990*u3 + 825*u4 - 420*u5 + 126*u6 - 20*u7) - 8*(70*u2 - 60*u3 + 25*u4 - 4*u5)*u7 + 5*(-4*u1 + 6*u2 - 4*u3 + u4)*u8 + u0*(-780*u1 + 2002*u2 - 2860*u3 + 2475*u4 - 1320*u5 + 420*u6 - 72*u7 + 5*u8)) + (8/15)*t^15*(15*u0^2 + 728*u1^2 + 98*(66*u2^2 + 180*u3^2 + 175*u4^2 - 210*u4*u5 + 60*u5^2 + 75*u4*u6 - 40*u5*u6 + 6*u6^2 - 4*u3*(90*u4 - 56*u5 + 21*u6) + u2*(-220*u3 + 225*u4 - 144*u5 + 56*u6)) - 56*(21*u2 - 30*u3 + 25*u4 - 12*u5 + 3*u6)*u7 + 8*u7^2 + 7*(15*u2 - 20*u3 + 15*u4 - 6*u5 + u6)*u8 - 14*u1*(312*u2 - 528*u3 + 550*u4 - 360*u5 + 144*u6 - 32*u7 + 3*u8) + u0*(-210*u1 + 637*u2 - 1092*u3 + 1155*u4 - 770*u5 + 315*u6 - 72*u7 + 7*u8)) + t*v0^2 + 8*t^2*v0*(-v0 + v1) + (8/3)*t^3*(15*v0^2 - 30*v0*v1 + 8*v1^2 + 7*v0*v2) - 28*t^4*(5*v0^2 + 4*v1*(2*v1 - v2) - v0*(15*v1 - 7*v2 + v3)) + (28/5)*t^5*(65*v0^2 + 4*(52*v1^2 - 52*v1*v2 + 7*v2^2 + 8*v1*v3) + v0*(-260*v1 + 182*v2 - 52*v3 + 5*v4)) - (56/3)*t^6*(39*v0^2 + 208*v1^2 + 28*v2*(3*v2 - v3) - 2*v1*(156*v2 - 48*v3 + 5*v4) - v0*(195*v1 - 182*v2 + 78*v3 - 15*v4 + v5)) + 8*t^7*(143*v0^2 + 1144*v1^2 + 14*(66*v2^2 - 44*v2*v3 + 4*v3^2 + 5*v2*v4) - 11*v0*(78*v1 - 91*v2 + 52*v3 - 15*v4 + 2*v5) + 4*v1*(-572*v2 + 264*v3 - 55*v4 + 4*v5) + v0*v6) - 2*t^8*(715*v0^2 + 14*(572*v1^2 + 35*(22*v2^2 - 22*v2*v3 + 4*v3^2 + 5*v2*v4 - v3*v4) - 14*v2*v5 + v1*(-1430*v2 + 880*v3 - 275*v4 + 40*v5 - 2*v6)) - v0*(77*(65*v1 - 91*v2 + 65*v3 - 25*v4 + 5*v5) - 35*v6 + v7)) - t^16*(v0 - 7*v1 + 7*(3*v2 - 5*v3 + 5*v4 - 3*v5 + v6) - v7)*(v0 - 8*v1 + 28*v2 - 56*v3 + 70*v4 - 56*v5 + 28*v6 - 8*v7 + v8) + (1/17)*t^17*(v0 - 8*v1 + 28*v2 - 56*v3 + 70*v4 - 56*v5 + 28*v6 - 8*v7 + v8)^2 - (8/5)*t^10*(715*v0^2 + 13728*v1^2 - 84*v1*(572*v2 - 528*v3 + 275*v4 - 80*v5 + 12*v6) + 98*(396*v2^2 + 240*v3^2 + 5*v4*(5*v4 - v5) - 2*v3*(90*v4 - 16*v5 + v6) + v2*(-660*v3 + 300*v4 - 72*v5 + 8*v6)) + 64*v1*v7 - 28*v2*v7 - v1*v8 + v0*(-6435*v1 + 6*(77*(26*v2 - 26*v3 + 15*v4 - 5*v5) + 70*v6 - 6*v7) + v8)) + (2/9)*t^9*(6435*v0^2 + 96096*v1^2 - 336*v1*(858*v2 - 660*v3 + 275*v4 - 60*v5 + 6*v6) + 98*(1980*v2^2 + 720*v3^2 - 360*v3*v4 + 25*v4^2 + 32*v3*v5 + 4*v2*(-660*v3 + 225*v4 - 36*v5 + 2*v6)) + 64*v1*v7 + v0*(-51480*v1 + 84084*v2 - 72072*v3 + 34650*v4 - 9240*v5 + 1260*v6 - 72*v7 + v8)) - (28/3)*t^12*(39*v0^2 + 2*(572*v1^2 - 6*v1*(429*v2 - 528*v3 + 385*v4 + 42*(-4*v5 + v6)) + 7*(396*v2^2 + 504*v3^2 + 175*v4^2 - 105*v4*v5 + 12*v5^2 + 15*v4*v6 - 2*v5*v6 - 14*v3*(45*v4 - 16*v5 + 3*v6) + 14*v2*(-66*v3 + 45*v4 - 18*v5 + 4*v6)) + 32*v1*v7 + (-42*v2 + 24*v3 - 5*v4)*v7) - (3*v1 - 3*v2 + v3)*v8 + v0*(-429*v1 + 1001*v2 - 1287*v3 + 990*v4 - 462*v5 + 126*v6 - 18*v7 + v8)) - 8*t^14*(5*v0^2 + 2*(104*v1^2 + 7*(110*v2^2 + 240*v3^2 - 420*v3*v4 + 175*v4^2 + 224*v3*v5 - 175*v4*v5 + 40*v5^2 - 10*(7*v3 - 5*v4 + 2*v5)*v6 + 2*v6^2 + v2*(-330*v3 + 300*v4 + 56*(-3*v5 + v6))) - 2*(35*v2 - 40*v3 + 25*v4 - 8*v5 + v6)*v7 + v1*(-572*v2 + 880*v3 - 825*v4 + 480*v5 - 168*v6 + 32*v7)) - (5*v1 - 5*(2*v2 - 2*v3 + v4) + v5)*v8 + v0*(-65*v1 + 182*v2 - 286*v3 + 275*v4 - 165*v5 + 60*v6 - 12*v7 + v8)) + (56/11)*t^11*(143*v0^2 + 3432*v1^2 + 14*(924*v2^2 + 840*v3^2 + 175*v4^2 - 70*v4*v5 + 4*v5^2 - 14*v2*(132*v3 - 75*v4 + 24*v5 - 4*v6) + 5*v4*v6 - 28*v3*(30*v4 - 8*v5 + v6)) + 8*(-7*v2 + 2*v3)*v7 + v2*v8 - 2*v1*(6864*v2 - 7392*v3 + 4620*v4 - 1680*v5 + 336*v6 - 32*v7 + v8) + v0*(-1430*v1 + 3003*v2 - 6*(572*v3 - 385*v4 + 154*v5 - 35*v6 + 4*v7) + v8)) + (28/13)*t^13*(65*v0^2 + 2288*v1^2 + 14*(990*v2^2 + 1680*v3^2 + 875*v4^2 - 700*v4*v5 + 120*v5^2 - 4*v2*(660*v3 - 525*v4 + 252*v5 - 70*v6) + 150*v4*v6 - 40*v5*v6 + 2*v6^2 - 280*v3*(9*v4 - 4*v5 + v6)) - 16*v1*(715*v2 - 990*v3 + 825*v4 - 420*v5 + 126*v6 - 20*v7) - 8*(70*v2 - 60*v3 + 25*v4 - 4*v5)*v7 + 5*(-4*v1 + 6*v2 - 4*v3 + v4)*v8 + v0*(-780*v1 + 2002*v2 - 2860*v3 + 2475*v4 - 1320*v5 + 420*v6 - 72*v7 + 5*v8)) + (8/15)*t^15*(15*v0^2 + 728*v1^2 + 98*(66*v2^2 + 180*v3^2 + 175*v4^2 - 210*v4*v5 + 60*v5^2 + 75*v4*v6 - 40*v5*v6 + 6*v6^2 - 4*v3*(90*v4 - 56*v5 + 21*v6) + v2*(-220*v3 + 225*v4 - 144*v5 + 56*v6)) - 56*(21*v2 - 30*v3 + 25*v4 - 12*v5 + 3*v6)*v7 + 8*v7^2 + 7*(15*v2 - 20*v3 + 15*v4 - 6*v5 + v6)*v8 - 14*v1*(312*v2 - 528*v3 + 550*v4 - 360*v5 + 144*v6 - 32*v7 + 3*v8) + v0*(-210*v1 + 637*v2 - 1092*v3 + 1155*v4 - 770*v5 + 315*v6 - 72*v7 + 7*v8))"
            coeffs = "t u0 u1 u2 u3 u4 u5 u6 u7 u8 v0 v1 v2 v3 v4 v5 v6 v7 v8 g0 g1 g2 g3 g4 g5 g6 g7 g8 h0 h1 h2 h3 h4 h5 h6 h7 h8"
            expr_mathematica = mathematica(expr_math)
            coeffs_mathematica = var(coeffs)
            return lambdify(coeffs_mathematica, expr_mathematica)

        f_arclength = arclength()
        u0, v0, g0, h0 = cs.vertsplit(A0)
        u1, v1, g1, h1 = cs.vertsplit(A1)
        u2, v2, g2, h2 = cs.vertsplit(A2)
        u3, v3, g3, h3 = cs.vertsplit(A3)
        u4, v4, g4, h4 = cs.vertsplit(A4)
        u5, v5, g5, h5 = cs.vertsplit(A5)
        u6, v6, g6, h6 = cs.vertsplit(A6)
        u7, v7, g7, h7 = cs.vertsplit(A7)
        u8, v8, g8, h8 = cs.vertsplit(A8)

        L = f_arclength(
            xi,
            u0,
            u1,
            u2,
            u3,
            u4,
            u5,
            u6,
            u7,
            u8,
            v0,
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
            v8,
            g0,
            g1,
            g2,
            g3,
            g4,
            g5,
            g6,
            g7,
            g8,
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            h8,
        )

        f_L = cs.Function(
            "f_L",
            [xi, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [L],
            ["xi", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["L"],
        )

        # ------------------- quaternion control points to position ------------------ #

        # hodograph controlpoints
        h0 = AstB(A0, A0)
        h1 = AstB(A0, A1)
        h2 = (1 / 15) * (7 * AstB(A0, A2) + 8 * AstB(A1, A1))
        h3 = (1 / 10) * (2 * AstB(A0, A3) + 8 * AstB(A1, A2))
        h4 = (1 / 65) * (5 * AstB(A0, A4) + 32 * AstB(A1, A3) + 28 * AstB(A2, A2))
        h5 = (1 / 39) * (AstB(A0, A5) + 10 * AstB(A1, A4) + 28 * AstB(A2, A3))
        h6 = (1 / 143) * (
            AstB(A0, A6) + 16 * AstB(A1, A5) + 70 * AstB(A2, A4) + 56 * AstB(A3, A3)
        )
        h7 = (1 / 715) * (
            AstB(A0, A7) + 28 * AstB(A1, A6) + 196 * AstB(A2, A5) + 490 * AstB(A3, A4)
        )
        h8 = (1 / 6435) * (
            AstB(A0, A8)
            + 64 * AstB(A1, A7)
            + 784 * AstB(A2, A6)
            + 3136 * AstB(A3, A5)
            + 2450 * AstB(A4, A4)
        )
        h9 = (1 / 715) * (
            AstB(A1, A8) + 28 * AstB(A2, A7) + 196 * AstB(A3, A6) + 490 * AstB(A4, A5)
        )
        h10 = (1 / 143) * (
            AstB(A2, A8) + 16 * AstB(A3, A7) + 70 * AstB(A4, A6) + 56 * AstB(A5, A5)
        )
        h11 = (1 / 39) * (AstB(A3, A8) + 10 * AstB(A4, A7) + 28 * AstB(A5, A6))
        h12 = (1 / 65) * (5 * AstB(A4, A8) + 32 * AstB(A5, A7) + 28 * AstB(A6, A6))
        h13 = (1 / 10) * (2 * AstB(A5, A8) + 8 * AstB(A6, A7))
        h14 = (1 / 15) * (7 * AstB(A6, A8) + 8 * AstB(A7, A7))
        h15 = AstB(A7, A8)
        h16 = AstB(A8, A8)
        hodograph = [
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            h8,
            h9,
            h10,
            h11,
            h12,
            h13,
            h14,
            h15,
            h16,
        ]

        # curve controlpoints
        p0 = cs.SX.sym("p0", 3)
        p = cs.SX.zeros(18, 3)
        p[0, :] = p0
        for j in range(1, 18):
            # sum_hodograph = cs.SX.zeros(1,3)
            # for i in range(j):
            #    sum_hodograph += hodograph[i][1:,0].T
            p[j, :] = p[j - 1, :] + 1 / 17 * hodograph[j - 1][1:, 0].T

        f_h = cs.Function("f_h", [p0, A0, A1, A2, A3, A4, A5, A6, A7, A8], hodograph)
        f_p = cs.Function(
            "f_p",
            [p0, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [p],
            ["p0", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["p"],
        )

        # curve
        gamma = Bezier(xi, p, 17).T
        gamma_d = cs.jacobian(gamma, xi)
        gamma_dd = cs.jacobian(gamma_d, xi)
        gamma_ddd = cs.jacobian(gamma_dd, xi)
        gamma_dddd = cs.jacobian(gamma_ddd, xi)
        f_gamma = cs.Function(
            "f_gamma",
            [xi, p0, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [gamma],
            ["xi", "p0", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["gamma"],
        )
        f_gammad = cs.Function(
            "f_gammad",
            [xi, p0, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [gamma_d],
            ["xi", "p0", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["gamma_d"],
        )
        f_gammadd = cs.Function(
            "f_gammadd",
            [xi, p0, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [gamma_dd],
            ["xi", "p0", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["gamma_dd"],
        )
        f_gammaddd = cs.Function(
            "f_gammaddd",
            [xi, p0, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [gamma_ddd],
            ["xi", "p0", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["gamma_ddd"],
        )
        f_gammadddd = cs.Function(
            "f_gammadddd",
            [xi, p0, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [gamma_dddd],
            ["xi", "p0", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["gamma_dddd"],
        )

        # ------------------------------- adapted frame ------------------------------ #
        erf, X = Z_to_erf(xi, Z)

        erf_d = cs.horzcat(
            cs.jacobian(erf[:, 0], xi),
            cs.jacobian(erf[:, 1], xi),
            cs.jacobian(erf[:, 2], xi),
        )
        erf_dd = cs.horzcat(
            cs.jacobian(erf_d[:, 0], xi),
            cs.jacobian(erf_d[:, 1], xi),
            cs.jacobian(erf_d[:, 2], xi),
        )
        # erf_ddd = cs.jacobian(erf_dd, xi)

        X_d = cs.jacobian(X, xi)
        X_dd = cs.jacobian(X_d, xi)

        f_erf = cs.Function(
            "f_erf",
            [xi, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [erf],
            ["xi", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["erf"],
        )
        f_erfd = cs.Function(
            "f_erfd",
            [xi, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [erf_d],
            ["xi", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["erfd"],
        )
        f_erfdd = cs.Function(
            "f_erfdd",
            [xi, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [erf_dd],
            ["xi", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["erfdd"],
        )
        # f_erfddd = cs.Function('f_erfdd', [xi, A0, A1, A2, A3, A4], [erf_ddd], [
        #    'xi', 'A0', 'A1', 'A2', 'A3', 'A4'], ['erfddd'])

        f_X = cs.Function(
            "f_X",
            [xi, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [X],
            ["xi", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["X"],
        )
        f_Xd = cs.Function(
            "f_X",
            [xi, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [X_d],
            ["xi", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["X_d"],
        )
        f_Xdd = cs.Function(
            "f_X",
            [xi, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [X_dd],
            ["xi", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["X_dd"],
        )

        # -------------------------- dictionary of functions ------------------------- #
        func_dict = {
            "Z": f_Z,
            "Zd": f_Zd,
            "Zdd": f_Zdd,
            "Zddd": f_Zddd,
            "A": f_A,
            "gamma": f_gamma,
            "gamma_d": f_gammad,
            "gamma_dd": f_gammadd,
            "gamma_ddd": f_gammaddd,
            "gamma_dddd": f_gammadddd,
            "hodograph": f_h,
            "sigma": f_sigma,
            "sigma_d": f_sigmad,
            "sigma_dd": f_sigmadd,
            "erf": f_erf,
            "erf_d": f_erfd,
            "erf_dd": f_erfdd,
            "X": f_X,
            "X_d": f_Xd,
            "X_dd": f_Xdd,
            "L": f_L,
        }
    return func_dict


def convert_path_to_ph(path, n_segments):
    """Converts a path (decribed by a casadi function) to a ph spline (symboli
    functions and numerical evaluations)

    Args:
    path (dict): Dictionary with 1) symbolic path parameter and 2) casadi
    function for original path (maps path parameter [xi] to euclidean coordinates
    [x,y,z] of path
    n_segments (int, optional): Number of segments to discretize the path

    Returns:
    ph_path (dict): Dictionary with the following  fields-->
        - xi_grid (np.ndarray, [n_segments+1]): Grid of progress variables
        - A_rot (np.ndarray, [n_segments, 9, 4]): Matrix with the control points
        of the ROTATED quaternion polynomials
        - translation (np.ndarray, [n_segments, 3]): Matrix with euclidean coordinates
        for the position offset of every segment
    """
    rotate = True

    # ---------------------------------------------------------------------------c- #
    #                              symbolic functions                              #
    # ---------------------------------------------------------------------------- #

    # ---------------------------- PH spline functions --------------------------- #
    func_dict = ph_curve(interp_order=4)
    f_A = func_dict["A"]
    f_erf = func_dict["erf"]

    # --------------------------- get analytical curve --------------------------- #
    xi = path["xi"]
    r = path["r"]
    # r = get_path(xi)

    dr = cs.jacobian(r, xi)  # first derivative of analytical curve
    ddr = cs.jacobian(dr, xi)  # second derivative of analytical curve
    dddr = cs.jacobian(ddr, xi)  # third derivative of analytical curve
    ddddr = cs.jacobian(dddr, xi)  # fourth derivative of analytical curve
    f_r = cs.Function(
        "f_r", [xi], [r, dr, ddr, dddr, ddddr], ["xi"], ["p", "v", "a", "j", "s"]
    )

    # ---------------------------------------------------------------------------- #
    #                           numerical implementation                           #
    # ---------------------------------------------------------------------------- #

    # ------------------------ get quaternion coefficients ----------------------- #
    xi_grid = np.zeros(n_segments + 1)
    for i in range(n_segments):
        xi_grid[i + 1] = (i + 1) / (n_segments)  # * path["xi_knots"][-1]
        delta_xi = xi_grid[1]

    A0 = np.zeros((n_segments, 4))
    A1 = np.zeros((n_segments, 4))
    A2 = np.zeros((n_segments, 4))
    A3 = np.zeros((n_segments, 4))
    A4 = np.zeros((n_segments, 4))
    A5 = np.zeros((n_segments, 4))
    A6 = np.zeros((n_segments, 4))
    A7 = np.zeros((n_segments, 4))
    A8 = np.zeros((n_segments, 4))

    translation = np.zeros((n_segments, 3))
    rotation = np.zeros((n_segments, 3, 3))

    h = []
    for m in range(n_segments):

        # get hermite data and choice of interpolants
        pb, vb, ab, jb, sb = f_r(xi_grid[m])
        pe, ve, ae, je, se = f_r(xi_grid[m + 1])

        vb = vb * (delta_xi)
        ve = ve * (delta_xi)
        ab = ab * (delta_xi**2)
        ae = ae * (delta_xi**2)
        jb = jb * (delta_xi**3)
        je = je * (delta_xi**3)
        sb = sb * (delta_xi**4)
        se = se * (delta_xi**4)

        o0 = 0
        t1 = 0
        t2 = 0
        t3 = 0
        o4 = 0
        t5 = 0
        t6 = 0
        t7 = 0
        o8 = 0

        # translate hermite data into standard position
        translation[m, :] = np.squeeze(pb)
        if rotate:
            rotation[m, :, :] = tangent_to_rotation(np.squeeze(vb + ve))
        else:
            rotation[m, :, :] = np.eye(3)

        pb = pb - translation[m, :]
        pe = pe - translation[m, :]
        pb = np.matmul(np.squeeze(pb), rotation[m, :, :])
        pe = np.matmul(np.squeeze(pe), rotation[m, :, :])
        vb = np.matmul(np.squeeze(vb), rotation[m, :, :])
        ve = np.matmul(np.squeeze(ve), rotation[m, :, :])
        ab = np.matmul(np.squeeze(ab), rotation[m, :, :])
        ae = np.matmul(np.squeeze(ae), rotation[m, :, :])
        jb = np.matmul(np.squeeze(jb), rotation[m, :, :])
        je = np.matmul(np.squeeze(je), rotation[m, :, :])
        sb = np.matmul(np.squeeze(sb), rotation[m, :, :])
        se = np.matmul(np.squeeze(se), rotation[m, :, :])

        # compute quaternion polynomial coefficients
        a0, a1, a2, a3, a4, a5, a6, a7, a8 = f_A(
            pb, pe, vb, ve, ab, ae, jb, je, sb, se, o0, t1, t2, t3, o4, t5, t6, t7, o8
        )
        A0[m, :] = np.squeeze(a0)
        A1[m, :] = np.squeeze(a1)
        A2[m, :] = np.squeeze(a2)
        A3[m, :] = np.squeeze(a3)
        A4[m, :] = np.squeeze(a4)
        A5[m, :] = np.squeeze(a5)
        A6[m, :] = np.squeeze(a6)
        A7[m, :] = np.squeeze(a7)
        A8[m, :] = np.squeeze(a8)

        # get transformations for preimage
        q_rot = []
        A0_rot = np.zeros((n_segments, 4))
        A1_rot = np.zeros((n_segments, 4))
        A2_rot = np.zeros((n_segments, 4))
        A3_rot = np.zeros((n_segments, 4))
        A4_rot = np.zeros((n_segments, 4))
        A5_rot = np.zeros((n_segments, 4))
        A6_rot = np.zeros((n_segments, 4))
        A7_rot = np.zeros((n_segments, 4))
        A8_rot = np.zeros((n_segments, 4))

    for m in range(n_segments):

        # rotate back the quaternion polynomial from the standard from to original
        q_rot = Quaternion(matrix=rotation[m, :, :].T).conjugate
        A0_rot[m, :] = (q_rot * Quaternion(A0[m])).q
        A1_rot[m, :] = (q_rot * Quaternion(A1[m])).q
        A2_rot[m, :] = (q_rot * Quaternion(A2[m])).q
        A3_rot[m, :] = (q_rot * Quaternion(A3[m])).q
        A4_rot[m, :] = (q_rot * Quaternion(A4[m])).q
        A5_rot[m, :] = (q_rot * Quaternion(A5[m])).q
        A6_rot[m, :] = (q_rot * Quaternion(A6[m])).q
        A7_rot[m, :] = (q_rot * Quaternion(A7[m])).q
        A8_rot[m, :] = (q_rot * Quaternion(A8[m])).q
        Rot = np.zeros((n_segments, 3, 3))
        Rot[0] = np.eye(3)

        # rotate adapted frame around e1 (roll) for continuity in e2 and e3
        if m > 0:

            Rk = np.squeeze(
                f_erf(
                    0,
                    A0_rot[m],
                    A1_rot[m],
                    A2_rot[m],
                    A3_rot[m],
                    A4_rot[m],
                    A5_rot[m],
                    A6_rot[m],
                    A7_rot[m],
                    A8_rot[m],
                )
            )
            Rk_old = np.squeeze(
                f_erf(
                    1,
                    A0_rot[m - 1],
                    A1_rot[m - 1],
                    A2_rot[m - 1],
                    A3_rot[m - 1],
                    A4_rot[m - 1],
                    A5_rot[m - 1],
                    A6_rot[m - 1],
                    A7_rot[m - 1],
                    A8_rot[m - 1],
                )
            )

            # roll --> rotation around e1
            theta = angle_between(Rk[:, 1], Rk_old[:, 1])
            Rot[m, :, :] = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)],
                ]
            )

            # roll clockwise or anti clockwise
            if np.linalg.norm(np.matmul(Rk, Rot[m]) - Rk_old) < np.linalg.norm(
                np.matmul(Rk, Rot[m].T) - Rk_old
            ):
                qrot = Quaternion(matrix=Rot[m])
            else:
                qrot = Quaternion(matrix=Rot[m].T)
            # Rq = ((Quaternion(matrix=Rk)*qrot)).rotation_matrix
            # print(Rq - Rk_old)

            # rotate ctrlpts of quaternion polynomial
            A0_rot[m, :] = (Quaternion(A0_rot[m, :]) * qrot).q
            A1_rot[m, :] = (Quaternion(A1_rot[m, :]) * qrot).q
            A2_rot[m, :] = (Quaternion(A2_rot[m, :]) * qrot).q
            A3_rot[m, :] = (Quaternion(A3_rot[m, :]) * qrot).q
            A4_rot[m, :] = (Quaternion(A4_rot[m, :]) * qrot).q
            A5_rot[m, :] = (Quaternion(A5_rot[m, :]) * qrot).q
            A6_rot[m, :] = (Quaternion(A6_rot[m, :]) * qrot).q
            A7_rot[m, :] = (Quaternion(A7_rot[m, :]) * qrot).q
            A8_rot[m, :] = (Quaternion(A8_rot[m, :]) * qrot).q

    A_rot = np.stack(
        [A0_rot, A1_rot, A2_rot, A3_rot, A4_rot, A5_rot, A6_rot, A7_rot, A8_rot]
    ).transpose(1, 0, 2)

    ph_path = {
        "xi_grid": xi_grid,
        "A_rot": A_rot,
        "translation": translation,
    }
    return ph_path


def evaluate_ph_point(xi, ph_func, ph_path):
    """Given a progress value, a symbolic function for a ph path and the
    associated coefficients, computes the parametric variables

    Args:
        xi (float or cs.SX.sym): Value of progress variable to evaluate
        ph_func (cs.Function): Symbolic function to evaluate the ph
        ph_path (dict): Dictionary with grid, translation and preimage
        coefficients of the PH path

    Returns:
        gamma(np.ndarray or cs.SX.sym): Euclidean coordinates of the path [x,y,z]
        in xi
        sigma(np.ndarray or cs.SX.sym): Parametric speed in xi
        erf(np.ndarray or cs.SX.sym): Adapted frame in xi
        X(np.ndarray or cs.SX.sym): Angular veloctiy of adapted frame in xi
    """
    xi_grid = ph_path["xi_grid"]
    translation = ph_path["translation"]
    A_rot = ph_path["A_rot"]
    A0, A1, A2, A3, A4, A5, A6, A7, A8 = np.squeeze(np.hsplit(A_rot, 9))
    gamma_k, sigma_k, erf_k, X_k, _ = ph_func(
        xi, xi_grid, translation, A0, A1, A2, A3, A4, A5, A6, A7, A8
    )
    return gamma_k, sigma_k, erf_k, X_k


def evaluate_ph_spline(xi_eval, ph_func, ph_path, planar=False, visualize=False):
    """Given an array of progress values, a symbolic function for a ph path and
    the associated coefficients, computes the parametric variables at every
    point of the array

    Args:
        xi_eval (float or cs.SX.sym): Value of progress variable to evaluate
        ph_func (cs.Function): Symbolic function to evaluate the ph
        ph_path (dict): Dictionary with grid, translation and preimage
        coefficients of the PH path
        planar (bool, optional): Boolean to reduce to 2D. Defaults to False.
        visualize (bool, optional): Boolean to visualize. Defaults to False.

    Returns:
        ph_spline (dict): Dictionary with the parametric variables
    """

    xi_grid = ph_path
    n_eval = xi_eval.shape[0]

    gamma = np.zeros((3, n_eval))
    sigma = np.zeros(n_eval)
    erf = np.zeros((3, 3, n_eval))
    X = np.zeros((3, n_eval))
    for k in range(n_eval):
        gamma_k, sigma_k, erf_k, X_k = evaluate_ph_point(xi_eval[k], ph_func, ph_path)
        gamma[:, k] = np.squeeze(gamma_k)
        sigma[k] = np.squeeze(sigma_k)
        erf[:, :, k] = np.squeeze(erf_k)
        X[:, k] = np.squeeze(X_k)

        erf[:, 1:, k] *= -1
        X[2, k] *= -1

    if planar:
        gamma = gamma[:2, :]
        erf = erf[:2, :2, :]
        X = X[2, :]

    if visualize and planar:
        scale = 0.2
        e1 = erf[:, 0, :]
        e2 = erf[:, 1, :]
        plt.figure()
        plt.plot(gamma[0], gamma[1], "k--")

        for k in range(0, gamma.shape[1], int(gamma.shape[1] / 10)):
            plt.plot(
                [gamma[0, k], gamma[0, k] + scale * e1[0, k]],
                [gamma[1, k], gamma[1, k] + scale * e1[1, k]],
                "r-",
            )
            plt.plot(
                [gamma[0, k], gamma[0, k] + scale * e2[0, k]],
                [gamma[1, k], gamma[1, k] + scale * e2[1, k]],
                "g-",
            )

        plt.plot(gamma[0, 0], gamma[1, 0], "og")
        plt.plot(gamma[0, -1], gamma[1, -1], "or")

        # pt = np.squeeze(ph_funcs["gamma"](0.16))
        # pt2 = np.squeeze(ph_funcs["gamma"](0.16 + 0.01))
        # plt.plot(pt[0], pt[1], "ko")
        # plt.plot(pt2[0], pt2[1], "bo")

        plt.axis("equal")
        # mng = plt.get_current_fig_manager()
        # mng.full_screen_toggle()

        plt.figure()
        plt.subplot(411)
        plt.plot(xi_eval, sigma)
        plt.ylabel(r"$\sigma$")
        plt.subplot(412)
        plt.plot(xi_eval, e1.T)
        plt.ylabel(r"$e_1$")
        plt.subplot(413)
        plt.plot(xi_eval, e2.T)
        plt.ylabel(r"$e_2$")
        plt.subplot(414)
        plt.plot(xi_eval, X.T)
        plt.ylabel(r"$\chi$")
        # mng = plt.get_current_fig_manager()
        # mng.full_screen_toggle()
        plt.show()

    elif visualize and planar == False:
        # print("Not implemented. TODO!")
        scale = 0.2
        e1 = erf[:, 0, :]
        e2 = erf[:, 1, :]
        e3 = erf[:, 2, :]
        ax = plt.figure().add_subplot(111, projection="3d")
        ax.plot(gamma[0], gamma[1], gamma[2], "k--")

        for k in range(0, gamma.shape[1], int(gamma.shape[1] / 20)):
            ax.plot(
                [gamma[0, k], gamma[0, k] + scale * e1[0, k]],
                [gamma[1, k], gamma[1, k] + scale * e1[1, k]],
                [gamma[2, k], gamma[2, k] + scale * e1[2, k]],
                "r-",
            )
            ax.plot(
                [gamma[0, k], gamma[0, k] + scale * e2[0, k]],
                [gamma[1, k], gamma[1, k] + scale * e2[1, k]],
                [gamma[2, k], gamma[2, k] + scale * e2[2, k]],
                "g-",
            )
            ax.plot(
                [gamma[0, k], gamma[0, k] + scale * e3[0, k]],
                [gamma[1, k], gamma[1, k] + scale * e3[1, k]],
                [gamma[2, k], gamma[2, k] + scale * e3[2, k]],
                "b-",
            )

        ax.plot(gamma[0, 0], gamma[1, 0], gamma[2, 0], "og")
        ax.plot(gamma[0, -1], gamma[1, -1], gamma[2, -1], "or")
        ax = axis_equal(gamma[0, :], gamma[1, :], gamma[2, :], ax=ax)

        plt.figure()
        plt.subplot(611)
        plt.plot(xi_eval, sigma)
        plt.ylabel(r"$\sigma$")
        plt.subplot(612)
        plt.plot(xi_eval, e1.T)
        plt.ylabel(r"$e_1$")
        plt.subplot(613)
        plt.plot(xi_eval, e2.T)
        plt.ylabel(r"$e_2$")
        plt.subplot(614)
        plt.plot(xi_eval, e3.T)
        plt.ylabel(r"$e_3$")
        plt.subplot(615)
        plt.plot(xi_eval, X.T)
        plt.ylabel(r"$\chi$")
        # plt.subplot(616)
        # plt.plot(xi_eval, curvature)
        # plt.ylabel(r"$\kappa$")
        plt.show()

    ph_spline = {"xi": xi_eval, "gamma": gamma, "erf": erf, "sigma": sigma, "X": X}
    return ph_spline


def generate_interpolators(ph_spline):
    xi_interp = ph_spline["xi"]
    sigma = ph_spline["sigma"]
    gamma = ph_spline["gamma"]
    erf = ph_spline["erf"]
    X = ph_spline["X"]

    planar = False
    if gamma.shape[0] == 2:
        planar = True

    fi_sigma = cs.interpolant("fi_sigma", "bspline", [xi_interp], sigma)
    fi_gammax = cs.interpolant("fi_gammax", "bspline", [xi_interp], gamma[0, :])
    fi_gammay = cs.interpolant("fi_gammay", "bspline", [xi_interp], gamma[1, :])
    fi_e1x = cs.interpolant("fi_e1x", "bspline", [xi_interp], erf[0, 0, :])
    fi_e1y = cs.interpolant("fi_e1y", "bspline", [xi_interp], erf[1, 0, :])
    fi_e2x = cs.interpolant("fi_e2x", "bspline", [xi_interp], erf[0, 1, :])
    fi_e2y = cs.interpolant("fi_e2y", "bspline", [xi_interp], erf[1, 1, :])
    if planar:
        fi_X = cs.interpolant("fi_X3", "bspline", [xi_interp], X)
    else:
        fi_gammaz = cs.interpolant("fi_gammaz", "bspline", [xi_interp], gamma[2, :])
        fi_e1z = cs.interpolant("fi_e1y", "bspline", [xi_interp], erf[2, 0, :])
        fi_e2z = cs.interpolant("fi_e2y", "bspline", [xi_interp], erf[2, 1, :])
        fi_e3x = cs.interpolant("fi_e3x", "bspline", [xi_interp], erf[0, 2, :])
        fi_e3y = cs.interpolant("fi_e3y", "bspline", [xi_interp], erf[1, 2, :])
        fi_e3z = cs.interpolant("fi_e3y", "bspline", [xi_interp], erf[2, 2, :])
        fi_X1 = cs.interpolant("fi_X1", "bspline", [xi_interp], X[0, :])
        fi_X2 = cs.interpolant("fi_X2", "bspline", [xi_interp], X[1, :])
        fi_X3 = cs.interpolant("fi_X3", "bspline", [xi_interp], X[2, :])

    xi = cs.MX.sym("xi")
    if planar:
        gamma = cs.vertcat(fi_gammax(xi), fi_gammay(xi), 0)
        erf = cs.horzcat(
            cs.vertcat(fi_e1x(xi), fi_e1y(xi), 0),
            cs.vertcat(fi_e2x(xi), fi_e2y(xi), 0),
            cs.DM([0, 0, 1]),
        )

        X = cs.vertcat(0, 0, fi_X(xi))

        fi_q = 0
        fi_gamma_d = 0
        fi_gamma_dd = 0
        fi_X_d = 0
    else:
        gamma = cs.vertcat(fi_gammax(xi), fi_gammay(xi), fi_gammaz(xi))
        erf = cs.horzcat(
            cs.vertcat(fi_e1x(xi), fi_e1y(xi), fi_e1z(xi)),
            cs.vertcat(fi_e2x(xi), fi_e2y(xi), fi_e2z(xi)),
            cs.vertcat(fi_e3x(xi), fi_e3y(xi), fi_e3z(xi)),
        )
        X = cs.vertcat(fi_X1(xi), fi_X2(xi), fi_X3(xi))

        # ---------------------------------------------------------------------------- #
        gamma_d = cs.jacobian(gamma, xi)
        gamma_dd = cs.jacobian(gamma_d, xi)
        X_d = cs.jacobian(X, xi)
        sigma_d = cs.jacobian(fi_sigma(xi), xi)

        e1_d = cs.jacobian(cs.vertcat(fi_e1x(xi), fi_e1y(xi), fi_e1z(xi)), xi)
        e2_d = cs.jacobian(cs.vertcat(fi_e2x(xi), fi_e2y(xi), fi_e2z(xi)), xi)
        e3_d = cs.jacobian(cs.vertcat(fi_e3x(xi), fi_e3y(xi), fi_e3z(xi)), xi)
        erf_d = cs.horzcat(e1_d, e2_d, e3_d)

        q = []
        for k in range(ph_spline["erf"].shape[2]):
            q += [rotation_to_quaternion(ph_spline["erf"][:, :, k])]
            if k > 0:
                if np.abs(q[-1][-1] - q[-2][-1]) > 0.1:  # avoid flipping quaterions
                    q[-1] = -q[-1]
        q = np.squeeze(q)

        fi_qw = cs.interpolant("fi_qw", "bspline", [ph_spline["xi"]], q[:, 0])
        fi_qx = cs.interpolant("fi_qx", "bspline", [ph_spline["xi"]], q[:, 1])
        fi_qy = cs.interpolant("fi_qy", "bspline", [ph_spline["xi"]], q[:, 2])
        fi_qz = cs.interpolant("fi_qz", "bspline", [ph_spline["xi"]], q[:, 3])
        q = cs.vertcat(fi_qx(xi), fi_qy(xi), fi_qz(xi), fi_qw(xi))
        fi_q = cs.Function("fi_q", [xi], [q], ["xi"], ["q"])
        # ---------------------------------------------------------------------------- #

    fi_gamma = cs.Function("fi_gamma", [xi], [gamma], ["xi"], ["gamma"])
    fi_gamma_d = cs.Function("fi_gamma_d", [xi], [gamma_d], ["xi"], ["gamma_d"])
    fi_gamma_dd = cs.Function("fi_gamma_dd", [xi], [gamma_dd], ["xi"], ["gamma_dd"])
    fi_erf = cs.Function("fi_erf", [xi], [erf], ["xi"], ["erf"])
    fi_erf_d = cs.Function("fi_erf", [xi], [erf_d], ["xi"], ["erf_d"])
    fi_X = cs.Function("fi_X", [xi], [X], ["xi"], ["X"])
    fi_X_d = cs.Function("fi_X", [xi], [X_d], ["xi"], ["X_d"])
    fi_sigma_d = cs.Function("fi_sigma_d", [xi], [sigma_d], ["xi"], ["sigma_d"])

    ph_interpolator = {
        "xi": xi_interp,
        "gamma": fi_gamma,
        "gamma_d": fi_gamma_d,
        "gamma_dd": fi_gamma_dd,
        "sigma": fi_sigma,
        "sigma_d": fi_sigma_d,
        "erf": fi_erf,
        "erf_d": fi_erf_d,
        "X": fi_X,
        "X_d": fi_X_d,
        "q": fi_q,
    }

    # ---------------------------------------------------------------------------- #
    ph_interpolator["p"] = ph_interpolator["gamma"]
    ph_interpolator["v"] = ph_interpolator["gamma_d"]
    ph_interpolator["a"] = ph_interpolator["gamma_dd"]

    ###
    q_swapped = cs.vertcat(q[3], q[0], q[1], q[2])
    dqdxi = cs.jacobian(q_swapped, xi)
    w = 2 * quat_mult(dqdxi, quat_conj(q_swapped))[1:]
    fi_w = cs.Function("fi_w", [xi], [w], ["xi"], ["w"])
    fi_r = cs.Function("fi_w", [xi], [cs.jacobian(w, xi)], ["xi"], ["r"])
    ###
    ph_interpolator["w"] = fi_w
    ph_interpolator["r"] = fi_r

    # ---------------------------------------------------------------------------- #
    return ph_interpolator

    # ---------------------------------------------------------------------------- #


def time_interpolators(ph_interp, t_total):

    # create timing law
    time = np.zeros(ph_interp["xi"].shape[0])
    for k in range(ph_interp["xi"].shape[0] - 1):
        xidot = 1 / t_total  # np.squeeze(1 / ph_spline["sigma"][k])
        delta_xi = ph_interp["xi"][k + 1] - ph_interp["xi"][k]
        time[k + 1] = time[k] + delta_xi / xidot

    fi_t2xi = cs.interpolant("fi_t", "bspline", [time], ph_interp["xi"])
    ph_time_interpolator = dict.fromkeys(["time", "xi", "p", "v", "a", "q", "w"])

    t = cs.MX.sym("t")
    ph_time_interpolator["xi"] = cs.Function("f_time", [t], [fi_t2xi(t)], ["t"], ["xi"])
    ph_time_interpolator["time"] = time
    pos = ph_interp["gamma"](fi_t2xi(t))
    vel = cs.jacobian(pos, t)
    acc = cs.jacobian(vel, t)
    q = ph_interp["q"](fi_t2xi(t))

    # w = ph_interp["X"](fi_t2xi(t))
    ##########################################
    q_swapped = cs.vertcat(q[3], q[0], q[1], q[2])
    dqdt = cs.jacobian(q_swapped, t)
    w = 2 * quat_mult(dqdt, quat_conj(q_swapped))[1:]
    # erf = quaternion_to_rotation(q)
    # # Darboux vector
    # e1_d = cs.jacobian(e1, t)  # .T
    # e2_d = cs.jacobian(e2, t)  # .T
    # e3_d = cs.jacobian(e3, t)  # .T

    # X1 = cs.dot(e2_d, e3)
    # X2 = cs.dot(e3_d, e1)
    # X3 = cs.dot(e1_d, e2)
    # # omega =  X1*e1+X2*e2+X3*e3
    # X = cs.horzcat(X1, X2, X3)

    ##########################################
    r = cs.jacobian(w, t)

    ph_time_interpolator["p"] = cs.Function("f_p", [t], [pos], ["t"], ["p"])
    ph_time_interpolator["v"] = cs.Function("f_v", [t], [vel], ["t"], ["v"])
    ph_time_interpolator["a"] = cs.Function("f_a", [t], [acc], ["t"], ["a"])
    ph_time_interpolator["w"] = cs.Function("f_w", [t], [w], ["t"], ["w"])
    ph_time_interpolator["r"] = cs.Function("f_r", [t], [r], ["t"], ["r"])
    ph_time_interpolator["q"] = cs.Function("f_q", [t], [q], ["t"], ["q"])

    # TODO: When using this to create the dual quaternions with an integration,
    # attention must be paid. Either use euler with very small step size or RK4
    # (whose coding is not trivial)
    return ph_time_interpolator


def xi_interpolators(ph_interp, xi_total):

    # create timing law
    ph_xi_interpolator = dict.fromkeys(["xi", "p", "v", "a", "q", "w"])

    # interpolator for euclidean variables
    ph_xi_interpolator["xi"] = ph_interp["xi"]
    xi = cs.MX.sym("xi")
    p = ph_interp["gamma"](xi)
    v = cs.jacobian(p, xi)
    a = cs.jacobian(v, xi)
    q = ph_interp["q"](xi)
    ##########################################
    q_swapped = cs.vertcat(q[3], q[0], q[1], q[2])
    dqdt = cs.jacobian(q_swapped, xi)
    w = 2 * quat_mult(dqdt, quat_conj(q_swapped))[1:]
    ##########################################
    r = cs.jacobian(w, xi)

    ph_xi_interpolator["p"] = cs.Function("f_p", [xi], [p], ["xi"], ["p"])
    ph_xi_interpolator["v"] = cs.Function("f_v", [xi], [v], ["xi"], ["v"])
    ph_xi_interpolator["a"] = cs.Function("f_a", [xi], [a], ["xi"], ["a"])
    ph_xi_interpolator["w"] = cs.Function("f_w", [xi], [w], ["xi"], ["w"])
    ph_xi_interpolator["r"] = cs.Function("f_r", [xi], [r], ["xi"], ["r"])
    ph_xi_interpolator["q"] = cs.Function("f_q", [xi], [q], ["xi"], ["q"])

    # TODO: When using this to create the dual quaternions with an integration,
    # attention must be paid. Either use euler with very small step size or RK4
    # (whose coding is not trivial)
    return ph_xi_interpolator


def test_interpolators(ph_time_interp, ph_spline, variable):
    time_eval = np.linspace(
        ph_time_interp[variable][0],
        ph_time_interp[variable][-1],
        len(ph_time_interp[variable]) * 10,
    )
    for k in range(len(time_eval) - 1):

        # get time
        time = time_eval[k]  # ph_time_interp["time"][k]
        time_next = time_eval[k + 1]  # ph_time_interp["time"][k + 1]
        if k == 0:
            time += 1e-6
            p = np.squeeze(ph_time_interp["p"](time))
            v = np.squeeze(ph_time_interp["v"](time))
            q = np.squeeze(ph_time_interp["q"](time))
            w = np.squeeze(ph_time_interp["w"](time))

            dqd, dtd = euclidean_to_dq(p, v, q, w)
            # dqd.normalize()

            P = [p]
            V = [v]
            Q = [q]
            W = [w]
            A = []
            R = []
            T = [time]

        d_time = time_next - time

        # get inputs
        a = np.squeeze(ph_time_interp["a"](time))
        r = np.squeeze(ph_time_interp["r"](time))

        # forward integrate
        # x_next = f_kinematic(d_time, x=np.hstack([p, v, q, w]), u=np.hstack([a, r]))
        # p = x_next[:3]
        # v = x_next[3:6]
        # q = x_next[6:10]
        # w = x_next[10:13]

        dqd_next, dtd_next, _ = f_kinematic_dq(d_time, dqd, dtd, a, r)
        p, v, q, w = dq_to_euclidean(dqd_next, dtd_next)
        # p = np.squeeze(ph_time_interp["p"](time_next))
        # v = np.squeeze(ph_time_interp["v"](time_next))
        # q = np.squeeze(ph_time_interp["q"](time_next))
        # w = np.squeeze(ph_time_interp["w"](time_next))
        dqd = dqd_next.copy()
        dtd = dtd_next.copy()

        # store results
        P += [p]
        V += [v]
        A += [a]
        Q += [q]
        W += [w]
        R += [r]
        T += [time]

    P = np.squeeze([P])
    V = np.squeeze([V])
    A = np.squeeze([A])
    Q = np.squeeze([Q])
    W = np.squeeze([W])
    R = np.squeeze([R])

    t_vec = np.squeeze(T)  # ph_time_interp["time"]
    plt.figure()
    plt.subplot(611)
    plt.plot(t_vec, P)
    plt.ylabel(r"p")
    plt.subplot(612)
    plt.plot(t_vec, V)
    plt.ylabel(r"v")
    plt.subplot(613)
    plt.plot(t_vec[:-1], A)
    plt.ylabel(r"a")
    plt.subplot(614)
    plt.plot(t_vec, Q)
    plt.ylabel(r"q")
    plt.subplot(615)
    plt.plot(t_vec, W)
    plt.ylabel(r"$\omega$")
    plt.subplot(616)
    plt.plot(t_vec[:-1], R)
    plt.ylabel(r"r")
    if variable == "time":
        plt.xlabel(r"t")
    else:
        plt.xlabel(r"$\xi$")

    ###################
    q = []
    for k in range(ph_spline["erf"].shape[2]):
        q += [rotation_to_quaternion(ph_spline["erf"][:, :, k])]
        if k > 0:
            if np.abs(q[-1][-1] - q[-2][-1]) > 0.1:  # avoid flipping quaterions
                q[-1] = -q[-1]
    ph_spline_q = np.squeeze(q)

    # ph_spline_q = []
    # for k in range(ph_spline["erf"].shape[2]):
    #     erf = ph_spline["erf"][:, :, k]
    #     ph_spline_q += [rotation_to_quaternion(erf, swap=True)]

    plt.figure()
    plt.subplot(211)
    plt.plot(ph_spline["xi"], ph_spline["gamma"].T, "--")
    plt.plot(t_vec / t_vec[-1], P[:, :3])
    plt.subplot(212)
    plt.plot(ph_spline["xi"], ph_spline_q, "--")
    plt.plot(t_vec / t_vec[-1], Q)
    plt.suptitle("Drift: interp vs original PH spline")
    plt.show()
    ###################

    return t_vec, P


# ---------------------------------------------------------------------------- #


def get_parametric_function(n_grid):
    """Generates a symbolic (casadi) parametric function that given a location for
    the progress variable (xi) and the coefficients respective to a PH path
    (grid, translation, quat poly coeffs), outputs the parametric variables
    (gamma,sigma,erf,X)

    Args:
        n_grid (int): Number of elements in the grid

    Returns:
        ph_func (cs.Function): Symbolic function to evaluate parametric variables
    """

    # -------------------------- segment identification -------------------------- #
    # ph spline constants
    xi = cs.SX.sym("xi")
    xi_grid = cs.SX.sym("xi_grid", n_grid)
    TR = cs.SX.sym("TR", n_grid - 1, 3)
    A0 = cs.SX.sym("A0", n_grid - 1, 4)
    A1 = cs.SX.sym("A1", n_grid - 1, 4)
    A2 = cs.SX.sym("A2", n_grid - 1, 4)
    A3 = cs.SX.sym("A3", n_grid - 1, 4)
    A4 = cs.SX.sym("A4", n_grid - 1, 4)
    A5 = cs.SX.sym("A5", n_grid - 1, 4)
    A6 = cs.SX.sym("A6", n_grid - 1, 4)
    A7 = cs.SX.sym("A7", n_grid - 1, 4)
    A8 = cs.SX.sym("A8", n_grid - 1, 4)

    # segment variables
    xi_seg = xi
    TR_seg = TR[0, :]
    A0_seg = A0[0, :]
    A1_seg = A1[1, :]
    A2_seg = A2[2, :]
    A3_seg = A3[3, :]
    A4_seg = A4[4, :]
    A5_seg = A5[5, :]
    A6_seg = A6[6, :]
    A7_seg = A7[7, :]
    A8_seg = A8[8, :]

    for k in range(n_grid - 1):
        xi_segment = (xi - xi_grid[k]) / (xi_grid[k + 1] - xi_grid[k])
        xi_seg = cs.if_else(
            cs.logic_and(xi_grid[k] <= xi, xi < xi_grid[k + 1]), xi_segment, xi_seg
        )
        TR_seg = cs.if_else(
            cs.logic_and(xi_grid[k] <= xi, xi < xi_grid[k + 1]), TR[k, :], TR_seg
        )
        A0_seg = cs.if_else(
            cs.logic_and(xi_grid[k] <= xi, xi < xi_grid[k + 1]), A0[k, :], A0_seg
        )
        A1_seg = cs.if_else(
            cs.logic_and(xi_grid[k] <= xi, xi < xi_grid[k + 1]), A1[k, :], A1_seg
        )
        A2_seg = cs.if_else(
            cs.logic_and(xi_grid[k] <= xi, xi < xi_grid[k + 1]), A2[k, :], A2_seg
        )
        A3_seg = cs.if_else(
            cs.logic_and(xi_grid[k] <= xi, xi < xi_grid[k + 1]), A3[k, :], A3_seg
        )
        A4_seg = cs.if_else(
            cs.logic_and(xi_grid[k] <= xi, xi < xi_grid[k + 1]), A4[k, :], A4_seg
        )
        A5_seg = cs.if_else(
            cs.logic_and(xi_grid[k] <= xi, xi < xi_grid[k + 1]), A5[k, :], A5_seg
        )
        A6_seg = cs.if_else(
            cs.logic_and(xi_grid[k] <= xi, xi < xi_grid[k + 1]), A6[k, :], A6_seg
        )
        A7_seg = cs.if_else(
            cs.logic_and(xi_grid[k] <= xi, xi < xi_grid[k + 1]), A7[k, :], A7_seg
        )
        A8_seg = cs.if_else(
            cs.logic_and(xi_grid[k] <= xi, xi < xi_grid[k + 1]), A8[k, :], A8_seg
        )

    xi_seg = cs.if_else(xi <= xi_grid[0], xi_grid[0], xi_seg)
    xi_seg = cs.if_else(xi >= xi_grid[-1], xi_grid[-1], xi_seg)
    TR_seg = cs.if_else(xi <= xi_grid[0], TR[0, :], TR_seg)
    TR_seg = cs.if_else(xi >= xi_grid[-1], TR[-1, :], TR_seg)
    A0_seg = cs.if_else(xi <= xi_grid[0], A0[0, :], A0_seg)
    A0_seg = cs.if_else(xi >= xi_grid[-1], A0[-1, :], A0_seg)
    A1_seg = cs.if_else(xi <= xi_grid[0], A1[0, :], A1_seg)
    A1_seg = cs.if_else(xi >= xi_grid[-1], A1[-1, :], A1_seg)
    A2_seg = cs.if_else(xi <= xi_grid[0], A2[0, :], A2_seg)
    A2_seg = cs.if_else(xi >= xi_grid[-1], A2[-1, :], A2_seg)
    A3_seg = cs.if_else(xi <= xi_grid[0], A3[0, :], A3_seg)
    A3_seg = cs.if_else(xi >= xi_grid[-1], A3[-1, :], A3_seg)
    A4_seg = cs.if_else(xi <= xi_grid[0], A4[0, :], A4_seg)
    A4_seg = cs.if_else(xi >= xi_grid[-1], A4[-1, :], A4_seg)
    A5_seg = cs.if_else(xi <= xi_grid[0], A5[0, :], A5_seg)
    A5_seg = cs.if_else(xi >= xi_grid[-1], A5[-1, :], A5_seg)
    A6_seg = cs.if_else(xi <= xi_grid[0], A6[0, :], A6_seg)
    A6_seg = cs.if_else(xi >= xi_grid[-1], A6[-1, :], A6_seg)
    A7_seg = cs.if_else(xi <= xi_grid[0], A7[0, :], A7_seg)
    A7_seg = cs.if_else(xi >= xi_grid[-1], A7[-1, :], A7_seg)
    A8_seg = cs.if_else(xi <= xi_grid[0], A8[0, :], A8_seg)
    A8_seg = cs.if_else(xi >= xi_grid[-1], A8[-1, :], A8_seg)

    # --------------------------- parametric variables --------------------------- #
    func_dict = ph_curve(interp_order=4)
    A_coeff = cs.vertcat(
        A0_seg,
        A1_seg,
        A2_seg,
        A3_seg,
        A4_seg,
        A5_seg,
        A6_seg,
        A7_seg,
        A8_seg,
    )

    gamma = (
        func_dict["gamma"](
            xi_seg,
            cs.DM.zeros(3),
            A0_seg,
            A1_seg,
            A2_seg,
            A3_seg,
            A4_seg,
            A5_seg,
            A6_seg,
            A7_seg,
            A8_seg,
        )
        + TR_seg.T
    )

    erf = func_dict["erf"](
        xi_seg,
        A0_seg,
        A1_seg,
        A2_seg,
        A3_seg,
        A4_seg,
        A5_seg,
        A6_seg,
        A7_seg,
        A8_seg,
    )

    gamma_d = cs.jacobian(gamma, xi)
    sigma = cs.sqrt(gamma_d[0] ** 2 + gamma_d[1] ** 2 + gamma_d[2] ** 2)
    # sigma = func_dict["sigma"](
    #     xi_seg,
    #     A0_seg,
    #     A1_seg,
    #     A2_seg,
    #     A3_seg,
    #     A4_seg,
    #     A5_seg,
    #     A6_seg,
    #     A7_seg,
    #     A8_seg,
    # )

    e1, e2, e3 = cs.horzsplit(erf)
    e1_d = cs.jacobian(e1, xi)
    e2_d = cs.jacobian(e2, xi)
    e3_d = cs.jacobian(e3, xi)
    X1 = cs.dot(e2_d, e3)
    X2 = cs.dot(e3_d, e1)
    X3 = cs.dot(e1_d, e2)
    # Z = f_Preimage(XI)
    # Z_d = cs.jacobian(Z, XI)
    # u, v, g, h = cs.vertsplit(Z)
    # u_d, v_d, g_d, h_d = cs.vertsplit(Z_d)
    # X3 = (
    #     2
    #     * (u * h_d - u_d * h - v * g_d + v_d * g)
    #     / (u**2 + v**2 + g**2 + h**2)
    # )
    X = cs.horzcat(X1, X2, X3)
    # X = func_dict["X"](
    #     xi_seg,
    #     A0_seg,
    #     A1_seg,
    #     A2_seg,
    #     A3_seg,
    #     A4_seg,
    #     A5_seg,
    #     A6_seg,
    #     A7_seg,
    #     A8_seg,
    # )

    # functions
    ph_func = cs.Function(
        "f_ph",
        [
            xi,
            xi_grid,
            TR,
            A0,
            A1,
            A2,
            A3,
            A4,
            A5,
            A6,
            A7,
            A8,
        ],
        [gamma, sigma, erf, X, A_coeff],
        [
            "xi",
            "xi_grid",
            "TR",
            "A0",
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A6",
            "A7",
            "A8",
        ],
        ["gamma", "sigma", "erf", "X", "A_coeff"],
    )

    return ph_func


# -------------------------- visualization functions ------------------------- #
def visualize_interpolators(ph_interp, ph_time_interp):
    xi_eval = np.linspace(1e-6, 1 - 1e-6, ph_interp["xi"].shape[0] * 10)
    time_eval = np.linspace(
        ph_time_interp["time"][0] + 1e-6,
        ph_time_interp["time"][-1] - 1e-6,
        ph_interp["xi"].shape[0] * 10,
    )

    pd = np.zeros((3, xi_eval.shape[0]))
    qd = np.zeros((4, xi_eval.shape[0]))
    wd = np.zeros((3, xi_eval.shape[0]))
    erf = np.zeros((3, 3, xi_eval.shape[0]))
    rd = np.zeros((3, xi_eval.shape[0]))

    pd_t = np.zeros((3, xi_eval.shape[0]))
    qd_t = np.zeros((4, xi_eval.shape[0]))
    wd_t = np.zeros((3, xi_eval.shape[0]))
    rd_t = np.zeros((3, xi_eval.shape[0]))
    erf_t = np.zeros((3, 3, xi_eval.shape[0]))

    for k, (xik, tk) in enumerate(zip(xi_eval, time_eval)):

        dqd = np.squeeze(ph_interp["dqd"](xik))
        dtd = np.squeeze(ph_interp["dtd"](xik))
        dtd_d = np.squeeze(ph_interp["dtd_d"](xik))

        pd[:, k] = DualQuaternion.from_vector(dqd).to_pose()[:3]
        qd[:, k] = DualQuaternion.from_vector(dqd).to_pose()[3:]
        wd[:, k] = DualQuaternion.from_vector(dtd).q_rot.q[:3]
        rd[:, k] = DualQuaternion.from_vector(dtd_d).q_rot.q[:3]
        erf[:, :, k] = quaternion_to_rotation(
            qd[:, k], swap=True
        )  # ph_interp['erf'](xik)

        dqd_t = np.squeeze(ph_time_interp["dqd"](tk))
        dtd_t = np.squeeze(ph_time_interp["dtd"](tk))
        dtd_d_t = np.squeeze(ph_time_interp["dtd_d"](tk))

        pd_t[:, k] = DualQuaternion.from_vector(dqd_t).to_pose()[:3]
        qd_t[:, k] = DualQuaternion.from_vector(dqd_t).to_pose()[3:]
        wd_t[:, k] = DualQuaternion.from_vector(dtd_t).q_rot.q[:3]
        rd_t[:, k] = DualQuaternion.from_vector(dtd_d_t).q_rot.q[:3]

    ax = plt.figure().add_subplot(131, projection="3d")
    ax = visualize_path_with_frames(pd=pd, erf=erf, scale=0.1, ax=ax)

    ax.plot(pd[0, 0], pd[1, 0], pd[2, 0], "og")
    ax.plot(pd[0, -1], pd[1, -1], pd[2, -1], "or")

    plt.subplot(432)
    plt.plot(xi_eval, pd.T)
    plt.ylabel(r"$p_{d}$")
    plt.subplot(433)
    plt.plot(time_eval, pd_t.T)
    plt.ylabel(r"$p_d$")

    plt.subplot(435)
    plt.plot(xi_eval, qd.T)
    plt.ylabel(r"$q_{d}$")
    plt.subplot(436)
    plt.plot(time_eval, qd_t.T)
    plt.ylabel(r"$q_d$")

    plt.subplot(438)
    plt.plot(xi_eval, wd.T)
    plt.ylabel(r"$w_{d}$")
    plt.xlabel(r"$\xi$")
    plt.subplot(439)
    plt.plot(time_eval, wd_t.T)
    plt.ylabel(r"$w_d$")

    plt.subplot(4, 3, 11)
    plt.plot(xi_eval, rd.T)
    plt.ylabel(r"$\dot{w}_{d}$")
    plt.xlabel(r"$\xi$")
    plt.subplot(4, 3, 12)
    plt.plot(time_eval, rd_t.T)
    plt.ylabel(r"$\dot{w}_{d}$")
    plt.xlabel(r"$t\,[s]$")

    plt.show()


def visualize_path_with_frames(pd, erf, scale, ax, secondary=False):
    e1 = erf[:, 0, :]
    e2 = erf[:, 1, :]
    e3 = erf[:, 2, :]

    alpha = 1
    color_path = "k"
    if secondary:
        alpha = 0.5
        scale = scale / 2
        color_path = "m"
    if ax is None:
        ax = plt.figure().add_subplot(111, projection="3d")
    ax.plot(pd[0], pd[1], pd[2], "--", color=color_path, alpha=alpha)

    for k in range(0, pd.shape[1], int(pd.shape[1] / 20)):
        ax.plot(
            [pd[0, k], pd[0, k] + scale * e1[0, k]],
            [pd[1, k], pd[1, k] + scale * e1[1, k]],
            [pd[2, k], pd[2, k] + scale * e1[2, k]],
            "r-",
            alpha=alpha,
        )
        ax.plot(
            [pd[0, k], pd[0, k] + scale * e2[0, k]],
            [pd[1, k], pd[1, k] + scale * e2[1, k]],
            [pd[2, k], pd[2, k] + scale * e2[2, k]],
            "g-",
            alpha=alpha,
        )
        ax.plot(
            [pd[0, k], pd[0, k] + scale * e3[0, k]],
            [pd[1, k], pd[1, k] + scale * e3[1, k]],
            [pd[2, k], pd[2, k] + scale * e3[2, k]],
            "b-",
            alpha=alpha,
        )
    ax = axis_equal(pd[0, :], pd[1, :], pd[2, :], ax=ax)

    return ax


def axis_equal(X, Y, Z, ax=None):
    """
    Sets axis bounds to "equal" according to the limits of X,Y,Z.
    If axes are not given, it generates and labels a 3D figure.

    Args:
        X: Vector of points in coord. x
        Y: Vector of points in coord. y
        Z: Vector of points in coord. z
        ax: Axes to be modified

    Returns:
        ax: Axes with "equal" aspect


    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    max_range = (
        np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    )
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    return ax
