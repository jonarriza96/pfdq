import numpy as np
import matplotlib.pyplot as plt
import pickle
import subprocess


def get_pfdq_path():
    pacor_path = subprocess.run(
        "echo $PFDQ_PATH", shell=True, capture_output=True, text=True
    ).stdout.strip("\n")
    return pacor_path


def save_pickle(path, file_name, data):
    pickle_path = path + "/" + file_name + ".pickle"
    with open(pickle_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def visualize_path_frames(pd, erf, scale, ax, secondary=False, hide_e1=True, alpha=0):
    e1 = erf[:, 0, :]
    e2 = erf[:, 1, :]
    # e3 = erf[:, 2, :]

    if alpha == 0:
        alpha = 0.5  # 0.25
    # if secondary:
    #     alpha = 0.5
    #     # scale = scale / 2

    for k in range(0, pd.shape[1], int(pd.shape[1] / 100)):
        if not hide_e1:
            ax.plot(
                [pd[0, k], pd[0, k] + scale * e1[0, k]],
                [pd[1, k], pd[1, k] + scale * e1[1, k]],
                # [pd[2, k], pd[2, k] + scale * e1[2, k]],
                "r-",
                alpha=alpha,
            )
        ax.plot(
            [pd[0, k], pd[0, k] + scale * e2[0, k]],
            [pd[1, k], pd[1, k] + scale * e2[1, k]],
            # [pd[2, k], pd[2, k] + scale * e2[2, k]],
            "g-",
            alpha=alpha,
        )
        # ax.plot(
        #     [pd[0, k], pd[0, k] + scale * e3[0, k]],
        #     [pd[1, k], pd[1, k] + scale * e3[1, k]],
        #     [pd[2, k], pd[2, k] + scale * e3[2, k]],
        #     "b-",
        #     alpha=alpha,
        # )
    # ax = axis_equal(pd[0, :], pd[1, :], pd[2, :], ax=ax)

    return ax


def visualize_path_frames_spatial(pd, erf, scale, ax, secondary=False, interval=20):
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
    # ax.plot(pd[0], pd[1], pd[2], "--", color=color_path, alpha=alpha)

    for k in range(0, pd.shape[1], int(pd.shape[1] / interval)):
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
    # ax = axis_equal(pd[0, :], pd[1, :], pd[2, :], ax=ax)

    return ax


def visualize_path_frames_spatial_top(pd, erf, scale, ax, secondary=False, interval=20):
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
    # ax.plot(pd[0], pd[1], pd[2], "--", color=color_path, alpha=alpha)

    for k in range(0, pd.shape[1], int(pd.shape[1] / interval)):
        ax.plot(
            [pd[0, k], pd[0, k] + scale * e1[0, k]],
            [pd[1, k], pd[1, k] + scale * e1[1, k]],
            # [pd[2, k], pd[2, k] + scale * e1[2, k]],
            "r-",
            alpha=alpha,
        )
        ax.plot(
            [pd[0, k], pd[0, k] + scale * e2[0, k]],
            [pd[1, k], pd[1, k] + scale * e2[1, k]],
            # [pd[2, k], pd[2, k] + scale * e2[2, k]],
            "g-",
            alpha=alpha,
        )
        ax.plot(
            [pd[0, k], pd[0, k] + scale * e3[0, k]],
            [pd[1, k], pd[1, k] + scale * e3[1, k]],
            # [pd[2, k], pd[2, k] + scale * e3[2, k]],
            "b-",
            alpha=alpha,
        )
    # ax = axis_equal(pd[0, :], pd[1, :], pd[2, :], ax=ax)

    return ax


def visualize_path_frames_spatial_front(
    pd, erf, scale, ax, secondary=False, interval=20
):
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
    # ax.plot(pd[0], pd[1], pd[2], "--", color=color_path, alpha=alpha)

    for k in range(0, pd.shape[1], int(pd.shape[1] / interval)):
        ax.plot(
            [pd[0, k], pd[0, k] + scale * e1[0, k]],
            # [pd[1, k], pd[1, k] + scale * e1[1, k]],
            [pd[2, k], pd[2, k] + scale * e1[2, k]],
            "r-",
            alpha=alpha,
        )
        ax.plot(
            [pd[0, k], pd[0, k] + scale * e2[0, k]],
            # [pd[1, k], pd[1, k] + scale * e2[1, k]],
            [pd[2, k], pd[2, k] + scale * e2[2, k]],
            "g-",
            alpha=alpha,
        )
        ax.plot(
            [pd[0, k], pd[0, k] + scale * e3[0, k]],
            # [pd[1, k], pd[1, k] + scale * e3[1, k]],
            [pd[2, k], pd[2, k] + scale * e3[2, k]],
            "b-",
            alpha=alpha,
        )
    # ax = axis_equal(pd[0, :], pd[1, :], pd[2, :], ax=ax)

    return ax


def visualize_path_frames_spatial_side(
    pd, erf, scale, ax, secondary=False, interval=20
):
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
    # ax.plot(pd[0], pd[1], pd[2], "--", color=color_path, alpha=alpha)

    for k in range(0, pd.shape[1], int(pd.shape[1] / interval)):
        ax.plot(
            # [pd[0, k], pd[0, k] + scale * e1[0, k]],
            [pd[1, k], pd[1, k] + scale * e1[1, k]],
            [pd[2, k], pd[2, k] + scale * e1[2, k]],
            "r-",
            alpha=alpha,
        )
        ax.plot(
            # [pd[0, k], pd[0, k] + scale * e2[0, k]],
            [pd[1, k], pd[1, k] + scale * e2[1, k]],
            [pd[2, k], pd[2, k] + scale * e2[2, k]],
            "g-",
            alpha=alpha,
        )
        ax.plot(
            # [pd[0, k], pd[0, k] + scale * e3[0, k]],
            [pd[1, k], pd[1, k] + scale * e3[1, k]],
            [pd[2, k], pd[2, k] + scale * e3[2, k]],
            "b-",
            alpha=alpha,
        )
    # ax = axis_equal(pd[0, :], pd[1, :], pd[2, :], ax=ax)

    return ax
