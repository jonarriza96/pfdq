# PFDQ: Pose-Following with Dual Quaternions

A pose-following framework for simultaneously controlling the translation and rotation of a rigid body.

## Requirements

### Python environment

Install the specific versions of every package from `requirements.txt`.
In a new conda environment:

```
conda create --name pfdq python=3.9
conda activate pfdq
pip install -r requirements.txt
```

Update the `~/.bashrc` with

```
export PFDQ_PATH=/path_to_pfdq
export PYTHONPATH=$PYTHONPATH:/$PFDQ_PATH
```

## Usage

### Replicating the experiments

To replicate the two case studies of the paper, run [this file](pose_following.py). When doing so, you can:

1. choose to use **path-following or path-tracking** by [changing this variable](pose_following.py#L33), i.e., `pf=True` activates path-following.

2. choose the **case study** by [changing this variable](pose_following.py#L29), i.e., `case_study=1` or `case_study=2`. The specific settings for each case study are as follows:

   - **Case study 1 - Comparison to pose-tracking**: you can activate or deactive the disturbance with [this variable](pose_following.py#L40) and the [velocity profile](pose_following.py#L37) ("c" for conservative, "m" for medium, "p" for progressive).

   - **Case study 2 - Almost global asymptotic stability on pose-following with velocity assignment**: you can choose the [starting point](pose_following.py#L47) (0 to 4) and the [velocity profile](pose_following.py#L46) ("s" for slow, "f" for fast and "v" for variant, i.e., sinusoidal). In order to deactivate the lambda (as mentioned in the paper), you need to uncomment [this condition](pfdq/utils/pose_following.py#L443). Notice that in this case-study only uses pose-following, i.e., `pf=True`.

3. **save the results** by [changing this variable](ph.dq.py#L34). It is recommended not to save trigger this, since you will overwrite the results of the paper.

### Visualizing the results

If you run the case studies as mentioned above and save the respective results, you can generate the same plots as in the paper:

1. **Case study 1 - Comparison to pose-tracking**: Run [this file](pfdq/results/case_study1_results.py).

2. **Case study2 - Almost global asymptotic stability on pose-following with velocity assignment**: For the first column, run [this file](pfdq/results/case_study2_col1_results.py) and, for the second column [this other file](pfdq/results/case_study2_col2_results.py).
