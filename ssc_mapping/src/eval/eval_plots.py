#!/usr/bin/env python3

import subprocess
import os
import csv
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import math
import sys
import pandas
from pathlib import Path
import seaborn as sns
from sympy import E

plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16


def create_figure_object(num_plots, width=16, height=10):
    cols = 3
    rows = math.ceil(num_plots/cols)

    fig, axs = plt.subplots(rows, cols)
    fig.set_figheight(height)
    fig.set_figwidth(width)
    num_extra_plots = ((rows * cols) - num_plots)
    if num_extra_plots > 0:
        for ax in axs.flat[-num_extra_plots:]:
            fig.delaxes(ax)
    return fig, axs


def read_plot_data_csv(csv_file):
    data = {}
    with open(csv_file, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)
        for key in header:
            data[key] = []

        for row in reader:
            for i, r in enumerate(row):
                data[header[i]].append(float(r))
    return data


def plot_metrics(data, quality_ax, coverage_ax, label=""):
    headers = list(data.keys())
    # Quality plot.
    for i, ax in enumerate(quality_ax.flat):
        j = i + 1
        ax.set_ylim([0, 1])
        ax.set_title(headers[j])
        sns.lineplot(x=np.asarray(
            data[headers[0]]) * 0.5, y=data[headers[j]], ci="sd", label=label, ax=ax)
        ax.set(xlabel='Time in Minutes')
        ax.legend()

    # Coverage plot.
    for i, ax in enumerate(coverage_ax.flat):
        j = i + 8
        ax.set_ylim([0, 1])
        ax.set_title(headers[j])
        sns.lineplot(x=np.asarray(
            data[headers[0]]) * 0.5, y=data[headers[j]], ci="sd", label=label, ax=ax)
        ax.set(xlabel='Time in Minutes')
        ax.legend()


def save_mean_csv(csv_file, column="time_segment_id"):
    df = pandas.read_csv(csv_file)
    df_mean = df.groupby(column).mean()
    df_mean.to_csv(csv_file)


def evaluate_comp(data_folder, gt_path, eval_type, force_evaluate=False, ssc_prefix="", confidence_weight=0):
    # Evaluate all maps and get data.
    dirs = os.listdir(data_folder)
    prefix_info_str = f"prefix '{ssc_prefix}' " if ssc_prefix else ""
    if ssc_prefix:
        ssc_prefix = ssc_prefix + "_"

    suffix_info_str = ""
    ssc_suffix=""
    if confidence_weight > 0:
        ssc_suffix = "_" + f"{confidence_weight:.2f}"[2:]
        suffix_info_str = f"confidence={confidence_weight:.2f} "

    print(
        f"========== Evaluating '{data_folder}' for '{eval_type}' {prefix_info_str}{suffix_info_str}==========")
    for d in dirs:
        curr_data_dir = os.path.join(data_folder, d)
        if not os.path.isdir(curr_data_dir) or "plots" in d:
            continue

        print(
            f"===== Evaluating '{curr_data_dir}' for '{eval_type}' {prefix_info_str}{suffix_info_str}=====")

        # Setup files to write evaluation to.
        eval_dir = os.path.join(curr_data_dir, 'metrics')
        if not os.path.isdir(eval_dir):
            os.mkdir(eval_dir)

        eval_files = [eval_type]
        if eval_type == "all":
            eval_files = ["tsdf", ssc_prefix+"ssc"+ssc_suffix, ssc_prefix +
                          "hierarchical" + ssc_suffix, ssc_prefix+"ssc_unobs" + ssc_suffix]
        elif eval_type != "tsdf":
            eval_files = [ssc_prefix+eval_type+ssc_suffix]
        needs_eval = force_evaluate
        for e in eval_files:
            file_name = os.path.join(eval_dir, e+".csv")
            if not os.path.isfile(file_name):
                needs_eval = True
                break
        if not needs_eval:
            print(f"Evaluation already up to date.")
            continue
        for e in eval_files:
            file_name = os.path.join(eval_dir, e+".csv")
            if os.path.isfile(file_name):
                print(
                    f"Reset existing 'metrics/{e}.csv' {'(force update)' if force_evaluate else ''}.")
            with open(file_name, 'w') as f:
                f.write("time_segment_id,precision_occ,precision_free,precision_overall,recall_occ,recall_free,IoU_occ,IoU_free,explored_occ,explored_free,explored_overall,coverage_occ,coverage_free,coverage_overall\n")

        # Evaluate all maps.
        map_names = sorted(
            [x[:-5] for x in os.listdir(os.path.join(curr_data_dir, 'voxblox_maps')) if x.endswith('.tsdf')])
        for i, m in enumerate(map_names):
            print(f"Evaluating map {i+1}/{len(map_names)}.")
            subprocess.run(
                f"rosrun ssc_mapping ssc_map_eval_node _gt_layer_path:={gt_path} _tsdf_layer_path:={os.path.join(curr_data_dir, 'voxblox_maps', m +'.tsdf')} _ssc_layer_path:={os.path.join(curr_data_dir, 'ssc_maps', ssc_prefix + m +'.ssc')} _output_path:={curr_data_dir} _refine_ob_layer:=true _publish_visualization:=false _eval_type:={eval_type} _ssc_prefix:={ssc_prefix} _ssc_confidence_threshold:={confidence_weight}", shell=True)

    # Setup figures for plotting.
    print(
        f"===== Plotting '{data_folder}' for '{eval_type}' {prefix_info_str}{suffix_info_str}=====")
    for e in eval_files:
        print(f"Plotting results for '{e}'.")
        file_name = os.path.join(eval_dir, e+".csv")
        quality_metrics_fig, quality_metrics_axs = create_figure_object(7)
        coverage_metrics_fig, coverage_metrics_axs = create_figure_object(6)

        quality_plot_out_file = Path(
            data_folder) / "plots" / f"{e}_quality.png"
        coverage_plot_out_file = Path(
            data_folder) / "plots" / f"{e}_coverage.png"
        quality_plot_out_file.parent.mkdir(parents=True, exist_ok=True)
        for d in dirs:
            curr_data_dir = os.path.join(data_folder, d)
            if not os.path.isdir(curr_data_dir) or "plots" in d:
                continue
            plot_metrics(read_plot_data_csv(os.path.join(curr_data_dir,'metrics',
                         e + '.csv')), quality_metrics_axs, coverage_metrics_axs, label=d)

        # save quality metric
        quality_metrics_fig.tight_layout()
        quality_metrics_fig.savefig(quality_plot_out_file, bbox_inches='tight')

        # save coverage metric
        coverage_metrics_fig.tight_layout()
        coverage_metrics_fig.savefig(
            coverage_plot_out_file, bbox_inches='tight')
    print(
        f"========== Evaluation of '{data_folder}' for '{eval_type}' {prefix_info_str}{suffix_info_str}finished ==========")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(
            "usage: eval_plots.py <target_dir> <gt_tsdf_map> <eval_type> <opt: force_evaluate> <opt: ssc_prefix> <opt: confidence>")
        exit(-1)
    # eval_type: tsdf, ssc, hierarchical, ssc_unobs, all
    force_evaluate = False
    if len(sys.argv) >= 5:
        force_evaluate = sys.argv[4] == "true"
    ssc_prefix = ""
    if len(sys.argv) >= 6:
        ssc_prefix = "" # sys.argv[5]
    confidence_weight = 0
    if len(sys.argv) >= 7:
        confidence_weight = float(sys.argv[6])
    evaluate_comp(sys.argv[1], sys.argv[2], sys.argv[3],
                  force_evaluate, ssc_prefix, confidence_weight)
