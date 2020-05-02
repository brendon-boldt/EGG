from pathlib import Path
import pickle as pkl
from typing import Dict, List, Any, Tuple
import re
import numpy as np
from matplotlib import pyplot as plt
from random import shuffle

def load_from_path(path: Path) -> List[Tuple[str, Dict]]:
    data = []
    for p in path.glob(r"config_*.pkl"):
        if not re.match(r"config_\d+.pkl", p.name):
            continue
        try:
            data.append((str(p), pkl.load(p.open('rb'))))
        except Exception as e:
            print(f"Couldn't load {p} due to {e}")
    return data

def extract(raw: List[Tuple], fields: List[str]) -> Dict[int, Any]:
    d = {}
    for name, data in raw:
        k = int(re.search(r'config_(\d+).pkl', name).group(1))
        v: Any = data[fields[0]]
        for f in fields[1:]:
            v = v[f]
        d[k] = v
    return d

def compare():
    for i in range(200):
        if i in er0 and i in er1:
            print(f"{i:03d} {er0[i]:.3f} {er1[i]:.3f} {abs(er0[i]-er1[i]):.3f}")

def print_sorted():
    for k, v in er1.items():
        max_acc_idxs = er1[k].argmax()
        er1_ts[k] = er1_ts[k][max_acc_idxs]
    for x, y in sorted(er1_ts.items(), key=lambda x: x[1]):
        print(x)

def flatten_dict(d, prefix=""):
    e = {}
    if prefix:
        prefix += "_"
    for k, v in d.items():
        if type(v) == dict:
            e = {**e, **flatten_dict(v, prefix + k)}
        else:
            e[prefix + k] = v
    return e

def merge_summaries(raw, group_by, other=None):
    raw = [(x[0], flatten_dict(x[1])) for x in raw]
    d = {}
    for k, v in raw:
        if type(group_by) == list:
            group = "_".join(str(getattr(v['opts'], gb)) for gb in group_by)
        else:
            group = getattr(v['opts'], group_by) 
        # if not group in ['1_64', '2_32', '4_8', '8_4']:
        if other == 'dp_prior':
            if group[:9] == "None_0.00": continue # For DP prior vocab counts
        if group not in d:
            d[group] = v
            d[group]['run_ids'] = k
            for k1, v1 in v.items():
                d[group][k1] = [v1]
        else:
            for k1, v1 in v.items():
                d[group][k1] += [v1]
    for k, v in d.items():
        for k1, v1 in v.items():
            v[k1] = np.array(v1)
    return d

def plot_raw(r1):
    shuffle(r1)
    for k, d in r1:
        v = d['valid']['acc']
        v2 = d['valid']['loss']
        vmax = [v[0]]
        for x in v[1:]:
            if x > vmax[-1]:
                vmax.append(x)
            else:
                vmax.append(vmax[-1])
        max_v = vmax[-1]
        vmax = np.array(vmax)
        print(k)
        print(d['opts'].train_mode)
        # plt.plot(np.log(v/(1-v)))
        x_axis = np.arange(len(v)) * d['examples_per_epoch'] / 1e3
        # plt.ylim(0., 4.2)
        # plt.plot(x_axis, np.log(max_v/(1-max_v)) - np.log(vmax/(1-vmax)))
        plt.plot(x_axis, v)
        plt.plot(x_axis, v2)
        plt.show()
        plt.clf()

def logit(x):
    return np.log(x / (1 - x))

def plot_word_freqs(data):
    items = sorted(data.items(), key=lambda x: x[0])
    for k, v in items:
        print(k)
        arr = v['valid_word_freqs_sorted'][:, -1, :]
        stderr = arr.std(0) / arr.shape[0]**.5
        
        print((logit(v['valid_acc']) - logit(.1)).mean(0)[-1])
        print(v['valid_toposim'].mean(0)[-1])
        xss = [
            arr.mean(0),
        ]
        x_axis = np.arange(1, arr.shape[-1] + 1, dtype=np.int64)
        for xs in xss:
            plt.plot(x_axis, xs)
            plt.fill_between(x_axis, xs - 2*stderr, xs + 2 * stderr, alpha=.3)
            # plt.scatter(xs, ys)
    legend = {
        "None_0.01": "\u03BB=0",
        "dp_0.0001": "\u03BB=0.0001",
        "dp_0.001": "\u03BB=0.001",
        "dp_0.01": "\u03BB=0.01",
            }
    plt.legend([legend[k] for k, _ in items])
    plt.xlabel("Word Rank")
    plt.ylabel("Frequency")
    # plt.show()
    plt.clf()

def plot_train_modes(data):
    items = sorted(data.items(), key=lambda x: x[0])
    for k, v in items:
        print(k)
        # arr = logit(v['valid_acc']) - logit(.1)
        arr = v['valid_toposim']
        # for i in range(arr.shape[1]):
        #     arr[:, i] = arr[:, :i+1].max(-1)
        stderr = arr.std(0) / arr.shape[0]**.5
        print(arr.mean(0)[-1])
        print(arr.std(0)[-1])
        print(stderr[-1])
        xss = [
            arr.mean(0),
        ]
        x_axis = np.arange(1, arr.shape[-1] + 1, dtype=np.int64)
        for xs in xss:
            plt.plot(x_axis, xs)
            plt.fill_between(x_axis, xs - 2*stderr, xs + 2 * stderr, alpha=.3)
            # plt.scatter(xs, ys)
    legend = {
        "stgs": "ST Gumbel-Softmax",
        "gs": "Gumbel-Softmax",
        "rf": "REINFORCE",
    }
    plt.legend([legend[k] for k, _ in items])
    plt.xlabel("Thousands of examples seen")
    plt.ylabel("Adjusted logit accuracy")
    # plt.show()
    # plt.savefig("/home/brendon/10701-figs/max_acc.png")
    # plt.savefig("/home/brendon/10701-figs/avg_acc.png")
    plt.clf()

def plot_n_distractors(data):
    items = sorted(data.items(), key=lambda x: -int(x[0]))
    for k, v in items:
        n_examples = v['opts'][0].n_distractors + 1
        arr = logit(v['valid_acc']) - logit(n_examples**-1)
        # arr = v['valid_acc'] / (v['opts'][0].n_distractors + 1)**-1
        arr_orig = arr.copy()
        # for i in range(arr.shape[1]):
        #     arr[:, i] = arr[:, :i+1].max(-1)
        print(k)
        print(arr_orig[:, -1].std())
        print(arr[:, -1].std())
        stderr = arr.std(0) / arr.shape[0]**.5
        xss = [
            arr.mean(0),
        ]
        # ys = logit(v['valid_acc']).mean(0)
        x_axis = np.arange(len(xss[0])) * v['examples_per_epoch'][0] * n_examples / 1e3
        for xs in xss:
            plt.plot(x_axis, xs)
            plt.fill_between(x_axis, xs - 2*stderr, xs + 2 * stderr, alpha=.3)
            # plt.scatter(xs, ys)
    plt.xlabel("Thousands of individual examples seen")
    plt.ylabel("Adjusted logit accuracy")
    plt.legend([int(k) + 1 for k, _ in items])
    # plt.show()
    plt.savefig("/home/brendon/10701-figs/n_distractors.png")
    plt.clf()

def plot_merged(data):
    items = sorted(data.items(), key=lambda x: -int(x[0]))
    for k, v in items:
        plt.title(k)
        n_examples = v['opts'][0].n_distractors + 1
        arr = logit(v['valid_acc']) - logit(n_examples**-1)
        # arr = v['valid_acc'] / (v['opts'][0].n_distractors + 1)**-1
        arr_orig = arr.copy()
        # for i in range(arr.shape[1]):
        #     arr[:, i] = arr[:, :i+1].max(-1)
        print(k)
        print(arr_orig[:, -1].std())
        print(arr[:, -1].std())
        stderr = arr.std(0) / arr.shape[0]**.5
        xss = [
            arr.mean(0),
        ]
        # ys = logit(v['valid_acc']).mean(0)
        x_axis = np.arange(len(xss[0])) * v['examples_per_epoch'][0] * n_examples / 1e3
        for xs in xss:
            plt.plot(x_axis, xs)
            plt.fill_between(x_axis, xs - 2*stderr, xs + 2 * stderr, alpha=.3)
            # plt.scatter(xs, ys)
    plt.legend([k for k, _ in items])
    plt.show()
    plt.clf()

def table_vocab(data):
    xs = []
    ys = []
    for m_size in [1, 2, 4, 8, 16]:
        for v_size in [2, 4, 8, 16, 32, 64]:
            k = f"{m_size}_{v_size}"
            # arr = logit(data[k]['valid_acc']) - logit(.1)
            # arr = data[k]['valid_toposim']
            xs.append(data[k]['valid_toposim'].mean(0)[-1])
            ys.append(logit(data[k]['valid_acc'].mean(0)[-1]))
            # print(f"& ${arr.mean(0)[-1]:.2f}$ ", end="")
            # for i in range(arr.shape[1]):
            #     arr[:, i] = arr[:, :i+1].max(-1)
            # stderr = arr.std(0) / arr.shape[0]**.5
        # print(r"\\")
    plt.scatter(xs, ys)
    plt.show()
    plt.clf()


# raw = load_from_path(Path('vocab_prior0'))
# data = merge_summaries(raw, ['vocab_prior', 'dp_lambda_0'], other='dp_prior')
# plot_word_freqs(data)

# raw = load_from_path(Path('distractors0'))
# data = merge_summaries(raw, ['n_distractors'])
# plot_n_distractors(data)

raw = load_from_path(Path('100x'))
data = merge_summaries(raw, ['train_mode'])
plot_train_modes(data)

# raw = load_from_path(Path('vocab_space'))
# data = merge_summaries(raw, ['max_len', 'vocab_size'])
# table_vocab(data)

# raw = load_from_path(Path('distractors0'))
# # data = merge_summaries(raw, ['max_len', 'vocab_size'])
# data = merge_summaries(raw, ['n_distractors'])
# plot_n_distractors(data)
