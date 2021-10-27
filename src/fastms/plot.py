import os.path
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser(description='Plot some magic')
parser.add_argument('path', type=str, default='./outputs/results.json')
parser.add_argument('outpath', type=str, default='./outputs/')
args = parser.parse_args()

def _yearly_mean(x):
    return np.mean(x, axis=1)

def plot(path, outpath):
    with open(path, 'r') as f:
        results = json.load(f)
    x = [m for r in results for m in _yearly_mean(r['truth'])]
    y = [m for r in results for m in _yearly_mean(r['prediction'])]
    plt.plot(x, y, linestyle = '', marker = 'o', markersize=0.7)
    plt.xlabel('truth')
    plt.ylabel('prediction')
    plt.savefig(os.path.join(outpath, 'scatter.png'))

    plt.clf()
    sns.kdeplot(x=x, y=y, cmap = 'Reds', shade = True)
    plt.savefig(os.path.join(outpath, 'density.png'))

    errors = [
        mean_squared_error(r['truth'], r['prediction']) for r in results
    ]
    plt.clf()
    plt.boxplot(errors)
    plt.savefig(os.path.join(outpath, 'boxplot.png'))


if __name__ == "__main__":
    plot(args.path, args.outpath)
