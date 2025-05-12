import os, gc
import sys, pdb
import copy, time
import json, random

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
from colorama import Style, Fore

from arc.utils import render_grid

test_path = '../input/arc-prize-2024/arc-agi_test_challenges.json'
sample_path = '../input/arc-prize-2024/sample_submission.json'

# ......................................................................................................
cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)
color_list = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]

# ......................................................................................................

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

import signal
import psutil
import itertools
import subprocess
import base64, gzip
import networkx as nx
import multiprocessing
#::::::::::::::::::::::::::::::::::::::::::::::
from numpy import array
from pathlib import Path
from scipy import ndimage
from scipy.stats import mode
from tqdm.auto import trange
from functools import partial
from tqdm.notebook import tqdm
from PIL import Image, ImageDraw
from xgboost import XGBClassifier
from itertools import combinations, product
from collections import defaultdict, Counter
from skimage.measure import label, regionprops
#::::::::::::::::::::::::::::::::::::::::::::::
from sklearn.tree import *
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import MinMaxScaler
#::::::::::::::::::::::::::::::::::::::::::::::
import warnings # suppress warnings
warnings.filterwarnings('ignore')
#::::::::::::::::::::::::::::::::::::::::::::::

################################################################################ 
# 40 Functions - Via Different Solvers
################################################################################ 1
def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

################################################################################ 2
def get_objects(task):
    xs, ys = [], []
    for obj in task['train']:
        xs.append(np.array(obj['input']))
        ys.append(np.array(obj['output']))
    return xs, ys

################################################################################ 3
def find_sub(matrix, sub, ignore=None):
    positions = []
    mask = sub != ignore
    sub_ = sub[mask]
    for x in range(matrix.shape[0]-sub.shape[0]+1):
        for y in range(matrix.shape[1]-sub.shape[1]+1):
            if np.array_equal(matrix[x:x+sub.shape[0], y:y+sub.shape[1]][mask], sub_):
                positions.append((x,y,x+sub.shape[0],y+sub.shape[1]))
    return positions

################################################################################ 4
def check_subitem(task):
    for x, y in zip(*get_objects(task)):
        positions = find_sub(x, y)
        if len(positions) == 0:
            return False
    return True
            
################################################################################ 5
def check_samesize(task):
    for x,y in zip(*get_objects(task)):
        if x.shape != y.shape:
            return False
    return True

################################################################################ 6
def check_sub_mask(task):
    if check_samesize(task):
        return False
    for x,y in zip(*get_objects(task)):
        colors, counts = np.unique(x, return_counts=True)
        found = 0
        for c, area in zip(colors, counts):
            cxs, cys = np.where(x == c)
            xmin,ymin,xmax,ymax = min(cxs),min(cys),max(cxs)+1,max(cys)+1
            shape = (xmax-xmin, ymax-ymin)
            if shape == y.shape and area == np.prod(y.shape):
                found += 1
        if found != 1:
            return False
    return True

################################################################################ 7
def get_cells(x, cols, rows):
    if cols[0] != 0:
        cols = [-1]+cols
    if rows[0] != 0:
        rows = [-1]+rows
    if cols[-1] != x.shape[0]-1:
        cols.append(x.shape[0])
    if rows[-1] != x.shape[1]-1:
        rows.append(x.shape[1])
    cells = np.full((len(cols)-1, len(rows)-1), object)
    for i in range(len(cols)-1):
        for j in range(len(rows)-1):
            cells[i][j] = x[cols[i]+1:cols[i+1], rows[j]+1:rows[j+1]]
    return cells

################################################################################ 8
def get_grid(x):
    cols = defaultdict(list)
    rows = defaultdict(list)
    if x.shape[0] < 3 or x.shape[1] < 3:
        return -1, [], []
    for i in range(x.shape[0]):
        if len(np.unique(x[i])) == 1:
            cols[x[i,0]].append(i)
    for i in range(x.shape[1]):
        if len(np.unique(x[:,i])) == 1:
            rows[x[0,i]].append(i)
    for c in cols:
        if c in rows and all(np.diff(cols[c])>1) and all(np.diff(rows[c])>1):
            return c, cols[c], rows[c]
    return -1, [], []

################################################################################ 9
def check_grid(task):
    for x,y in zip(*get_objects(task)):
        color_of_grid, cols, rows = get_grid(x)
        if color_of_grid == -1:
            return False
    return True

################################################################################ 10
def mul_ratio(x, x_ratio):
    x_shape = (x.shape[0]*x_ratio[0], x.shape[1]*x_ratio[1])
    x_ = np.array([x[i//x_ratio[0]][j//x_ratio[1]] for i, j in np.ndindex(x_shape)]).reshape(x_shape)
    return x_

################################################################################ 11
def predict_transforms(xs, ys, test):
    fn = get_transform(xs, ys)
    if fn:
        return [fn(test)]
    ratio = get_ratio(xs, ys)
    if ratio:
        x_ratio, y_ratio = ratio
        xs_ = []
        ys_ = []
        for x, y in zip(xs, ys):
            x, y = mul_ratios(x, y, x_ratio, y_ratio)
            xs_.append(x)
            ys_.append(y)
        fn = get_transform(xs_, ys_)
        if fn:
            test = mul_ratio(test, x_ratio)
            return [fn(test)]
        fns = np.full(x_ratio, object)
        for i, j in np.ndindex(x_ratio):
            ys_ = []
            for y in ys:
                m1 = y.shape[0]//x_ratio[0]
                m2 = y.shape[1]//x_ratio[1]
                ys_.append(y[i*m1:(i+1)*m1,j*m2:(j+1)*m2])
            fn = get_transform(xs, ys_)
            if fn:
                fns[i,j] = fn
            else:
                return []
        return [np.concatenate([np.concatenate([fns[i,j](test) for i in range(x_ratio[0])], axis=0) for j in range(x_ratio[1])], axis=1)]
    return []

################################################################################ 12
def predict_grid_transforms(task, test):
    xs, ys = get_objects(task)
    xs = [grid_filter(x) for x in xs]
    return predict_transforms(xs, ys, grid_filter(test))

################################################################################ 13
def get_transform(xs, ys):
    for tranform in get_all_transforms():
        tranformed = True
        for x, y in zip(xs, ys):
            if tranform(x).shape != y.shape:
                tranformed = False
                break
            if not np.equal(tranform(x), y).all():
                tranformed = False
        if tranformed:
            return tranform
    return None

################################################################################ 14
def get_transforms(xs, ys):
    fn = get_transform(xs, ys)
    if fn:
        return fn
    ratio = get_ratio(xs, ys)
    if ratio:
        x_ratio, y_ratio = ratio
        xs_ = []
        ys_ = []
        for x, y in zip(xs, ys):
            x, y = mul_ratios(x, y, x_ratio, y_ratio)
            xs_.append(x)
            ys_.append(y)
        fn = get_transform(xs_, ys_)
        if fn:
            return fn
        fns = np.full(x_ratio, object)
        for i, j in np.ndindex(x_ratio):
            ys_ = []
            for y in ys:
                m1 = y.shape[0]//x_ratio[0]
                m2 = y.shape[1]//x_ratio[1]
                ys_.append(y[i*m1:(i+1)*m1,j*m2:(j+1)*m2])
            fn = get_transform(xs, ys_)
            if fn:
                fns[i,j] = fn
            else:
                return None
        return fns
    return None

################################################################################ 15
def check_grid_transforms(task):
    xs, ys = get_objects(task)
    xs = [grid_filter(x) for x in xs]
    return get_transforms(xs, ys) is not None

################################################################################ 16
def get_mode_color(ar):
    colors, counts = np.unique(ar, return_counts=True)
    return colors[np.argmax(counts)]

################################################################################ 17
def grid_filter(x):
    color_of_grid, cols, rows = get_grid(x)
    if color_of_grid == -1:
        return x
    cells = get_cells(x, cols, rows)
    return np.array([get_mode_color(cell) for cell in cells.reshape(-1)]).reshape(cells.shape)

################################################################################ 18
def mul_ratios(x, y, x_ratio, y_ratio):
    x_shape = (x.shape[0]*x_ratio[0], x.shape[1]*x_ratio[1])
    x_ = np.array([x[i//x_ratio[0]][j//x_ratio[1]] for i, j in np.ndindex(x_shape)]).reshape(x_shape)
    y_shape = (y.shape[0]*y_ratio[0], y.shape[1]*y_ratio[1])
    y_ = np.array([y[i//y_ratio[0]][j//y_ratio[1]] for i, j in np.ndindex(y_shape)]).reshape(y_shape)
    return x_, y_

################################################################################ 19
def get_ratio(xs, ys):
    x_ratio = []
    y_ratio = []
    for i in range(2):
        if   all(x.shape[i]%y.shape[i] == 0  for x, y in zip(xs, ys)):
            if len(set(x.shape[i]//y.shape[i] for x, y in zip(xs, ys))) == 1:
                x_ratio.append(1)
                y_ratio.append(xs[0].shape[i]//ys[0].shape[i])
        elif all(y.shape[i]%x.shape[i] == 0  for x, y in zip(xs, ys)):
            if len(set(y.shape[i]//x.shape[i] for x, y in zip(xs, ys))) == 1:
                x_ratio.append(ys[0].shape[i]//xs[0].shape[i])
                y_ratio.append(1)
    if len(x_ratio) != 2:
        return None
    return tuple(x_ratio), tuple(y_ratio)

################################################################################ 20
def check_sub_grid_2x(task):
    if check_samesize(task) or check_subitem(task):
        return False
    for x,y in zip(*get_objects(task)):
        color_of_grid, cols, rows = get_grid(x)
        if color_of_grid == -1:
            return False
        cells = grid_filter(x)
        if (cells.shape[0]*2 != y.shape[0] or cells.shape[1]*2 != y.shape[1]):
            return False
    return True

################################################################################ 21
def check_chess(task, input=False, output=True):
    xs, ys = get_objects(task)
    if input:
        for x in xs:
            if not has_chess(x) and not has_antichess(x):
                return False
    if output:
        for y in ys:
            if not has_chess(y) and not has_antichess(y):
                return False
    return True

################################################################################ 22
def has_chess(g):
    colors = np.unique(g)
    counts = len(colors)
    if counts < 2:
        return False
    indexes = np.zeros(counts, bool)
    for c in colors:
        pts = np.where(g == c)
        s = set([(x+y)%counts for x, y in zip(*pts)])
        if len(s) > 1:
            return False
        index = s.pop()
        if indexes[index]:
            return False
        indexes[index] = True
    return True

################################################################################ 23
def has_antichess(g):
    colors = np.unique(g)
    counts = len(colors)
    if counts < 2:
        return False
    indexes = np.zeros(counts, bool)
    for c in colors:
        pts = np.where(g == c)
        s = set([(g.shape[0]-x+y-1)%counts for x, y in zip(*pts)])
        if len(s) > 1:
            return False
        index = s.pop()
        if indexes[index]:
            return False
        indexes[index] = True
    return True

################################################################################ 24
def find_colors(g):
    colors = np.unique(g)
    counts = len(colors)
    for cnt in range(counts, 1, -1):
        q_colors = np.full(cnt, -1, int)
        for c in colors:
            pts = np.where(g == c)
            s = set([(x+y)%cnt for x, y in zip(*pts)])
            if len(s) > 1:
                continue
            index = s.pop()
            q_colors[index] = c
        
        if -1 not in q_colors:
            return q_colors
    return None

################################################################################ 25
def predict_chess(g):
    q_colors = find_colors(g)
    if q_colors is None:
        colors, counts = np.unique(g, return_counts=True)
        q_colors = colors[np.argsort(counts)][:2]
        
    results = []
    counts = len(q_colors)
    for i in range(counts):
        result = g.copy()
        for x, y in np.ndindex(g.shape):
            result[x,y] = q_colors[(x+y)%counts]
        results.append(result)
        q_colors = np.roll(q_colors, 1)
    return results

################################################################################ 26
def predict_transforms_grid_2x(task, test):
    xs, ys = get_objects(task)
    xs = [grid_filter(x) for x in xs]
    return predict_transforms_2x(xs, ys, grid_filter(test))

################################################################################ 27
def predict_transforms_2x(xs, ys, test):
    predictions = []
    transforms = [
        lambda x: np.rot90(x.T, k=1),
        lambda x: np.rot90(x.T, k=3),
        lambda x: np.rot90(x, k=2),
        lambda x: x,
    ]
    quads = [(1,3,2,0),(3,1,0,2),(2,0,1,3)] # 3 full symmetrical shapes
    for f1,f2,f3,f4 in quads:
        fns = np.array([[transforms[f1],transforms[f2]],[transforms[f3],transforms[f4]]])
        x_ = np.concatenate([np.concatenate([fns[i,j](test) for i in range(2)], axis=0) for j in range(2)], axis=1)
        predictions.append(x_)
    return predictions

################################################################################ 28
def has_repeating(g, ignore=0):
    size0b = int(.6 * g.shape[0])
    size1b = int(.6 * g.shape[1])
    t = np.full((g.shape[0]+2*size0b,g.shape[1]+2*size1b), -1)
    t[size0b:-size0b,size1b:-size1b] = g
    t[t==ignore] = -1
    for size0 in range(2, size0b+1):
        for size1 in range(2, size1b+1):
            for shift0 in range(size0):
                for shift1 in range(size1):
                    pattern = t[size0b+shift0:size0b+shift0+size0,size1b+shift1:size1b+shift1+size1].copy()
                    found = True
                    for d0 in range(size0b+shift0-size0, t.shape[0]-size0, size0):
                        for d1 in range(size1b+shift1-size1, t.shape[1]-size1, size1):
                            test = t[d0:d0+size0,d1:d1+size1]
                            mask = (test != -1) & (pattern != -1)
                            if np.array_equal(test[mask], pattern[mask]):
                                ind = test != -1
                                pattern[ind] = test[ind]
                            else:
                                found = False
                                break
                    if found:
                        return shift0, shift1, pattern
    return None

################################################################################ 29
def check_repeating(task, has_complete=False):
    patterns = []
    for x, y in zip(*get_objects(task)):
        if len(np.unique(x)) < 3 or not has_repeating(y,-1):
            return False
        result = None
        for c in np.unique(x):
            # if c not in np.unique(y):
            result = has_repeating(x,c)
            if result:
                sh0,sh1,pattern = result
                pattern[pattern == -1] = c
                if has_complete:
                    pred = np.tile(pattern, (x.shape[0]//pattern.shape[0]+2, x.shape[1]//pattern.shape[1]+2))
                    pred1 = pred[sh0:sh0+x.shape[0],sh1:sh1+x.shape[1]]
                    pred2 = pred[sh0+1:sh0+1+x.shape[0],sh1:sh1+x.shape[1]]
                    pred3 = pred[sh0:sh0+x.shape[0],sh1+1:sh1+1+x.shape[1]]
                    if np.array_equal(pred1, y) or np.array_equal(pred2, y) or np.array_equal(pred3, y):
                        break
                    result = None
                else:
                    break
        if not result:
            return False
    return True

################################################################################ 30
def predict_repeating(x):
    for c in np.unique(x):
        result = has_repeating(x, c)
        if result:
            sh0,sh1,pattern = result
            pattern[pattern == -1] = c
            pred = np.tile(pattern, (x.shape[0]//pattern.shape[0]+2, x.shape[1]//pattern.shape[1]+2))
            pred1 = pred[sh0:sh0+x.shape[0],sh1:sh1+x.shape[1]]
            pred2 = pred[sh0+1:sh0+1+x.shape[0],sh1:sh1+x.shape[1]]
            pred3 = pred[sh0:sh0+x.shape[0],sh1+1:sh1+1+x.shape[1]]
            return [pred1,pred2,pred3]
    return []

################################################################################ 31
def predict_repeating_mask(x):
    predictions = predict_repeating(x)
    if len(predictions) > 0:
        rows, cols = np.where(predictions[0] != x)
        return [predictions[0][min(rows):max(rows)+1,min(cols):max(cols)+1]]
    return []

################################################################################ 32
def trim_matrix(x):
    if len(np.unique(x)) == 1:
        return x
    for c in np.unique(x):
        xs,ys = np.where(x!=c)
        xmin,ymin,xmax,ymax = min(xs),min(ys),max(xs)+1,max(ys)+1
        if xmin > 0 or ymin > 0 or xmax < x.shape[0] or ymax < x.shape[1]:
            return x[xmin:xmax,ymin:ymax]
    return x

################################################################################ 33
def trim_matrix_box(g, mask=None):
    if mask is None:
        mask = np.unique(g)
    if len(np.unique(g)) == 1:
        return None
    for c in mask:
        xs,ys = np.where(g!=c)
        xmin,ymin,xmax,ymax = min(xs),min(ys),max(xs)+1,max(ys)+1
        if xmin > 0 or ymin > 0 or xmax < g.shape[0] or ymax < g.shape[1]:
            return (xmin,ymin,xmax,ymax)
    return None

################################################################################ 34
def has_tiles(g, ignore=0):
    for size0b, size1b in [(g.shape[0], int(0.6*g.shape[1])), (int(0.6*g.shape[0]), g.shape[1])]:
        t = np.full((g.shape[0]+size0b, g.shape[1]+size1b), -1)
        t[:-size0b,:-size1b] = g
        t[t==ignore] = -1
        box_trim = trim_matrix_box(g,[ignore])
        min_size0 = 1
        min_size1 = 1
        if box_trim is not None and ignore != -1:
            xmin,ymin,xmax,ymax = box_trim
            t[xmin:xmax,ymin:ymax] = g[xmin:xmax,ymin:ymax]
            min_size0 = xmax-xmin
            min_size1 = ymax-ymin
        for size0 in range(min_size0, size0b+1):
            for size1 in range(min_size1, size1b+1):
                pattern = t[:size0,:size1].copy()
                found = True
                for d0 in range(0, t.shape[0]-size0, size0):
                    for d1 in range(0, t.shape[1]-size1, size1):
                        test = t[d0:d0+size0,d1:d1+size1]
                        mask = (test != -1) & (pattern != -1)
                        if np.array_equal(test[mask], pattern[mask]):
                            ind = test != -1
                            pattern[ind] = test[ind]
                        else:
                            found = False
                            break
                if found:
                    return pattern
    return None

################################################################################ 35
def roll_color(g):
    from_values = np.unique(g)
    to_values = np.roll(from_values, 1)

    sort_idx = np.argsort(from_values)
    idx = np.searchsorted(from_values, g, sorter = sort_idx)
    return to_values[sort_idx][idx]

################################################################################ 36
def get_all_transforms():
    return [
        lambda x: roll_color(x),
        lambda x: np.roll(x, -1, axis=0),
        lambda x: np.roll(x,  1, axis=0),
        lambda x: np.roll(x, -1, axis=1),
        lambda x: np.roll(x,  1, axis=1),
        lambda x: np.rot90(x.T, k=1),
        lambda x: np.rot90(x.T, k=2),
        lambda x: np.rot90(x.T, k=3),
        lambda x: np.rot90(x.T, k=4),
        lambda x: np.rot90(x, k=1),
        lambda x: np.rot90(x, k=2),
        lambda x: np.rot90(x, k=3),
        lambda x: x,
    ]

################################################################################ 37
def has_tiles_shape(g, shape, ignore=0):
    for size0b, size1b in [(g.shape[0], int(0.6*g.shape[1])), (int(0.6*g.shape[0]), g.shape[1])]:
        t = np.full((g.shape[0]+size0b, g.shape[1]+size1b), -1)
        t[:-size0b,:-size1b] = g
        t[t==ignore] = -1
        box_trim = trim_matrix_box(g,[ignore])
        min_size0 = 1
        min_size1 = 1
        if box_trim is not None and ignore != -1:
            xmin,ymin,xmax,ymax = box_trim
            t[xmin:xmax,ymin:ymax] = g[xmin:xmax,ymin:ymax]
            min_size0 = xmax-xmin
            min_size1 = ymax-ymin
        size0 = shape[0]
        size1 = shape[1]
        pattern = t[:size0,:size1].copy()
        found = True
        for d0 in range(0, t.shape[0]-size0, size0):
            for d1 in range(0, t.shape[1]-size1, size1):
                test = t[d0:d0+size0,d1:d1+size1]
                mask = (test != -1) & (pattern != -1)
                if np.array_equal(test[mask], pattern[mask]):
                    ind = test != -1
                    pattern[ind] = test[ind]
                else:
                    found = False
                    break
        if found:
            return pattern
    return None

################################################################################ 38
def check_tiles_shape(task, has_complete=0): 
    patterns = []
    for x, y in zip(*get_objects(task)):
        o_pattern = has_tiles(y,-1)
        if len(np.unique(x)) < 2 or o_pattern is None:
            return False
        found = False
        for c in [-1, *np.unique(x)]:
            pattern = has_tiles_shape(x, o_pattern.shape, c)
            if pattern is not None:
                pattern[pattern == -1] = c
                if has_complete:
                    for transform in get_all_transforms():
                        transformed_pattern = transform(pattern)
                        pred = np.tile(transformed_pattern, (x.shape[0]//transformed_pattern.shape[0]+2, x.shape[1]//transformed_pattern.shape[1]+2))
                        pred = pred[:x.shape[0],:x.shape[1]]
                        if np.array_equal(pred, y):
                            found = True
                            patterns.append(pattern)
                            break
                else:
                    found = True
                    patterns.append(pattern)
        if not found:
            return False
    return True

################################################################################ 39
def predict_tiles_shape(task, test_input):
    has_transforms = set()
    has_shapes = set()
    for x, y in zip(*get_objects(task)):
        o_pattern = has_tiles(y,-1)
        if len(np.unique(x)) < 2 or o_pattern is None:
            return []
        found = False
        for c in [-1, *np.unique(x)]:
            pattern = has_tiles_shape(x, o_pattern.shape, c)
            if pattern is not None:
                pattern[pattern == -1] = c
                for transform in get_all_transforms():
                    transformed_pattern = transform(pattern)
                    pred = np.tile(transformed_pattern, (x.shape[0]//transformed_pattern.shape[0]+2, x.shape[1]//transformed_pattern.shape[1]+2))
                    pred = pred[:x.shape[0],:x.shape[1]]
                    if np.array_equal(pred, y):
                        found = True
                        has_transforms.add(transform)
                        has_shapes.add(o_pattern.shape)
                        break
        if not found:
            return []
     
    preds = []
    for c in np.unique(test_input):
        for shape in has_shapes:
            pattern = has_tiles_shape(test_input, shape, c)
            if pattern is None:
                continue
            pattern[pattern == -1] = c

            pred = np.tile(pattern, (test_input.shape[0]//pattern.shape[0]+2, test_input.shape[1]//pattern.shape[1]+2))
            for transform in has_transforms:
                transformed_pattern = transform(pattern)
                pred = np.tile(transformed_pattern, (test_input.shape[0]//transformed_pattern.shape[0]+2, test_input.shape[1]//transformed_pattern.shape[1]+2))
                pred = pred[:test_input.shape[0],:test_input.shape[1]]
                preds.append(pred)
    return preds

################################################################################ 40

################################################################################ 
# 8 Functions - Via Tree
################################################################################ 1
def plot_objects(objects, titles=None):
    if titles is None:
        titles = np.full(len(objects), '')
    cmap = matplotlib.colors.ListedColormap(['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = matplotlib.colors.Normalize(vmin=0, vmax=9)
    fig, axs = plt.subplots(1, len(objects), figsize=(30,3), gridspec_kw = {'wspace':0.02, 'hspace':0.02}, squeeze=False)

    for i, (obj, title) in enumerate(zip(objects, titles)):
        obj = np.array(obj)
        axs[0,i].grid(True,which='both',color='lightgrey', linewidth=0.5)  
#         axs[i].axis('off')
        shape = ' '.join(map(str, obj.shape))
        axs[0,i].set_title(f"{title} {shape}")
        axs[0,i].set_yticks([x-0.5 for x in range(1+len(obj))])
        axs[0,i].set_xticks([x-0.5 for x in range(1+len(obj[0]))])
        axs[0,i].set_yticklabels([])     
        axs[0,i].set_xticklabels([])
        axs[0,i].imshow(obj, cmap=cmap, norm=norm)
    plt.show()

################################################################################ 2
def find_sub(matrix, sub):
    positions = []
    for x in range(matrix.shape[0]-sub.shape[0]+1):
        for y in range(matrix.shape[1]-sub.shape[1]+1):
            if np.equal(matrix[x:x+sub.shape[0], y:y+sub.shape[1]], sub).all():
                positions.append((x,y,x+sub.shape[0],y+sub.shape[1]))
    return positions

################################################################################ 3
def check_subitem(task):
    for key in ['train', 'test']:
        for obj in task[key]:
            if 'output' in obj:
                x = np.array(obj['input'])
                y = np.array(obj['output'])
                if len(find_sub(x, y)) == 0:
                    return False
    return True 

################################################################################ 4
def get_objects(task, has_train=True, has_test=False):
    xs, ys = [], []
    names = []
    if has_train:
        names.append('train')
    if has_test:
        names.append('test')
    for key in names:
        for obj in task[key]:
            xs.append(np.array(obj['input']))
            if 'output' not in obj:
                continue
            ys.append(np.array(obj['output']))
    return xs, ys

################################################################################ 5
def make_features(x, has_frame=False):
    def short_flattener(pred):
        str_pred = str([row for row in pred])
        str_pred = str_pred.replace(', ', '')
        str_pred = str_pred.replace('[[', '')
        str_pred = str_pred.replace('][', '|')
        str_pred = str_pred.replace(']]', '')
        return str_pred
    columns = pd.read_csv('arc/algo/features.tsv', sep='\t').columns
    columns = ["".join (c if c.isalnum() else "_" for c in str(col)) for col in columns]
    df = pd.DataFrame(np.fromfile('arc/algo/features.bin', dtype = [(col, '<f4') for col in columns]))
    
    df['rps4'] = False
    df['rps8'] = False
    labels = label(x, background=-1, connectivity=2)+2
    rps = regionprops(labels, cache=False)
    for r in rps:
        xmin, ymin, xmax, ymax = r.bbox
        df.loc[(df['xmin']==xmin)&(df['ymin']==ymin)&(df['xmax']==xmax)&(df['ymax']==ymax), 'rps8'] = True
    labels = label(x, background=-1, connectivity=1)+2
    rps = regionprops(labels, cache=False)
    for r in rps:
        xmin, ymin, xmax, ymax = r.bbox
        df.loc[(df['xmin']==xmin)&(df['ymin']==ymin)&(df['xmax']==xmax)&(df['ymax']==ymax), 'rps4'] = True
    
    if has_frame:
        df = df[(df['has_frame']==1)|(df['has_frame_1']==1)]
    for col in ['cnt_same_boxes', 'cnt_same_boxes_w_fr', 'cnt_same_boxes_wo_tr', 'ucnt_colors']:
        df[f"{col}_rank"]  = df[col].rank(method="dense")
        df[f"{col}_rank_"] = df[col].rank(method="dense", ascending=False)
    for col in df.columns:
        if 'iou' in col or col in ['has_frame', 'has_frame_1']:
            df[f"{col}_rank"]  = df.groupby([col])['area'].rank(method="dense")
            df[f"{col}_rank_"] = df.groupby([col])['area'].rank(method="dense", ascending=False)
    return df

################################################################################ 6
def decision_tree(train, test, test_input):
    print(type(train), type(test), type(test_input))
    y = train.pop('label')
    model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=4372).fit(train.drop(['xmin','ymin','xmax','ymax'], axis=1), y)
    preds = model.predict_proba(test.drop(['xmin','ymin','xmax','ymax'], axis=1))[:,1]
    
    indexes = np.argsort(preds)[::-1]
    objects,objs,titles = [],[],[]
    for score, (xmin,ymin,xmax,ymax) in zip(preds[indexes], test[['xmin','ymin','xmax','ymax']].astype(int).values[indexes]):
        obj = test_input[xmin:xmax,ymin:ymax]
        str_obj = flattener(obj.tolist())
        if str_obj not in objects:
            objects.append(str_obj)
            objs.append(obj)
            titles.append(str(np.round(score, 4)))
        if len(objects) > 10:
            break
    plot_objects(objs, titles) 
    return objects

################################################################################ 7
def tree1(train, test, test_input):
    y = train.pop('label')
    model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=4372).fit(train.drop(['xmin','ymin','xmax','ymax'], axis=1), y)
    preds = model.predict_proba(test.drop(['xmin','ymin','xmax','ymax'], axis=1))[:,1]
    
    indexes = np.argsort(preds)[::-1]
    objects,objs,titles = [],[],[]
    for score, (xmin,ymin,xmax,ymax) in zip(preds[indexes], test[['xmin','ymin','xmax','ymax']].astype(int).values[indexes]):
        obj = test_input[xmin:xmax,ymin:ymax]
        str_obj = flattener(obj.tolist())
        if str_obj not in objects:
            objects.append(str_obj)
            objs.append(obj)
            titles.append(str(np.round(score, 4)))
        if len(objects) > 1:
            break
    #plot_objects(objs, titles) 
    return objs

################################################################################ 8
def format_features(task):
    train = []
    for ttid, obj in enumerate(task['train']):
        x = np.array(obj['input'])
        y = np.array(obj['output'])
        df = make_features(x)
        df['label'] = False
#         df['tid'] = ttid
        positions = find_sub(x, y)
        for xmin,ymin,xmax,ymax in positions:
            df.loc[(df['xmin']==xmin)&(df['ymin']==ymin)&(df['xmax']==xmax)&(df['ymax']==ymax), 'label'] = True
        train.append(df)
    train = pd.concat(train).reset_index(drop=True)
    return train
################################################################################ 
# 31 Functions - Via Symmetry Repairing
################################################################################ 1
def Translation(x):
    n = len(x)
    k = len(x[0])
    Best_r = n
    Best_s = k
    x0 = np.array(x, dtype = int)
    for r in range(1,n):
        if x0[:n-r,:].tolist() == x0[r:,:].tolist():
            Best_r = r
            break
    for s in range(1,k):
        if x0[:,:k-s].tolist() == x0[:,s:].tolist():
            Best_s = s
            break
    if (Best_r, Best_s) == (n,k):
        return []
    r = Best_r
    s = Best_s
    E = {}
    for i in range(n):
        for j in range(k):
            u = i%r
            v = j%s
            p = (u,v)
            if p not in E:
                E[p] = [(i,j)]
            else:
                E[p] = E[p]+[(i,j)]
    Ans = []
    for p in E:
        item = E[p]
        if len(item) > 1:
            Ans.append(item)
    return Ans
                      
################################################################################ 2
def Translation1D(x):
    n = len(x)
    k = len(x[0])
 
    PossibleS = []
    
    for r in range(-n+1,n):
        for s in range(-k+1,k): 
            if s == 0 and r == 0:
                continue
            Equiv_Colors = {}
            possible = True
            for i in range(n):
                if possible == False:
                    break
                for j in range(k):
                    u = i*s-j*r 
                    v = (i*r+j*s+100*(r*r+s*s))%(r*r+s*s)
                    color = 0+  x[i][j]
                    if (u,v) not in Equiv_Colors:
                        Equiv_Colors[(u,v)] = color
                    elif color != Equiv_Colors[(u,v)]:
                        possible = False
                        break
            if possible:
                PossibleS.append((r,s))
                
    if len(PossibleS) == 0:
        return []
   
    Scores = []
    for p in PossibleS:
        r, s = p
        Scores.append((abs(r)+abs(s),p))
    Scores.sort()
    Best_r, Best_s = Scores[0][1]
    r = Best_r
    s = Best_s
    E = {}
    for i in range(n):
        for j in range(k):
            u = i*s-j*r
            v = (i*r+j*s+100*(r*r+s*s))%(r*r+s*s)
            p = (u,v)
            if p not in E:
                E[p] = [(i,j)]
            else:
                E[p] = E[p]+[(i,j)]
    Ans = []
    for p in E:
        item = E[p]
        if len(item) > 1:
            Ans.append(item)
    return Ans

################################################################################ 3   
def HorSym(x): 
    n = len(x)
    k = len(x[0])
    PossibleR = []
    
    for r in range(1,2*n-2): 
        possible = True
        for i in range(n):
            for j in range(k):
                i1 = r-i
                if i1 <0 or i1>=n:
                    continue
                color1 = x[i][j]
                color2 = x[i1][j]
                if color1 != color2:
                    possible = False
                    break
        if possible:
            PossibleR.append(r)
    if len(PossibleR) == 0:
        return []
    
    Scores = []
    for r in PossibleR:
        Scores.append((abs(r-n+1),r))
    Scores.sort()
    Best_r = Scores[0][1]
    r = Best_r
    Ans = []
    for i in range(n):
        for j in range(k):
            i1 = r-i
            if i1 <0 or i1 >= n:
                continue
            a = (i,j)
            b = (i1,j)
            i
            if [a,b] in Ans or [b,a] in Ans or a==b:
                continue
            Ans.append([a,b])
    return Ans

################################################################################ 4
def VertSym(x):
    n = len(x)
    k = len(x[0])
    PossibleS = []
 
    for s in range(1,2*k-2): 
        possible = True
        for i in range(n):
            for j in range(k):
                j1 = s-j
                if j1 <0 or j1>=k:
                    continue
                color1 = x[i][j]
                color2 = x[i][j1]
                if  color1 != color2:
                    possible = False
                    break
        if possible:
            PossibleS.append(s)
    if len(PossibleS) == 0:
        return []
    
    Scores = []
    for s in PossibleS:
        Scores.append((abs(s-k+1),s))
    Scores.sort()
    Best_s = Scores[0][1]
    s = Best_s
    Ans = []
    
    for i in range(n):
        for j in range(k):
            j1 = s-j
            if j1 <0 or j1 >= k:
                continue
            a = (i,j)
            b = (i,j1)
            if [a,b] in Ans or [b,a] in Ans or a==b:
                continue
            Ans.append([a,b])
    return Ans
    
    
################################################################################ 5
def NWSym(x):
    n = len(x)
    k = len(x[0])
    PossibleS= []
 
    for s in range(-k+2,n-1): 
        possible = True
        for i in range(n):
            for j in range(k):
                i1 = s+j
                j1 = -s+i
                
                if  i1 <0 or i1 >= n or j1 <0 or j1>=k:
                    continue
                color1 = x[i][j]
                color2 = x[i1][j1]
                if  color1 != color2:
                    possible = False
                    break
        if possible:
            PossibleS.append(s)
    if len(PossibleS) == 0:
        return []
    
    Scores = []
    for s in PossibleS:
        Scores.append((abs(s),s))
    Scores.sort()
    Best_s = Scores[0][1]
    s = Best_s
    Ans = []
    
    for i in range(n):
        for j in range(k):
            i1 = s+j
            j1 = -s+i
            if i1 <0 or i1 >=n or j1 <0 or j1 >= k:
                continue
            a = (i,j)
            b = (i1,j1)
            if [a,b] in Ans or [b,a] in Ans or a==b:
                continue
            Ans.append([a,b])
    return Ans
   
################################################################################ 6
def NESym(x):
    n = len(x)
    k = len(x[0])
    PossibleS = []
 
    for s in range(2,n+k-3):
        possible = True
        for i in range(n):
            for j in range(k):
                i1 = s-j
                j1 = s-i
                
                if  i1 <0 or i1 >= n or j1 <0 or j1>=k:
                    continue
                color1 = x[i][j]
                color2 = x[i1][j1]
                if  color1 != color2:
                    possible = False
                    break
        if possible:
            PossibleS.append(s)
    if len(PossibleS) == 0:
        return []
    
    Scores = []
    for s in PossibleS:
        Scores.append((abs(2*s-n-k-2),s))
    Scores.sort()
    Best_s = Scores[0][1]
    s = Best_s
    Ans = []
    
    for i in range(n):
        for j in range(k):
            i1 = s-j
            j1 = s-i
            if i1 <0 or i1 >=n or j1 <0 or j1 >= k:
                continue
            a = (i,j)
            b = (i1,j1)
            if [a,b] in Ans or [b,a] in Ans or a==b:
                continue
            Ans.append([a,b])
    return Ans
 
################################################################################ 7    
def Rotate180Sym(x):
 
    n = len(x)
    k = len(x[0])
 
    PossibleS = []
    
    for r in range(1,2*n-2):
        for s in range(1,2*k-2):#sum of indexes = r min 1, max = 2n-3
           
            possible = True
            for i in range(n):
                if possible == False:
                    break
                for j in range(k):
                    i1 = r-i
                    j1 = s-j
                    if j1 <0 or j1>=k or i1<0 or i1 >=n:
                        continue
                    color1 = x[i][j]
                    color2 = x[i1][j1]
                    if color1 != color2:
                        possible = False
                        break
            if possible:
                PossibleS.append((r,s))
    if len(PossibleS) == 0:
        return []
    
    Scores = []
    for p in PossibleS:
        r, s = p
        Scores.append((abs(r-n+1)+abs(s-k+1),p))
    Scores.sort()
    Best_r, Best_s = Scores[0][1]
    r = Best_r
    s = Best_s
    Ans = []
    
    for i in range(n):
        for j in range(k):
            i1 = r-i
            j1 = s-j
            if i1 <0 or i1 >=n or j1 <0 or j1 >= k:
                continue
            a = (i,j)
            b = (i1,j1)
            if [a,b] in Ans or [b,a] in Ans or a==b:
                continue
            Ans.append([a,b])
    return Ans

################################################################################ 8
def Rotate90Sym(x):
    n = len(x)
    k = len(x[0])
 
    PossibleS = []
   
    
    for r in range(1,2*n-2):
        for s in range(1,2*k-2): 
            if (s+r)%2 != 0:
                continue
            u = (r-s)//2
            v = (r+s)//2
            possible = True
            for i in range(n):
                if possible == False:
                    break
                for j in range(k):
                    Neig = [ (v-j, -u+i), (r-i,s-j), (j+u,v-i) ]
                    for i1, j1 in Neig:
                        if j1 <0 or j1>=k or i1<0 or i1 >=n:
                            continue
                        color1 = x[i][j]
                        color2 = x[i1][j1]
                        if color1 != color2:
                            possible = False
                            break
            if possible:
                PossibleS.append((r,s))
    
    if len(PossibleS) == 0:
        return []

    Scores = []
    for p in PossibleS:
        r, s = p
        Scores.append((abs(r-n+1)+abs(s-k+1),p))
    Scores.sort()
    Best_r, Best_s = Scores[0][1]
    r = Best_r
    s = Best_s
    u = (r-s)//2
    v = (r+s)//2
    Ans = []
    for i in range(n):
        for j in range(k):
            Neig = [ (v-j, -u+i), (r-i,s-j), (j+u,v-i) ]
            N2 = [(i,j)]
            for i1, j1 in Neig:
                if i1 <0 or i1 >=n or j1 <0 or j1 >= k:
                    continue
                else:
                    N2.append((i1,j1))
            N2.sort()
            if len(N2) ==1 or N2 in Ans:
                continue
            Ans.append(N2)
          
    return Ans

################################################################################ 9
def ReportColorChange(x,y):
    n= len(x)
    k = len(x[0])
    if len(x)!= len(y) or len(x[0]) != len(y[0]):
        return -1
    
    ChangingColors = []
    for i in range(n):
        for j in range(k):
            color1 = x[i][j]
            color2 = y[i][j]
            if color1 != color2 and color1 not in ChangingColors:
                ChangingColors.append(color1)
                
    for i in range(n):
        for j in range(k):
            color1 = x[i][j]
            color2 = y[i][j]
            if color1 != color2 and color2 in ChangingColors:
                return -1
    ColorsP = [] #Partially Disappearing Colors
    ColorsC = [] #Completely Disappearing Colors
    
    for i in range(n):
        for j in range(k):
            color1 = x[i][j]
            color2 = y[i][j]
            if color2 in ChangingColors and color2 not in ColorsP:
                ColorsP.append(color2)
    for c in ChangingColors:
        if c not in ColorsP:
            ColorsC.append(c)
    ColorsP.sort()
    ColorsC.sort()
    return ColorsP, ColorsC

################################################################################ 10
def Equivalence1(x,y,L):
    n= len(x)
    k = len(x[0])
    if len(x)!= len(y) or len(x[0]) != len(y[0]):
        return -1
    Report = ReportColorChange(x,y)
    if Report == -1:
        return -1
    ColorsP, ColorsC = Report
    F = [Translation, Translation1D, HorSym, VertSym, NWSym, NESym, Rotate90Sym, Rotate180Sym]
    A = [ ] 
    for i in L:
        f = F[i]
        A = A+ f(y)
    G = {}
    for i in range(n):
        for j in range(k):
            G[(i,j)] = (i,j)
    for Class in A:
        for p in Class:
            for q in Class:
                if G[p] == G[q]:
                    continue
                else:
                    a = G[p]
                    b = G[q]
                    if a < b:
                        G[q] = a
                    else:
                        G[p] = b
    H = {}
    for i in range(n):
        for j in range(k):
            p = (i,j)
            while G[p] != p:
                p = G[p]
            if p not in H:
                H[p] = [(i,j)]
            else:
                H[p] = H[p]+[(i,j)]
   
    for key in H: #key gives an equivalence class, where key is the smallest element
        item = H[key]
        Colors1 = [] #will contain the different colors of the eqivalence class
        Colors2 = [] # the new color of the orbit
        for i, j in item:
            c1 = x[i][j]
            c2 = y[i][j]
            if c1 not in Colors1:
                Colors1.append(c1)
                
            if c2 not in Colors2:
                Colors2.append(c2)
        
        if len(Colors2) != 1:
            # plot_picture(y)
            print("Problem")
            print(item)
            return -1
            
        # Allowed cases : 
        # 1 : both Colors1 and Colors2 contain one element and they agree
        # 2 : Colors1 contain exactly one element that is not partially or completely disappearing and that
        # agrees with the new color for the orbit            
            
        Colors3 = []
        for c in Colors1:
            if c not in ColorsP+ColorsC:
                Colors3.append(c)
        if (len(Colors3) > 1 or (len(Colors3) == 1 and Colors3 != Colors2) or 
            (len(Colors3) == 0 and Colors1 != Colors2)):
            return -1
       
    return 1
            
################################################################################ 11
Cut = 30
def Translation_Params(x, badcolor = 20):
    n = len(x)
    k = len(x[0])
    x0 = np.array(x, dtype = int)
    R = []
    S = []
    for r in range(1,n):
        possible = True
        for j in range(k):
            if possible == False:
                break
            for t in range(r):
                Col = []
                for i in range(t,n,r):
                    color = x[i][j]
                    if color != badcolor and color not in Col:
                        Col.append(color)
                if len(Col) >1:
                    possible = False
                    break
        if possible:
            R.append(r)
    
    for s in range(1,k):
        possible = True
        for i in range(n):
            if possible == False:
                break
            for t in range(s):
                Col = []
                for j in range(t,k,s):
                    color = x[i][j]
                    if color != badcolor and color not in Col:
                        Col.append(color)
                if len(Col) >1:
                    possible = False
                    break
        if possible:
            S.append(s)  
            
 
    R.append(n)
    S.append(k)
    Param = []
    Scores = []
    for t in range(1,n+k):
        for r in R:
            for s in S:
                if r+s == t:
                    Param.append((r,s))
                    Scores.append(t)
                    
    Sym_Level = 0
    if Param != []:
        r,s = Param[0]
        Sym_Level = 2 - r/n - s/k
   
                   
    return Param[:Cut], Scores[:Cut], Sym_Level
                      
################################################################################ 12
def Translation1D_Params(x, badcolor = 20):
    n = len(x)
    k = len(x[0])
 
    PossibleS = []
    
    for r in range(-n+1,n):
        for s in range(-k+1,k): 
            if s == 0 and r == 0:
                continue
            Equiv_Colors = {}
            possible = True
            for i in range(n):
                if possible == False:
                    break
                for j in range(k):
                    u = i*s-j*r 
                    v = (i*r+j*s+100*(r*r+s*s))%(r*r+s*s)
                    color = 0+  x[i][j]
                    if (u,v) not in Equiv_Colors or Equiv_Colors[(u,v)] == badcolor:
                        Equiv_Colors[(u,v)] = color
                    elif color != badcolor and color !=Equiv_Colors[(u,v)]:
                        possible = False
                        break
            if possible:
                PossibleS.append((r,s))
    
    if PossibleS == []:
        return [], [], 0
    Scores = []
    for p in PossibleS:
        r, s = p
        Scores.append((abs(r)+abs(s),p))
    Scores.sort()
    Ans = [item[1] for item in Scores]
    Penalty = [item[0] for item in Scores]
    
    Sym_Level = 0
    if Ans != []:
        r,s = Ans[0]
        Sym_Level = 1 -(abs(r)+abs(s))/(n+k)
 
    return Ans[:Cut], Penalty[:Cut], Sym_Level

################################################################################ 13
def HorSym_Params(x, badcolor = 20): 
    n = len(x)
    k = len(x[0])
    PossibleR = []
    
    for r in range(1,2*n-2): 
        possible = True
        for i in range(n):
            for j in range(k):
                i1 = r-i
                if i1 <0 or i1>=n:
                    continue
                color1 = x[i][j]
                color2 = x[i1][j]
                if color1 != color2 and color1 != badcolor and color2 != badcolor:
                    possible = False
                    break
        if possible:
            PossibleR.append(r)
            
    if PossibleR == []:
        return [], [], 0
    Scores = []
    
    for r in PossibleR:
        Scores.append((abs(r-n+1),r))
    
    Scores.sort()
    Ans = [item[1] for item in Scores]
    
    Penalty = [item[0] for item in Scores]
    
    Sym_Level = 0
    if Ans != []:
        r = Ans[0]
        Sym_Level = 1 - abs(r-n+1)/n
 
    return Ans[:Cut], Penalty[:Cut], Sym_Level

################################################################################ 14
def VertSym_Params(x, badcolor = 20):
    n = len(x)
    k = len(x[0])
    PossibleS = []
 
    for s in range(1,2*k-2):
        possible = True
        for i in range(n):
            for j in range(k):
                j1 = s-j
                if j1 <0 or j1>=k:
                    continue
                color1 = x[i][j]
                color2 = x[i][j1]
                if  color1 != color2 and color1 != badcolor and color2 != badcolor:
                    possible = False
                    break
        if possible:
            PossibleS.append(s)
            
    if PossibleS == []:
        return [], [], 0
    Scores = []
    for s in PossibleS:
        Scores.append((abs(s-k+1),s))
   
    Scores.sort()
    Ans = [item[1] for item in Scores]
    Penalty = [item[0] for item in Scores]
    
    Sym_Level = 0
    if Ans != []:
        s = Ans[0]
        Sym_Level = 1 - abs(s-k+1)/k
        
    return Ans[:Cut], Penalty[:Cut], Sym_Level
      
################################################################################ 15
def NWSym_Params(x, badcolor = 20):
    n = len(x)
    k = len(x[0])
    PossibleS= []
 
    for s in range(-k+2,n-1): 
        possible = True
        for i in range(n):
            for j in range(k):
                i1 = s+j
                j1 = -s+i
                
                if  i1 <0 or i1 >= n or j1 <0 or j1>=k:
                    continue
                color1 = x[i][j]
                color2 = x[i1][j1]
                if  color1 != color2:
                    possible = False
                    break
        if possible:
            PossibleS.append(s)
    if PossibleS == []:
        return [], [], 0
    Scores = []
    for s in PossibleS:
        Scores.append((abs(s),s))
   
    Scores.sort()
    Ans = [item[1] for item in Scores]
    Penalty = [item[0] for item in Scores]
    
    Sym_Level = 0
    if Ans != []:
        s = Ans[0]
        Sym_Level = 1 - abs(s)/(n+k)
        
    return Ans[:Cut], Penalty[:Cut], Sym_Level      

################################################################################ 16
def NESym_Params(x, badcolor = 20):
    n = len(x)
    k = len(x[0])
    PossibleS = []
 
    for s in range(2,n+k-3): 
        possible = True
        for i in range(n):
            for j in range(k):
                i1 = s-j
                j1 = s-i
                
                if  i1 <0 or i1 >= n or j1 <0 or j1>=k:
                    continue
                color1 = x[i][j]
                color2 = x[i1][j1]
                if  color1 != color2 and color1 != badcolor and color2 != badcolor:
                    possible = False
                    break
        if possible:
            PossibleS.append(s)
    if PossibleS == []:
        return [], [], 0
    Scores = []
    for s in PossibleS:
        Scores.append((abs(2*s-n-k-2),s))

   
    Scores.sort()
    Ans = [item[1] for item in Scores]
    Penalty = [item[0] for item in Scores]
    
    Sym_Level = 0
    if Ans != []:
        s = Ans[0]
        Sym_Level = 1 - abs(2*s-n-k-2)/(n+k)
    
    return Ans[:Cut], Penalty[:Cut], Sym_Level
  
################################################################################ 17    
def Rotate180Sym_Params(x, badcolor = 20):
 
    n = len(x)
    k = len(x[0])
 
    PossibleS = []
    
    for r in range(1,2*n-2):
        for s in range(1,2*k-2):
           
            possible = True
            for i in range(n):
                if possible == False:
                    break
                for j in range(k):
                    i1 = r-i
                    j1 = s-j
                    if j1 <0 or j1>=k or i1<0 or i1 >=n:
                        continue
                    color1 = x[i][j]
                    color2 = x[i1][j1]
                    if color1 != color2 and color1 != badcolor and color2 != badcolor:
                        possible = False
                        break
            if possible:
                PossibleS.append((r,s))
                
    if PossibleS == []:
        return [], [], 0
    Scores = []
    for p in PossibleS:
        r, s = p
        Scores.append((abs(r-n+1)+abs(s-k+1),p))
     

   
    Scores.sort()
    Ans = [item[1] for item in Scores]
    Penalty = [item[0] for item in Scores]
    
    Sym_Level = 0
    if Ans != []:
        r, s = Ans[0]
        Sym_Level = 1 - ((abs(r-n+1)+abs(s-k+1))/(n+k))
        
    return Ans[:Cut], Penalty[:Cut], Sym_Level

################################################################################ 18  
def Rotate90Sym_Params(x, badcolor = 20):
    n = len(x)
    k = len(x[0])
 
    PossibleS = []
   
    
    for r in range(1,2*n-2):
        for s in range(1,2*k-2): 
            if (s+r)%2 != 0:
                continue
            u = (r-s)//2
            v = (r+s)//2
            possible = True
            for i in range(n):
                if possible == False:
                    break
                for j in range(k):
                    Neig = [ (v-j, -u+i), (r-i,s-j), (j+u,v-i) ]
                    for i1, j1 in Neig:
                        if j1 <0 or j1>=k or i1<0 or i1 >=n:
                            continue
                        color1 = x[i][j]
                        color2 = x[i1][j1]
                        if color1 != color2 and color1 !=badcolor and color2 !=badcolor:
                            possible = False
                            break
            if possible:
                PossibleS.append((r,s))
    if PossibleS == []:
        return [], [], 0
    Scores = []
    for p in PossibleS:
        r, s = p
        Scores.append((abs(r-n+1)+abs(s-k+1),p))
     

   
    Scores.sort()
    Ans = [item[1] for item in Scores]
    Penalty = [item[0] for item in Scores]
    
    Sym_Level = 0
    if Ans != []:
        r, s = Ans[0]
        Sym_Level = 1 - ((abs(r-n+1)+abs(s-k+1))/(n+k))
        
    return Ans[:Cut], Penalty[:Cut], Sym_Level 

################################################################################ 19
def SymScore(x,First_P):
    F = [Translation_Params, Translation1D_Params, HorSym_Params, VertSym_Params, 
         NWSym_Params, NESym_Params, Rotate90Sym_Params, Rotate180Sym_Params]
    Score = 0
    for s in First_P:
        f = F[s]
        value = f(x)[2]
        Score += value
    return Score
        
################################################################################ 20 
def Solvable2(task):
    V = [[0], [1], [0,1], [2], [3], [2,3], [4], [5], [4,5], [6], [7], [0,2], [0,3], [0,2,3],[0,4], [0,5],
         [0,4,5], [0,6], [0,7], [2,3,6], [0,2,3,6]]
    
    W = [1.5, 1, 2.3, 1, 1, 1.5, 1, 1, 1.5, 1, 1, 2.3, 2.3, 2.5, 2, 2, 2.3, 2, 2, 2, 3]
    
    Ans = []
    Input = [Defensive_Copy(x) for x in task[0]]
    Output = [Defensive_Copy(y) for y in task[1]]
    Test_Example = Input[:-1]
    for x,y in zip(Input, Output):
        if ReportColorChange(x,y) == -1:
            return -1
    
    F = [Translation_Params, Translation1D_Params, HorSym_Params, VertSym_Params, 
         NWSym_Params, NESym_Params, Rotate90Sym_Params, Rotate180Sym_Params]
    
    Input = Input[:-1]
    Can_Be_Solved = False
    Solutions = []
    
    
    for i  in range(len(V)):
        t = V[i]
        if len(t) >2:
            continue
        possible = True
        Sym_Scores = []
        for x, y in zip(Input, Output):
            
            if Equivalence1(x,y,t) == -1:
                possible = False
                break
            Local_Score = 0
            for s in t:
                f = F[s]
                value = f(y)[2]
                Local_Score+=value
            Local_Score = Local_Score/W[i]
            Sym_Scores.append(Local_Score)

        
            
        if possible:
            Can_Be_Solved = True
            Solutions.append((min(Sym_Scores), t))
    Solutions.sort()
    Solutions.reverse()
    Solutions2 = [ item[1] for item in Solutions]
    # if Solutions2 != []:
        # print("Symmetries found : " ,Solutions2)
    if Can_Be_Solved :
        return Solutions2
    
    return -1    

################################################################################ 21
Cut = 30
def Translation_Eq(x, Param):
    r, s = Param
    n = len(x)
    k = len(x[0])
   
    E = {}
    for i in range(n):
        for j in range(k):
            u = i%r
            v = j%s
            p = (u,v)
            if p not in E:
                E[p] = [(i,j)]
            else:
                E[p] = E[p]+[(i,j)]
    Ans = []
    for p in E:
        item = E[p]
        if len(item) > 1:
            Ans.append(item)
    return Ans
                      
################################################################################ 22
def Translation1D_Eq(x, Param):
    n = len(x)
    k = len(x[0])
    r, s = Param
    E = {}
    for i in range(n):
        for j in range(k):
            u = i*s-j*r
            v = (i*r+j*s+100*(r*r+s*s))%(r*r+s*s)
            p = (u,v)
            if p not in E:
                E[p] = [(i,j)]
            else:
                E[p] = E[p]+[(i,j)]
    Ans = []
    for p in E:
        item = E[p]
        if len(item) > 1:
            Ans.append(item)
    return Ans

################################################################################ 23
def HorSym_Eq(x, Param): # symmetric for reflection along a line parallel to the x axis
    n = len(x)
    k = len(x[0])
    r = Param
    Ans = []
    for i in range(n):
        for j in range(k):
            i1 = r-i
            if i1 <0 or i1 >= n:
                continue
            a = (i,j)
            b = (i1,j)
            i
            if [a,b] in Ans or [b,a] in Ans or a==b:
                continue
            Ans.append([a,b])
    return Ans

################################################################################ 24
def VertSym_Eq(x, Param):
    n = len(x)
    k = len(x[0])
    
    s = Param
    Ans = []
    
    for i in range(n):
        for j in range(k):
            j1 = s-j
            if j1 <0 or j1 >= k:
                continue
            a = (i,j)
            b = (i,j1)
            if [a,b] in Ans or [b,a] in Ans or a==b:
                continue
            Ans.append([a,b])
    return Ans
      
################################################################################ 25
def NWSym_Eq(x, Param):
    n = len(x)
    k = len(x[0])
    s = Param
    Ans = []
    for i in range(n):
        for j in range(k):
            i1 = s+j
            j1 = -s+i
            if i1 <0 or i1 >=n or j1 <0 or j1 >= k:
                continue
            a = (i,j)
            b = (i1,j1)
            if [a,b] in Ans or [b,a] in Ans or a==b:
                continue
            Ans.append([a,b])
    return Ans
   
################################################################################ 26
def NESym_Eq(x, Param):
    n = len(x)
    k = len(x[0])
    s = Param
    Ans = []
    for i in range(n):
        for j in range(k):
            i1 = s-j
            j1 = s-i
            if i1 <0 or i1 >=n or j1 <0 or j1 >= k:
                continue
            a = (i,j)
            b = (i1,j1)
            if [a,b] in Ans or [b,a] in Ans or a==b:
                continue
            Ans.append([a,b])
    return Ans
 
################################################################################ 27    
def Rotate180Sym_Eq(x, Param):
 
    n = len(x)
    k = len(x[0])
    r, s = Param
 
    Ans = []
    
    for i in range(n):
        for j in range(k):
            i1 = r-i
            j1 = s-j
            if i1 <0 or i1 >=n or j1 <0 or j1 >= k:
                continue
            a = (i,j)
            b = (i1,j1)
            if [a,b] in Ans or [b,a] in Ans or a==b:
                continue
            Ans.append([a,b])
    return Ans
   
################################################################################ 28    
def Rotate90Sym_Eq(x, Param):
    n = len(x)
    k = len(x[0])
    r, s = Param
 
    u = (r-s)//2
    v = (r+s)//2
    Ans = []
    for i in range(n):
        for j in range(k):
            Neig = [ (v-j, -u+i), (r-i,s-j), (j+u,v-i) ]
            N2 = [(i,j)]
            for i1, j1 in Neig:
                if i1 <0 or i1 >=n or j1 <0 or j1 >= k:
                    continue
                else:
                    N2.append((i1,j1))
            N2.sort()
            if len(N2) ==1 or N2 in Ans:
                continue
            Ans.append(N2)
          
    return Ans

################################################################################ 29
def Make_Picture(x, Relations, badcolor):
    # returns -1 if there are conflicts between non-badcolors
    n = len(x)
    k = len(x[0])
    A = Relations
    G = {}
    x0 = np.array(x, dtype = int)
    for i in range(n):
        for j in range(k):
            G[(i,j)] = (i,j)
    for Class in A:
        for p in Class:
            for q in Class:
                if G[p] == G[q]:
                    continue
                else:
                    a = G[p]
                    b = G[q]
                    if a < b:
                        G[q] = a
                    else:
                        G[p] = b
    H = {}
    for i in range(n):
        for j in range(k):
            p = (i,j)
            while G[p] != p:
                p = G[p]
            if p not in H:
                H[p] = [(i,j)]
            else:
                H[p] = H[p]+[(i,j)]
   
    for key in H:
        item = H[key]
        Colors = []
        for i, j in item:
            c = x[i][j]
            if c not in Colors:
                Colors.append(c)
        if len(Colors) <= 1:
            continue #nothing to recolor
        if len(Colors) > 2:
            return -1 #collision
        if len(Colors) ==2 and badcolor not in Colors:
            return -1 #collision
        if len(Colors) == 2 and badcolor == Colors[0]:
            fillcolor = Colors[1]
        else :
            fillcolor = Colors[0]
        for i, j in item:
            x0[i,j] = fillcolor
    return x0.tolist()
            
################################################################################ 30   
def Proba(task, Bad, First_P):
    Input = [Defensive_Copy(x) for x in task[0]]
    Output = [Defensive_Copy(y) for y in task[1]]
    Test_Picture = Input[-1]
    Input = Input[:-1]
    
    V1 = [Translation_Params, Translation1D_Params, HorSym_Params, VertSym_Params, NWSym_Params, 
          NESym_Params, Rotate90Sym_Params, Rotate180Sym_Params]
    
    V2 = [Translation_Eq, Translation1D_Eq, HorSym_Eq, VertSym_Eq, NWSym_Eq, 
          NESym_Eq, Rotate90Sym_Eq, Rotate180Sym_Eq]
    
    Ans = []
    
    if len(First_P) == 1:
        S = First_P[0]
        f = V1[S]
        g = V2[S]
        Params = f(Test_Picture,Bad)[0]
        for p in Params:
            A = g(Test_Picture,p)
            picture = Make_Picture(Test_Picture, A, Bad)
            if picture != -1 and picture not in Ans and np.min(picture) != np.max(picture):
                        Ans.append(picture)
                    
    if len(First_P) == 2:
        S1 = First_P[0]
        S2 = First_P[1]
        f1 = V1[S1]
        f2 = V1[S2]
        g1 = V2[S1]
        g2 = V2[S2]
        Params1 = f1(Test_Picture, Bad)[0]
        Params2 = f2(Test_Picture, Bad)[0]
        for K in range(6):
            for i in range(len(Params1)):
                for j in range(len(Params2)):
                    if i+j == K:
                        p1 = Params1[i]
                        p2 = Params2[j]
                        A1 = g1(Test_Picture,p1)
                        A2 = g2(Test_Picture,p2)
                        A = A1+A2
                        picture = Make_Picture(Test_Picture,A,Bad)
                        if picture != -1 and picture not in Ans and np.min(picture) != np.max(picture):
                            Ans.append(picture)
      
   
    if len(Ans) == 0:
        return -1
    return Ans[:6]
    
################################################################################ 31    
def symmetry_repairing(task):
    Input = [Defensive_Copy(x) for x in task[0]]
    Output = [Defensive_Copy(y) for y in task[1]]
    Test_Picture = Input[-1]
    Input = Input[:-1]
    Colors = []
    for x, y in zip(Input, Output):
        if len(x) != len(y) or len(x[0]) != len(y[0]):
            return -1
        Disappearing = []
        n = len(x)
        k = len(x[0])
        for i in range(n):
            for j in range(k):
                color1 = 0+x[i][j]
                color2 = 0 + y[i][j]
                if color2 != color1 and color1 not in Disappearing:
                    Disappearing.append(color1)
        if len(Disappearing) > 1:
            return -1
        if len(Disappearing) == 1:
            c = Disappearing[0]
        if c not in Colors:
            Colors.append(c)
    AAA = Solvable2(task)
    
    if AAA == -1:
        return -1
  
    if len(Colors) == 1:
        C2 = [Colors[0]]
    else:
        C2 = []
        for row in Test_Picture:
            for c in row:
                if c not in C2:
                    C2.append(c)
                    
   
    Ans = []
    First_P = -1
    
    for P in AAA:
        for c in C2:
            Candidates = Proba(task, c, P)
            if Candidates != -1:
                Ans = Ans+Candidates[:6]
        if Ans != []:
            First_P = P
            break
            
    if Ans == []:
        return -1
    
    Scores = []
    for picture in Ans:
        S = SymScore(picture, First_P)
        Scores.append((S,picture))
    Scores.sort()
    Scores.reverse()
    Ans2 =[]
    for _, picture in Scores:
        if picture not in Ans2:
            Ans2.append(picture)
    return Ans2[: 3]
################################################################################ 
# 9 Functions - Via Colors Counter 
################################################################################ 1
def Defensive_Copy(A): 
    n = len(A)
    k = len(A[0])
    L = np.zeros((n,k), dtype = int)
    for i in range(n):
        for j in range(k):
            L[i,j] = 0 + A[i][j]
    return L.tolist()

################################################################################ 2
def Create(task, task_id=0):
    n = len(task['train'])
    Input = [Defensive_Copy(task['train'][i]['input']) for i in range(n)]
    Output = [Defensive_Copy(task['train'][i]['output']) for i in range(n)]
    Input.append(Defensive_Copy(task['test'][task_id]['input']))
    return Input, Output

################################################################################ 3
def colors_counter(task):
    Input = task[0]
    Output = task[1]
    Test_Picture = Input[-1]
    Input = Input[:-1]
    N = len(Input)
    
    for x, y in zip(Input, Output):
        if len(x) != len(y) or len(x[0]) != len(y[0]):
            return -1
        
    Best_Dict = -1
    Best_Q1 = -1
    Best_Q2 = -1
    Best_v = -1
    # v ranges from 0 to 3. This gives an extra flexibility of measuring distance from any of the 4 corners
    Pairs = []
    for t in range(15):
        for Q1 in range(1,8):
            for Q2 in range(1,8):
                if Q1+Q2 == t:
                    Pairs.append((Q1,Q2))
                    
    for Q1, Q2 in Pairs:
        for v in range(4):
    
  
            if Best_Dict != -1:
                continue
            possible = True
            Dict = {}
                      
            for x, y in zip(Input, Output):
                n = len(x)
                k = len(x[0])
                for i in range(n):
                    for j in range(k):
                        if v == 0 or v ==2:
                            p1 = i%Q1
                        else:
                            p1 = (n-1-i)%Q1
                        if v == 0 or v ==3:
                            p2 = j%Q2
                        else :
                            p2 = (k-1-j)%Q2
                        color1 = x[i][j]
                        color2 = y[i][j]
                        if color1 != color2:
                            rule = (p1, p2, color1)
                            if rule not in Dict:
                                Dict[rule] = color2
                            elif Dict[rule] != color2:
                                possible = False
            if possible:
                
                # Let's see if we actually solve the problem
                for x, y in zip(Input, Output):
                    n = len(x)
                    k = len(x[0])
                    for i in range(n):
                        for j in range(k):
                            if v == 0 or v ==2:
                                p1 = i%Q1
                            else:
                                p1 = (n-1-i)%Q1
                            if v == 0 or v ==3:
                                p2 = j%Q2
                            else :
                                p2 = (k-1-j)%Q2
                           
                            color1 = x[i][j]
                            rule = (p1,p2,color1)
                            
                            if rule in Dict:
                                color2 = 0 + Dict[rule]
                            else:
                                color2 = 0 + y[i][j]
                            if color2 != y[i][j]:
                                possible = False 
                if possible:
                    Best_Dict = Dict
                    Best_Q1 = Q1
                    Best_Q2 = Q2
                    Best_v = v
                
                
    if Best_Dict == -1:
        return -1 #meaning that we didn't find a rule that works for the traning cases
    
    #Otherwise there is a rule: so let's use it:
    n = len(Test_Picture)
    k = len(Test_Picture[0])
    
    answer = np.zeros((n,k), dtype = int)
   
    for i in range(n):
        for j in range(k):
            if Best_v == 0 or Best_v ==2:
                p1 = i%Best_Q1
            else:
                p1 = (n-1-i)%Best_Q1
            if Best_v == 0 or Best_v ==3:
                p2 = j%Best_Q2
            else :
                p2 = (k-1-j)%Best_Q2
           
            color1 = Test_Picture[i][j]
            rule = (p1, p2, color1)
            if (p1, p2, color1) in Best_Dict:
                answer[i][j] = 0 + Best_Dict[rule]
            else:
                answer[i][j] = 0 + color1
                                          
            
    return answer.tolist()

################################################################################ 4
def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

################################################################################ 5
def plot_task(task):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=200)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Train-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Train-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    for i, t in enumerate(task["test"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Test-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Test-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    
    plt.tight_layout()
    plt.show()
    
################################################################################ 6
def plot_task1(task):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=200)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Train-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Train-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    for i, t in enumerate(task["test"]):
        t_in = np.array(t["input"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Test-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        fig_num += 1
    
    plt.tight_layout()
    plt.show()  
    
################################################################################ 7 
cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)
color_list = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]

# plt.figure(figsize=(5, 2), dpi=200)
# plt.imshow([list(range(10))], cmap=cmap, norm=norm)
# plt.xticks(list(range(10)))
# plt.yticks([])
# plt.show()

# ..................................................................................... 1
def ganswer_answer(ganswer):
    
    answer = []
    for j in range(len(ganswer)):
        ganswer_j = ganswer[j].tolist()
        
        if (ganswer_j not in answer):  
            answer.append(ganswer_j)   
            
    return answer

# ..................................................................................... 2
def ganswer_answer_1(ganswer):
    
    answer = []
    for j in range(len(ganswer)):
        ganswer_j = ganswer[j]
        
        if (ganswer_j not in answer):  
            answer.append(ganswer_j)   
            
    return answer

# ..................................................................................... 3
def prn_plus(prn, answer):
    
    for j in range(len(answer)):
        prn = prn + [answer[j]]  
        
        if (j == 0):
            prn = prn + [answer[j]]
            
    return prn

# ..................................................................................... 4
def prn_select(prn): 
    if (len(prn) > 1):
        
        value_list = []
        string_list = []
        for el in prn:
            value = 0
            for i in range(len(prn)):
                if el == prn[i]:
                    value +=1
            value_list.append(value)
            string_list.append(str(el))    
        
        prn_df  = pd.DataFrame({'prn': prn , 'value': value_list, 'string': string_list}) 
        prn_df1 = prn_df.drop_duplicates(subset=['string'])
        prn_df2 = prn_df1.sort_values(by='value', ascending=False)   
        
        prn = prn_df2['prn'].values.tolist()[0]
        
    return prn

# ..................................................................................... 5
cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)
color_list = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]

def plot_pic(x):
    plt.imshow(np.array(x), cmap=cmap, norm=norm)
    plt.show()

def run_main_solvers(examples, test_input): 
    task = {
        "train": examples,
        "test": [
            {
                "input": test_input
            }
        ]
    }
    prn = []
    
    test_input = np.array(test_input)
    # ............................................................................... 1 - Different Solvers       
    if check_repeating(task, True): 
        ganswer = predict_repeating(test_input)
        
        if (ganswer!= []):
            answer = ganswer_answer(ganswer)
            prn = prn_plus(prn, answer) 
    
    # ________________________________________________________
    if check_grid(task) and check_sub_grid_2x(task): 
        ganswer = predict_transforms_grid_2x(task, test_input)
        
        if (ganswer!= []):
            answer = ganswer_answer(ganswer)
            prn = prn_plus(prn, answer) 
    
    # ________________________________________________________
    if check_grid(task) and check_chess(task, False, True): 
        ganswer = predict_chess(grid_filter(test_input))
        
        if (ganswer!= []):
            answer = ganswer_answer(ganswer)
            prn = prn_plus(prn, answer) 
    
    # ________________________________________________________
    if check_tiles_shape(task, True): 
        ganswer = predict_tiles_shape(task, test_input)
        
        if (ganswer!= []):
            answer = ganswer_answer(ganswer)
            prn = prn_plus(prn, answer) 
    
    # ________________________________________________________
    if check_grid(task) and check_grid_transforms(task): 
        ganswer = predict_grid_transforms(task, test_input)
        
        if (ganswer!= []):
            answer = ganswer_answer(ganswer)
            prn = prn_plus(prn, answer) 
    
    # ________________________________________________________
    if check_sub_mask(task): 
        ganswer = predict_repeating_mask(test_input)
        
        if (ganswer!= []):
            answer = ganswer_answer(ganswer)
            prn = prn_plus(prn, answer) 

    # ............................................................................... 2 - Sklearn tree          
    if check_subitem(task):
        print(task)
        train_t = format_features(task)
        test_t = make_features(test_input) 
        ganswer = tree1(train_t, test_t, test_input)  
        
        if (ganswer!= []):
            answer = ganswer_answer(ganswer)
            prn = prn_plus(prn, answer) 

    # ............................................................................... 3 - Symmetry Repairing       
    basic_task = Create(task) 
    ganswer = symmetry_repairing(basic_task)   

    if (ganswer != -1):
        answer = ganswer_answer_1(ganswer)
        prn = prn_plus(prn, answer) 

    # ............................................................................... 4 - Colors Counter
    basic_task = Create(task) 
    answer = colors_counter(basic_task)   
    
    if (answer != -1):
        answer = [answer]
        prn = prn_plus(prn, answer) 
    
    # ...............................................................................  Conclusion
    if (prn != []):  
        prn = prn_select(prn)
        render_grid(prn[0])
        return prn[0]
        # plot_pic(prn[0])

    # ............................................................................... 
    # display(sub_solver)    
    return [[]]

# ...............................................................................    

# %% [code]

# This code has been copied from the 2020 Abstraction and Reasoning Corpus competition 
# https://www.kaggle.com/competitions/abstraction-and-reasoning-challenge
# Credits go to icecuber and his original write up and published notebook that can be found here:
# https://www.kaggle.com/competitions/abstraction-and-reasoning-challenge/discussion/154597

# We have slightly adapted it such that it adheres to the changed rules of ARC Prize 2024
# Note that the score is slightly lower as in ARC Prize 2024 we are only allowed to make 2 instead of 3 attempts per task

# print("Running")
# from subprocess import Popen, PIPE, STDOUT
# from glob import glob

# import os
# import json

#######################################################################################
# Adapt ARC Prize 2024 files to work with Abstraction and Resoning Corpus 2020 rules ##
#######################################################################################

# # Load the JSON content
# json_file_path = '/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json'  
# with open(json_file_path, 'r') as file:
#     data = json.load(file)

# # Create the 'test' directory
# output_dir = '/kaggle/working/abstraction-and-reasoning-challenge/test'  
# os.makedirs(output_dir, exist_ok=True)

# # Split the JSON content into individual files
# for task_id, task_data in data.items():
#     output_file_path = os.path.join(output_dir, f'{task_id}.json')
#     with open(output_file_path, 'w') as output_file:
#         json.dump(task_data, output_file, indent=4)

# # Verify the files have been created 
# print(f"Created ARC files in '{output_dir}':")
# print(os.listdir(output_dir))

############################################
# Beginning of icecuber's original solution#
##########################################

# if open("../input/arc-solution-source-files-by-icecuber/version.txt").read().strip() == "671838222":
#   print("Dataset has correct version")
# else:
#   print("Dataset version not matching!")
#   assert(0)

# def mySystem(cmd):
#     print(cmd)
#     process = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
#     for line in iter(process.stdout.readline, b''):
#         print(line.decode("utf-8"), end='')
#     assert(process.wait() == 0)
    
# dummy_run = False


# for fn in glob("/kaggle/working/abstraction-and-reasoning-challenge/test/*.json"):
#   if "017c7c7b" in fn:
#     print("Making dummy submission")
#     f = open("old_submission.csv", "w")
#     f.write("output_id,output\n")
#     f.close()
#     dummy_run = True


# if not dummy_run:
#   mySystem("cp -r ../input/arc-solution-source-files-by-icecuber ./absres-c-files")
#   mySystem("cd absres-c-files; make -j")
#   mySystem("cd absres-c-files; python3 safe_run.py")
#   mySystem("cp absres-c-files/submission_part.csv old_submission.csv")
#   mySystem("tar -czf store.tar.gz absres-c-files/store")
#   mySystem("rm -r absres-c-files")

# # Function to translate from old submission format (csv) to new one (json)
# def translate_submission(file_path):
#     # Read the original submission file
#     with open(file_path, 'r') as file:
#         lines = file.readlines()

#     submission_dict = {}

#     for line in lines[1:]:  # Skip the header line
#         output_id, output = line.strip().split(',')
#         task_id, output_idx = output_id.split('_')
#         predictions = output.split(' ')  # Split predictions based on ' '
        
#         # Take only the first two predictions
#         if len(predictions) > 2:
#             predictions = predictions[:2]

#         processed_predictions = []
#         for pred in predictions:
#             if pred:  # Check if pred is not an empty string
#                 pred_lines = pred.split('|')[1:-1]  # Remove empty strings from split
#                 pred_matrix = [list(map(int, line)) for line in pred_lines]
#                 processed_predictions.append(pred_matrix)

#         attempt_1 = processed_predictions[0] if len(processed_predictions) > 0 else []
#         attempt_2 = processed_predictions[1] if len(processed_predictions) > 1 else []
#         #if len(processed_predictions) > 2:
#         #   attempt_2 = processed_predictions[2]

#         if task_id not in submission_dict:
#             submission_dict[task_id] = []

#         attempt_dict = {
#             "attempt_1": attempt_1,
#             "attempt_2": attempt_2
#         }

#         if output_idx == '0':
#             submission_dict[task_id].insert(0, attempt_dict)
#         else:
#             submission_dict[task_id].append(attempt_dict)
    
#     # Write to the new json file
#     with open('submission2.json', 'w') as file:
#         json.dump(submission_dict, file, indent=4)
#     return submission_dict

# sub_solver2 = translate_submission('/kaggle/working/old_submission.csv')
# print("Done")

################################################################################ 

# def isdefault(tt):
#     return np.all(tt == np.asarray([[0, 0], [0, 0]]))

# for task_id in sub_solver:
#     for rep_id in range(len(sub_solver[task_id])):
#         t1a1 = sub_solver[task_id][rep_id]["attempt_1"]
#         t1a2 = sub_solver[task_id][rep_id]["attempt_2"]
#         if task_id not in sub_solver2:
#             print('skipping', task_id)
#             continue
#         t2a1 = sub_solver2[task_id][rep_id]["attempt_1"]
#         t2a2 = sub_solver2[task_id][rep_id]["attempt_2"]
        
#         if not t1a1 or isdefault(t1a1):
#             print('rewriting t1a1', task_id)
#             if t2a1:
#                 sub_solver[task_id][rep_id]["attempt_1"] = t2a1
#                 sub_solver[task_id][rep_id]["attempt_2"] = t2a2
#         elif not t1a2 or isdefault(t1a2):
#             print('rewriting t1a2', task_id)
#             if t2a1:
#                 sub_solver[task_id][rep_id]["attempt_2"] = t2a1
#         else:
#             print('leaving as is', task_id)

# with open('submission3.json', 'w') as file:
#     json.dump(sub_solver, file, indent=4)
                
################################################################################ 

# DEBUG = False

# def dbg(msg):
#     if DEBUG:
#         print(msg)
# MY_TEST = {}
# MY_TEST

if 0:
    json_sample_path = '/kaggle/input/arc-prize-2024/sample_submission.json'  
    with open(json_sample_path, 'r') as f:
        sample = json.load(f)
    for sample_id, sample_data in sample.items():
        print('SAMPLE:')
        print(sample_id)
        print(sample_data)
        break

def lib_sel_obj_in_square(xin, col, sx, sy):
    fld = np.copy(xin)
    if fld[sx,sy] != col:
        return -1,-1,-1,-1
    bx=sx
    ex=sx
    by=sy
    ey=sy
    
    for it in range(30):
        upd=False
        if bx > 0:
            if np.any(fld[bx-1,by-1 if by>0 else by:ey+1 if ey<fld.shape[1]-1 else ey] == col):
                bx=bx-1
                upd=True
        if by > 0:
            if np.any(fld[bx-1 if bx>0 else bx:ex+1 if ex<fld.shape[0] else ex,by-1] == col):
                by=by-1
                upd=True
        if ex < fld.shape[0]-1:
            if np.any(fld[ex+1,by-1 if by>0 else by:ey+1 if ey<fld.shape[1]-1 else ey] == col):
                ex=ex+1
                upd=True
        if ey < fld.shape[1]-1:
            if np.any(fld[bx-1 if bx>0 else bx:ex+1 if ex<fld.shape[0] else ex,ey+1] == col):
                ey=ey+1
                upd=True
        if not upd:
            break
    return bx,by,ex+1,ey+1
    
def lib_objects_by_color(xin, col):
    fld = np.copy(xin)
    mask = np.zeros(xin.shape)
    objs = []
    for i in range(xin.shape[0]):
        for j in range(xin.shape[1]):
            if not mask[i,j] and fld[i,j] == col:
                bx,by,ex,ey = lib_sel_obj_in_square(fld,col,i,j)
                mask[bx:ex,by:ey] = 1
                objs.append((bx,by,ex,ey))
    return objs

def lib_color_counts_dict(xin):
    res = []
    for c in range(10):
        r = np.where(xin == c, 1, 0).sum()
        if r > 0:
            res.append((c, r))
    return sorted(res, key=lambda x: x[1], reverse=True)
                   
def lib_get_all_objects(xin):
    cols = lib_color_counts_dict(xin)
    objs = {}
    for col, cnt in cols:
        objs[col] = lib_objects_by_color(xin, col)
    return objs
    
def lib_objects_hor_pattern(objs_info):
    patterns = {}
    for col, objs in objs_info.items():
        patterns[col] = [[1]]
        if len(objs) == 1:
            continue
        for i in range(1, len(objs)):
            if objs[i][0] == objs[i-1][0] and objs[i][2] == objs[i-1][2]:
                patterns[col][len(patterns[col])-1].append(1)
            else:
                patterns[col].append([1])
    return patterns

def flattener(pred):
    str_pred = str([list([int(x) for x in row]) for row in list(pred)])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

#Get stats array with number of each color in grid
def get_stats(d1):
    counts = [0] * 10
    for c in d1:
        for cc in c:
            counts[int(cc)] += 1
    return counts

# get color numbers in grid
def lib_get_colors_all(d1):
    res = []
    dd = get_stats(d1)
    for x in range(0, len(dd)): #
        if dd[x] > 0:
            res.append(x)
    return res
    
# get maximal coordinate for specified color vertically    
def lib_max_v_color_coord(d1, color):
    d1a = np.asarray(d1)
    res = -1
    for x in range(d1a.shape[0]):
        if color in list(d1a[x, :]):
            res = x
    return res
     
# get minimal coordinate for specified color vertically    
def lib_min_v_color_coord(d1, color):
    d1a = np.asarray(d1)
    for x in range(d1a.shape[0]):
        if color in list(d1a[x, :]):
            return x
    return -1

def lib_crop_color(a1, color):
    d1 = np.copy(np.asarray(a1))
    maxv = lib_max_v_color_coord(d1, color)
    minv = lib_min_v_color_coord(d1, color)
    maxh = lib_max_v_color_coord(d1.T, color)
    minh = lib_min_v_color_coord(d1.T, color)
    return d1[minv:maxv+1,minh:maxh+1]

def solvePatchEveryElementByPattern(d1a):
    # Solve tasks where each subelement is changed according to subelements formed bug pattern
    
    c1 = d1a[0,0]
    c2 = d1a[1,1]
    shape1 = 1
    for x in range(2,len(d1a[1,:])):
        if d1a[1,x] == c2:
            shape1 += 1
        else:
            break
    shape0 = 1
    for x in range(2,len(d1a[:,1])):
        if d1a[x,1] == c2:
            shape0 += 1
        else:
            break
    preds = np.ones((shape0, shape1))
    preds *= c2
    stepx = int(np.round(d1a.shape[0] / shape0))
    stepy = int(np.round(d1a.shape[1] / shape1))
    for x in range(shape0):
        for y in range(shape1):
            cc = d1a[int(1+(x*stepx)), int(1+(y*stepy))]
            preds[x,y] = c1 if cc==c2 else c2
    def getpatch(dd, posx, posy):
        catpos = dd[posx, posy]
        minx = posx
        miny = posy
        maxx = posx
        maxy = posy
        minxok = True
        maxxok = True
        minyok = True
        maxyok = True
        for i in range(1000):
            if minxok and posx-i>=0 and dd[posx-i, posy] == catpos:
                minx = posx-i
            else:
                minxok = False
            if minyok and posy-i>=0 and dd[posx, posy-i] == catpos:
                miny = posy-i
            else:
                minyok = False
            if maxxok and posx+i<dd.shape[0] and dd[posx+i, posy] == catpos:
                maxx = posx+i
            else:
                maxxok = False
            if maxyok and posy+i<dd.shape[1] and dd[posx, posy+i] == catpos:
                maxy = posy+i
            else:
                maxyok = False
        return minx, miny, maxx, maxy
    predsfull = np.copy(d1a)
    predsfull = np.where(d1a == c1, c2, c1)
    for x in range(shape0):
        for y in range(shape1):
            cc = d1a[int(1+(x*stepx)), int(1+(y*stepy))]
            if cc == c2:
                minx, miny, maxx, maxy = getpatch(d1a, int(1+(x*stepx)), int(1+(y*stepy)))
                predsfull[minx:1+maxx,miny:1+maxy] = preds
    return [predsfull.astype(int), predsfull.astype(int), predsfull.astype(int)] 

def lib_color_counts(df):
    cols = lib_get_colors_all(df)
    res = []
    topc = -1
    topn = -1
    for c in cols:
        r = np.where(df == c, 1, 0).sum()
        res.append(r)
        if r > topn:
            topn = r
            topc = c
    return sorted(res), topc

def lib_color_counts2(df):
    cols = lib_get_colors_all(df)
    res = []
    topc = -1
    topn = -1
    tops = []
    for c in cols:
        r = np.where(df == c, 1, 0).sum()
        res.append(r)
        if r > topn:
            topn = r
            topc = c
    tops.append(topc)
    colsleft = [x for x in cols if x != topc]
    for i in range(100):
        if len(colsleft) == 0:
            break
        topc2 = -1
        topn2 = -1
        for c in colsleft:
            r = np.where(df == c, 1, 0).sum()
            if r > topn2:
                topn2 = r
                topc2 = c
        tops.append(topc2)
        colsleft = [x for x in colsleft if x != topc2]
    return sorted(res), tops

def s4_color_counts(df):
    cols = lib_get_colors_all(df)
    res = []
    topc = -1
    topn = -1
    for c in cols:
        r = np.where(df == c, 1, 0).sum()
        res.append(r)
        if r > topn:
            topn = r
            topc = c
    return sorted(res), topc

def solveBicolorMaze(df):
    # Solve spiral mazes where walls are colored by multiple colors
    
    col_counts, topc = s4_color_counts(df)
    c1 = topc
    all_cols = lib_get_colors_all(df)
    oth_cols = [x for x in all_cols if x != c1]
    c0 = oth_cols[0]
    c2 = oth_cols[1]
    def get_border(df, c1):
        for x in range(df.shape[0]):
            for y in range(df.shape[1]):
                if x > 0 and df[x-1,y] != c1 and df[x,y] != c1 and df[x-1,y] != df[x,y]:
                    return x, y
        return -1, -1
    bx, by = get_border(df, c1)
    if bx == -1 or by == -1:
        return None
    colUpper = df[bx-1,by]
    colLower = df[bx,by]
    regUpper = lib_crop_color(df, colUpper)
    regLower = lib_crop_color(df, colLower)
    res1 = np.copy(df)
    res1[:bx,:] = colUpper
    res1[bx:,:] = colLower
    def quad(df, bx, by, sx, sy, col, side, level):
        res = np.copy(df)
        foutup = False
        foutdown = False
        for i in range(1, sx+2+level):
            bxt = bx - i
            byt = by - level - 1
            if bxt >= 0 and byt >= 0 and bxt < res.shape[0] and byt < res.shape[1]:
                if side in ['up','both']:
                    res[bxt, byt] = col
            else:
                foutup = True
            bxt = bx + (bx - bxt) - 1
            if bxt >= 0 and byt >= 0 and bxt < res.shape[0] and byt < res.shape[1]:
                if side in ['down','both']:
                    res[bxt, byt] = col
            else:
                foutdown = True
            bxt = bx - i
            byt = by + sy + level
            if bxt >= 0 and byt >= 0 and bxt < res.shape[0] and byt < res.shape[1]:
                if side in ['up','both'] and i != sx+2+level -2:
                    res[bxt, byt] = col
            else:
                foutup = True
            bxt = bx + (bx - bxt) - 1
            if bxt >= 0 and byt >= 0 and bxt < res.shape[0] and byt < res.shape[1]:
                if side in ['down','both']:
                    res[bxt, byt] = col
            else:
                foutdown = True
        for i in range(1, sy+1+level*2+2):
            bxt = bx - sx - level - 1
            byt = by - level - 1 + i
            if bxt >= 0 and byt >= 0 and bxt < res.shape[0] and byt < res.shape[1]:
                if side in ['up','both']:
                    res[bxt, byt] = col
            else:
                foutup = True
            bxt = bx + (bx - bxt) - 1
            if bxt >= 0 and byt >= 0 and bxt < res.shape[0] and byt < res.shape[1]:
                if side in ['down','both'] and i != sy+1+level*2+1:
                    res[bxt, byt] = col
            else:
                foutdown = True
        return res, foutup, foutdown
    res2 = np.copy(res1)
    mx = None
    for x in range(10):
        res1, foutup, foutdown = quad(res1, bx, by, regUpper.shape[0], regUpper.shape[1], c1, 'both', x*2)
        if foutup and mx is None:
            mx = x
    for x in range(mx):
        res2, _, _ = quad(res2, bx, by, regUpper.shape[0], regUpper.shape[1], c1, 'up', x*2)
    for x in range(10):
        res2, _, _ = quad(res2, bx, by, regUpper.shape[0], regUpper.shape[1], c1, 'down', x*2)
    res3 = np.copy(res1)
    res3[0,:] = colUpper
    return [res1, res2, res3]

def s5_findsquares(df):
    cls = lib_get_colors_all(df)
    squares = []
    for x in range(df.shape[0]-2):
        for y in range(df.shape[1]-2):
            if df[x,y] != df[x+1,y] or df[x,y] != df[x,y+1] or df[x,y] == df[x+1,y+1]:
                continue
            if x > 0 and (df[x,y] == df[x-1,y] or df[x+1,y+1] == df[x-1,y]):
                continue
            if y > 0 and (df[x,y] == df[x,y-1] or df[x+1,y+1] == df[x,y-1]):
                continue
            if x == 0 and y == 0 and df[x+2,y+2]!=df[x,y] and df[x+2,y+2]!=df[x+1,y+1]:
                continue
            squares.append((x,y))
    return squares

def s5_getcolors(df):
    squares = s5_findsquares(df)
    if len(squares) == 0:
        return -1, -1, -1
    s = squares[0]
    return df[s[0]-1,s[1]-1], df[s[0]+1,s[1]+1], df[s[0],s[1]]

def solvePutObjsInSquareByPattern(xin1):
    # Solve tasks where one object specifies pattern by which another object needs to be repeated
    
    cols = lib_get_colors_all(xin1)
    c1 = xin1[-1,-1]
    c2 = xin1[-2,-2]
    res = None
    info = None
    for x in range(1, xin1.shape[0]):
        if np.all(xin1[x,:] == c1) and np.all(xin1[x-1,:] == c1):
            info = xin1[:x,:]
            res = xin1[x:,:]
            break
    if res is None or info is None:
        return None
    c3 = -1
    shape = None
    for y in range(info.shape[1]):
        for x in range(info.shape[0]):
            if info[x,y] != c1:
                c3 = info[x,y]
                shape = lib_crop_color(info, c3)
                break
        if shape is not None:
            break
    c4 = -1
    position = None
    for y in range(info.shape[1]):
        for x in range(info.shape[0]):
            if info[x,y] != c1 and info[x,y] != c3:
                c4 = info[x,y]
                position = lib_crop_color(info, c4)
    if shape is None or position is None:
        return None
    if position.shape[0] < 2 or position.shape[1] < 2:
        minx = position.shape[0] if position.shape[0] >= 2 else 2
        miny = position.shape[1] if position.shape[1] >= 2 else 2
        position = info[1:1+minx,1+shape.shape[1]:1+shape.shape[1]+miny]
    shape = np.where(shape == c1, c2, shape)

    shaperot = np.copy(shape)
    shaperot2 = np.copy(shaperot)
    if shaperot[0,0] == c2:
        shaperot = np.rot90(shaperot, 2)
        shaperot2 = np.flipud(np.fliplr(shaperot2))
    elif shaperot[0,-1] == c2:
        shaperot = np.rot90(shaperot, 3)
        shaperot2 = np.flipud(shaperot2)
    elif shaperot[-1,0] == c2:
        shaperot = np.rot90(shaperot, 1)
        shaperot2 = np.fliplr(shaperot2)
                
    res1 = np.copy(res)
    res2 = np.copy(res)
    res3 = np.copy(res)
    
    shp1 = None
    for x in range(position.shape[0]):
        shp1t = None
        for y in range(position.shape[1]):
            det = np.copy(shape) if position[x,y] == c4 else np.ones((shape.shape[0],shape.shape[1]))*c2
            shp1t = det if shp1t is None else np.hstack((shp1t, det))
        shp1 = np.copy(shp1t) if shp1 is None else np.vstack((shp1, np.copy(shp1t)))
    res1[1:1+shp1.shape[0],1:1+shp1.shape[1]] = shp1
    return [res1, res1, res1]

def lib_color_coords(df, col):
    for x in range(df.shape[0]):
        for y in range(df.shape[1]):
            if df[x,y] == col:
                return x, y
    return -1, -1

def solveColorVerHorByColor(xin1):
    # Flow colors horizontally and vertically
    
    clrs, tops = lib_color_counts2(xin1)
    c1 = tops[0]
    c2 = tops[1]
    res1 = np.copy(xin1)
    for x in range(xin1.shape[0]):
        for y in range(xin1.shape[1]):
            if xin1[x,y] not in [c1, c2]:
                res1[x,:] = np.where(res1[x,:] != c1, xin1[x,y], res1[x,:])
                res1[:,y] = np.where(res1[:,y] != c1, xin1[x,y], res1[:,y])
    for x in range(xin1.shape[0]):
        for y in range(xin1.shape[1]):
            if xin1[x,y] not in [c1, c2]:
                res1[x,:] = np.where((res1[x,:] != c1) & (res1[x,:] != xin1[x,y]), -1, res1[x,:])
                res1[:,y] = np.where((res1[:,y] != c1) & (res1[:,y] != xin1[x,y]), -1, res1[:,y])
    res1 = np.where(res1 == -1, c2, res1)
    cols = lib_get_colors_all(xin1)
    cols = [x for x in cols if x not in [c1,c2]]
    res2 = np.copy(xin1)
    for c in cols:
        for x in range(xin1.shape[0]):
            for y in range(xin1.shape[1]):
                if xin1[x,y] == c:
                    res2[x,:] = np.where(res2[x,:] != c1, xin1[x,y], res2[x,:])
                    res2[:,y] = np.where(res2[:,y] != c1, xin1[x,y], res2[:,y])
    for c in cols:
        for x in range(xin1.shape[0]):
            for y in range(xin1.shape[1]):
                if xin1[x,y] == c:
                    res2[x,:] = np.where((res2[x,:] != c1) & (res2[x,:] != xin1[x,y]), -1, res2[x,:])
                    res2[:,y] = np.where((res2[:,y] != c1) & (res2[:,y] != xin1[x,y]), -1, res2[:,y])
                    
    res2 = np.where(res2 == -1, c1, res2)
    res3 = np.copy(xin1)
    for y in range(xin1.shape[1]):
        for x in range(xin1.shape[0]):
            if xin1[x,y] not in [c1, c2]:
                res3[x,:] = np.where(res3[x,:] != c1, xin1[x,y], res3[x,:])
                res3[:,y] = np.where(res3[:,y] != c1, xin1[x,y], res3[:,y])
    return [res1, res2, res1]

def lib_cut_info(ia, oa):
    fullcut = False
    fullcutside = -1
    fullcutdir = -1
    if ia.shape[0] == oa.shape[0]:
        if ia.shape[1] > oa.shape[1] and np.all(oa == ia[:,:oa.shape[1]]):
            fullcut = True
            fullcutside = 1
            fullcutdir = 1
        if ia.shape[1] < oa.shape[1] and np.all(ia == oa[:,:ia.shape[1]]):
            fullcut = True
            fullcutside = 2
            fullcutdir = 1

    if ia.shape[1] == oa.shape[1]:
        if ia.shape[0] > oa.shape[0] and np.all(oa == ia[:oa.shape[0],:]):
            fullcut = True
            fullcutside = 1
            fullcutdir = 2
        if ia.shape[0] < oa.shape[0] and np.all(ia == oa[:ia.shape[0],:]):
            fullcut = True
            fullcutside = 2
            fullcutdir = 2
    return fullcut,fullcutside,fullcutdir

def solveAppendMissingFigure(xin1):
    # Appends missing figure in each row
    
    def _solve(xin1):
        res = []
        c1 = xin1[0,0]
        s = 0
        brd = 1
        for x in range(1,10):
            if x < xin1.shape[0] and x < xin1.shape[1] and xin1[x,x] == c1:
                brd += 1
            else:
                break
        colect = []
        nmb = 0
        for x in range(1,xin1.shape[0]):
            if np.all(xin1[x,:] == c1) and np.where(xin1[s:x,:] == c1, 0, 1).sum() > 0:
                tmp = np.copy(xin1[s:x,:])
                s = x
                s2 = 0
                ethalon = None
                found = True
                nmb = 0
                for y in range(1, xin1.shape[1]):
                    if np.all(tmp[:,y] == c1) and np.where(tmp[:,s2:y] == c1, 0, 1).sum() > 0:
                        nmb += 1
                        tmp2 = np.copy(tmp[:,s2:y])
                        if ethalon is None:
                            ethalon = np.copy(tmp2)
                        else:
                            if not np.all(ethalon == tmp2):
                                found = False
                                break
                        s2 = y
                if not found:
                    return None
                colect.append(np.copy(ethalon[brd:,brd:]))
        one = colect[0]
        res1 = np.copy(xin1)
        for two in [one, np.rot90(one, 1), np.rot90(one, 2), np.rot90(one, 3)]:
            fnd = False
            for three in colect:
                if np.all(three == two):
                    fnd = True
                    break
            if not fnd:
                tmptmp = None
                for x in range(nmb):
                    tmp = np.hstack((np.ones((two.shape[0],brd))*c1, two))
                    tmptmp = tmp if tmptmp is None else np.hstack((tmptmp, tmp))
                tmptmp = np.hstack((tmptmp, np.ones((two.shape[0],brd))*c1))
                tmptmp = np.vstack((tmptmp, np.ones((brd,tmptmp.shape[1]))*c1))
                if tmptmp.shape[1] == xin1.shape[1]:
                    res1 = np.vstack((np.copy(res1), tmptmp))
                else:
                    break
        if res1 is not None:
            res.append(res1)
        res2 = np.copy(xin1)
        for two in [one, np.flipud(one), np.fliplr(one), np.flipud(np.fliplr(one))]:
            fnd = False
            for three in colect:
                if np.all(three == two):
                    fnd = True
                    break
            if not fnd:
                tmptmp = None
                for x in range(nmb):
                    tmp = np.hstack((np.ones((two.shape[0],brd))*c1, two))
                    tmptmp = tmp if tmptmp is None else np.hstack((tmptmp, tmp))
                tmptmp = np.hstack((tmptmp, np.ones((two.shape[0],brd))*c1))
                tmptmp = np.vstack((tmptmp, np.ones((brd,tmptmp.shape[1]))*c1))
                if tmptmp.shape[1] == xin1.shape[1]:
                    res2 = np.vstack((np.copy(res2), tmptmp))
                else:
                    break
        if res2 is not None:
            res.append(res2)
        res3 = np.copy(xin1)
        for two in [one, np.rot90(one, 3), np.rot90(one, 2), np.rot90(one, 1)]:
            fnd = False
            for three in colect:
                if np.all(three == two):
                    fnd = True
                    break
            if not fnd:
                tmptmp = None
                for x in range(nmb):
                    tmp = np.hstack((np.ones((two.shape[0],brd))*c1, two))
                    tmptmp = tmp if tmptmp is None else np.hstack((tmptmp, tmp))
                tmptmp = np.hstack((tmptmp, np.ones((two.shape[0],brd))*c1))
                tmptmp = np.vstack((tmptmp, np.ones((brd,tmptmp.shape[1]))*c1))
                if tmptmp.shape[1] == xin1.shape[1]:
                    res3 = np.vstack((np.copy(res3), tmptmp))
                else:
                    break
        if res3 is not None:
            res.append(res3)
        res = res + [np.copy(xin1), np.copy(xin1), np.copy(xin1)]
        res = res[:3]
        return res
    r = _solve(xin1)
    if r is None:
        r = _solve(np.rot90(xin1, 3))
        if r is not None:
            r = [np.rot90(np.copy(x), 1) for x in r]
    return r

def solveColoredMirror(xin1):
    # Solves tasks having mirror showing mirrored shapes, potentially changing their color
    
    topcnt, topcols = lib_color_counts2(xin1)
    def _find_col(df, col):
        for x in range(df.shape[0]):
            for y in range(df.shape[1]):
                if df[x,y] == col:
                    return x,y
        return -1, -1
    def _find_square(dfin, sxin, sy):
        sx  = sxin+1
        df = np.vstack((np.ones((1, dfin.shape[1]))*(11), dfin))
        col = df[sx,sy]
        bx = by = ex = ey = 0
        for xx in range(10):
            upd = False
            for x in range(sx-bx+1):
                if not np.any(df[sx-bx-x,sy-by:sy+ey+1] == col):
                    if x > 0:
                        bx += x
                        upd = True
                    break
            for y in range(sy-by+1):
                if not np.any(df[sx-bx:sx+ex+1,sy-by-y] == col):
                    if y > 0:
                        by += y
                        upd = True
                    break
            for x in range(df.shape[0]-sx+ex):
                if not np.any(df[sx+ex+x,sy-by:sy+ey+1] == col):
                    if x > 0:
                        ex += x
                        upd = True
                    break
            for y in range(df.shape[1]-sy+ey):
                if not np.any(df[sx-bx:sx+ex+1,sy+ey+y] == col):
                    if y > 0:
                        ey += y
                        upd = True
                    break
            if not upd:
                break
        return bx, by, ex, ey
    def _find_shapes(df, col, colbg):
        res = np.copy(df)
        arr2 = []
        for xx in range(10):
            sx, sy = _find_col(res, col) 
            if sx == -1 or sy == -1:
                break
            bx, by, ex, ey = _find_square(res, sx, sy)
            if bx == -1 or by == -1 or ex == -1 or ey == -1:
                break
            tt1 = np.copy(res[sx-bx+1:sx+ex,sy-by+1:sy+ey])
            arr2.append((tt1, (sx-bx+1,sx+ex,sy-by+1,sy+ey)))
            res[sx-bx+1:sx+ex,sy-by+1:sy+ey] = np.ones((tt1.shape[0], tt1.shape[1])) * colbg
        return arr2
    def _solve(xin1, c0, c1, c2):
        arr2 = _find_shapes(xin1, c2, c0)
        sx = -1
        for x in range(1,xin1.shape[0]):
            if np.any(xin1[x-1,:] == c1) and not np.any(xin1[x,:] == c1):
                sx = x
                break

        res = np.copy(xin1)
        for a in arr2:
            a_shape = a[0]
            a_coord = a[1]
            tmp = np.flipud(np.where(a_shape == c2, c1, c0))
            res[sx+1:sx+1+tmp.shape[0],a_coord[2]:a_coord[2]+tmp.shape[1]] = tmp
        return res
    
    def _solve2(xin1, c0, c1, c2, add=0):
        one2 = lib_crop_color(xin1, c2)
        sx = -1
        for x in range(1,xin1.shape[0]):
            if np.any(xin1[x-1,:] == c1) and not np.any(xin1[x,:] == c1):
                sx = x
                break

        sy = -1
        for y in range(xin1.shape[1]):
            if np.any(xin1[:,y] == c2):
                sy = y
                break
                
        res = np.copy(xin1)
        tmp = np.flipud(np.where(one2 == c2, c1, c0))
        res[sx+1+add:sx+1+add+tmp.shape[0],sy:sy+tmp.shape[1]] = tmp
        return res
                       
    any1 = False
    any2 = False
    found1 = False
    found2 = False
    fw = True
    for xx in range(xin1.shape[0]):
        if np.any(xin1[xx,:] == topcols[1]):
            any1 = True
        if np.any(xin1[xx,:] == topcols[2]):
            any2 = True
        if np.all(xin1[xx,1:-1] == topcols[1]):
            found1 = True
            if not any2:
                fw = False
        if np.all(xin1[xx,1:-1] == topcols[2]):
            found2 = True
            if not any1:
                fw = False

    any1 = False
    any2 = False
    found3 = False
    found4 = False
    for xx in range(xin1.shape[1]):
        if np.any(xin1[:,xx] == topcols[1]):
            any1 = True
        if np.any(xin1[:,xx] == topcols[2]):
            any2 = True
        if np.all(xin1[1:-1,xx] == topcols[1]):
            found3 = True
            if not any2:
                fw = False
        if np.all(xin1[1:-1,xx] == topcols[2]):
            found4 = True
            if not any1:
                fw = False

    c0 = topcols[0]
    c1 = topcols[1] if found1 or found3 else topcols[2]
    c2 = topcols[2] if c1 == topcols[1] else topcols[1]
    
    resin = np.copy(xin1)
    if found3 or found4:
        resin = np.rot90(resin, 3)
    if not fw:
        resin = np.flipud(resin)
            
    res1 = _solve(np.copy(resin), c0, c1, c2)
    res2 = _solve2(np.copy(resin), c0, c1, c2)
    res3 = _solve2(np.copy(resin), c0, c1, c2, 1)

    if not fw:
        res1 = np.flipud(res1)
        res2 = np.flipud(res2)
        res3 = np.flipud(res3)
    if found3 or found4:
        res1 = np.rot90(res1)
        res2 = np.rot90(res2)
        res3 = np.rot90(res3)

    return [res1, res2, res3]

def solveFixObjsSymetry(xin1):
    # Solves tasks where symetry should be restored for objects
    
    topcnt, topcols = lib_color_counts2(xin1)
    def _find_col(df, bgcol):
        for x in range(df.shape[0]):
            for y in range(df.shape[1]):
                if df[x,y] != bgcol:
                    return x,y
        return -1, -1
    def _find_square(dfin, sxin, syin, col):
        sx  = sxin+1
        sy  = syin+2
        df = np.vstack((np.ones((1, dfin.shape[1]))*col, dfin, np.ones((1, dfin.shape[1]))*col))
        df = np.hstack((np.ones((df.shape[0], 2))*col, df, np.ones((df.shape[0], 2))*col))
        bx = by = ex = ey = 0
        for xx in range(10):
            upd = False
            for x in range(sx-bx+1):
                if np.all(df[sx-bx-x,sy-by:sy+ey+1] == col):
                    if x > 0:
                        #print('bx:',bx,x)
                        bx += x
                        upd = True
                    break
            for y in range(sy-by+1):
                if np.all(df[sx-bx:sx+ex+1,sy-by-y] == col):
                    if y > 0:
                        by += y
                        upd = True
                    break
            for x in range(df.shape[0]-sx+ex):
                if np.all(df[sx+ex+x,sy-by:sy+ey+1] == col):
                    if x > 0:
                        ex += x
                        upd = True
                    break
            for y in range(df.shape[1]-sy+ey):
                if np.all(df[sx-bx:sx+ex+1,sy+ey+y] == col):
                    if y > 0:
                        ey += y
                        upd = True
                    break
            if not upd:
                break
        return bx, by, ex, ey
    def _find_shapes(df, bgcol):
        res = np.copy(df)
        arr2 = []
        for xx in range(10):
            sx, sy = _find_col(res, bgcol) 
            if sx == -1 or sy == -1:
                break
            bx, by, ex, ey = _find_square(res, sx, sy, bgcol)
            if bx == -1 or by == -1 or ex == -1 or ey == -1:
                break
            tt1 = np.copy(res[sx-bx+1:sx+ex,sy-by+1:sy+ey])
            arr2.append((tt1, (sx-bx+1,sx+ex,sy-by+1,sy+ey)))
            res[sx-bx+1:sx+ex,sy-by+1:sy+ey] = np.ones((tt1.shape[0], tt1.shape[1])) * bgcol
        return arr2
    shapes = _find_shapes(xin1, topcols[0])
    def _fixshp(df):
        for y in range(df.shape[0]):
            tmp = np.vstack((df, np.flipud(np.copy(df[:y+1]))))
            if np.all(tmp == np.flipud(np.copy(tmp))):
                return tmp
        return np.copy(df)
    res = np.copy(xin1)
    def _solve(df, col, fixud=True, fixudfirst=False):
        res = np.copy(df)
        for one in shapes:
            shp = one[0]
            newshp = np.copy(shp)
            xdirx = 1
            xdiry = 1
            if fixudfirst:
                if not np.all(newshp == np.flipud(np.copy(newshp))):
                    xdir = 1
                    for x in range(newshp.shape[0]):
                        if np.any(newshp[x,:] == col):
                            xdir = 2
                            break
                        if np.any(newshp[-x-1,:] == col):
                            xdir = 1
                            break
                    if xdir == 1:
                        newshp = _fixshp(np.copy(newshp))
                    else:
                        newshp = np.flipud(_fixshp(np.flipud(np.copy(newshp))))
                    xdirx = xdir
            if not np.all(newshp == np.fliplr(np.copy(newshp))):
                xdir = 1
                for y in range(newshp.shape[1]):
                    if np.any(newshp[:,y] == col):
                        xdir = 2
                        break
                    if np.any(newshp[:,-y-1] == col):
                        xdir = 1
                        break
                if xdir == 1:
                    newshp = np.rot90(_fixshp(np.rot90(np.copy(newshp), 3)))
                else:
                    newshp = np.rot90(_fixshp(np.rot90(np.copy(newshp))), 3)
                xdiry = xdir
            if fixud:
                if not np.all(newshp == np.flipud(np.copy(newshp))):
                    xdir = 1
                    for x in range(newshp.shape[0]):
                        if np.any(newshp[x,:] == col):
                            xdir = 2
                            break
                        if np.any(newshp[-x-1,:] == col):
                            xdir = 1
                            break
                    if xdir == 1:
                        newshp = _fixshp(np.copy(newshp))
                    else:
                        newshp = np.flipud(_fixshp(np.flipud(np.copy(newshp))))
                    xdirx = xdir
            if shp.shape != newshp.shape:
                try:
                    if xdirx == 1 and xdiry == 1:
                        res[one[1][0]:one[1][0]+newshp.shape[0],one[1][2]:one[1][2]+newshp.shape[1]] = np.copy(newshp)
                    if xdirx == 1 and xdiry == 2:
                        res[one[1][0]:one[1][0]+newshp.shape[0],one[1][3]-newshp.shape[1]:one[1][3]] = np.copy(newshp)
                    if xdirx == 2 and xdiry == 1:
                        res[one[1][1]-newshp.shape[0]:one[1][1],one[1][2]:one[1][2]+newshp.shape[1]] = np.copy(newshp)
                    if xdirx == 2 and xdiry == 2:
                        res[one[1][1]-newshp.shape[0]:one[1][1],one[1][3]-newshp.shape[1]:one[1][3]] = np.copy(newshp)
                except:
                    pass
        return res
    res1 = _solve(xin1, topcols[2])
    res2 = _solve(xin1, topcols[2], False)
    res3 = _solve(xin1, topcols[2], False, True)
    return [res1, res2, res3]

def solveShooting(xin1):
    # Solves tasks where "shooting" lines should be drawn (horizontally or vertically)
    # Starting point determines color and only background color can be drawn on
    
    topcnt, topcols = lib_color_counts2(xin1)
    c1 = topcols[0]

    def _rot(xin1, etha):
        for x in range(xin1.shape[0]-etha.shape[0]+1):
            for y in range(xin1.shape[1]-etha.shape[1]+1):
                for d in range(4):
                    ethac = np.copy(etha)
                    if d>0:
                        ethac = np.rot90(ethac, d)
                    if np.all(ethac == xin1[x:x+ethac.shape[0],y:y+ethac.shape[1]]):
                        return x+1, y+1, d
        return -1, -1, -1

    def _fill(df, sx, sy, sd, col, bg, algo=0):
        if sd == 0:
            for x in range(1,100):
                nx = sx + x*2
                ny = sy
                if nx >= df.shape[0] or nx < 0 or ny >= df.shape[1] or ny < 0:
                    break
                if df[nx,ny] == bg or algo==1:
                    df[nx,ny] = col
                elif df[nx,ny] != bg and algo == 3:
                    df[nx,ny] = bg
        if sd == 1:
            for y in range(1,100):
                nx = sx 
                ny = sy + y*2
                if nx >= df.shape[0] or nx < 0 or ny >= df.shape[1] or ny < 0:
                    break
                if df[nx,ny] == bg or algo==1:
                    df[nx,ny] = col
                elif df[nx,ny] != bg and algo == 3:
                    df[nx,ny] = bg
        if sd == 2:
            for x in range(1,100):
                nx = sx - x*2
                ny = sy
                if nx >= df.shape[0] or nx < 0 or ny >= df.shape[1] or ny < 0:
                    break
                if df[nx,ny] == bg or algo==1:
                    df[nx,ny] = col
                elif df[nx,ny] != bg and algo == 3:
                    df[nx,ny] = bg
        if sd == 3:
            for y in range(1,100):
                nx = sx
                ny = sy - y*2
                if nx >= df.shape[0] or nx < 0 or ny >= df.shape[1] or ny < 0:
                    break
                if df[nx,ny] == bg or algo==1:
                    df[nx,ny] = col
                elif df[nx,ny] != bg and algo == 3:
                    df[nx,ny] = bg
        return df
    
    combos = []
    if len(topcols) == 3:
        combos.append((topcols[1], topcols[2]))
    if len(topcols) == 4:
        combos.append((topcols[1], topcols[2]))
        combos.append((topcols[1], topcols[3]))
    if len(topcols) == 5:
        combos.append((topcols[1], topcols[2]))
        combos.append((topcols[1], topcols[3]))
        combos.append((topcols[1], topcols[4]))
        combos.append((topcols[2], topcols[1]))
        combos.append((topcols[2], topcols[3]))
        combos.append((topcols[2], topcols[4]))
        
    res3 = np.copy(xin1)
    res2 = np.copy(xin1)
    res1 = np.copy(xin1)
    for combo in combos:
        c2 = combo[0]
        c3 = combo[1]
        etha = np.asarray([[c2,c2,c2],[c2,c3,c2],[c1,c1,c1]])
    
        sx, sy, sd = _rot(xin1, etha)
        if sx != -1:
            res1 = _fill(res1, sx, sy, sd, c3, c1)
            res2 = _fill(res2, sx, sy, sd, c3, c1, 1)
            res3 = _fill(res3, sx, sy, sd, c3, c1, 3)
    
    return [res1, res2, res3]

def solveDrawBordersAroundShapes(xin1):
    # Solves tasks where borders must be drawn around shapes
    
    topcnt, topcols = lib_color_counts2(xin1)
    c1 = topcols[0]
    combos = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    res1 = np.copy(xin1)
    res2 = np.copy(xin1)
    res3 = np.copy(xin1)
    for x in range(xin1.shape[0]):
        for y in range(xin1.shape[1]):
            for c in combos:
                if xin1[x,y] != topcols[0]:
                    continue
                xx = x + c[0]
                yy = y + c[1]
                if xx < 0 or xx >= xin1.shape[0] or yy < 0 or yy >= xin1.shape[1]:
                    continue
                if res2[x,y] == topcols[0]:
                    if xin1[xx,yy] == topcols[1]:
                        res2[x,y] = topcols[2]
                    if xin1[xx,yy] == topcols[2]:
                        res2[x,y] = topcols[1]
                else:
                    if xin1[xx,yy] == topcols[1] and res2[x,y] == topcols[1]:
                        res2[x,y] = -1
                    if xin1[xx,yy] == topcols[2] and res2[x,y] == topcols[2]:
                        res2[x,y] = -1
                    
                if xin1[xx,yy] == topcols[1]:
                    res1[x,y] = topcols[2]
                if xin1[xx,yy] == topcols[2]:
                    res1[x,y] = topcols[1]
                    
                if res3[x,y] == topcols[0]:
                    if xin1[xx,yy] == topcols[1]:
                        res3[x,y] = topcols[2]
                    if xin1[xx,yy] == topcols[2]:
                        res3[x,y] = topcols[1]

    res2 = np.where(res2 == -1, topcols[0], res2)
    return [res1, res2, res3]

def lib_find_subshape(df, sub, bg=None):
    def _all(one, two, bg):
        for x in range(one.shape[0]):
            for y in range(one.shape[1]):
                if two[x,y] == bg:
                    continue
                if one[x,y] != two[x,y]:
                    return False
        return True
    for x in range(df.shape[0]-sub.shape[0]+1):
        for y in range(df.shape[1]-sub.shape[1]+1):
            cur = df[x:x+sub.shape[0],y:y+sub.shape[1]]
            if np.all(cur == sub):
                return True
    return False

def solveDrawHorVerLinesNewC1(xin1, newc):
    def _solve(df, newc, sol1=True):
        topcnt, topcols = lib_color_counts2(df)
        c1 = topcols[0]
        res = np.copy(df)
        for x in range(df.shape[0]):
            if np.all(df[x,:] == c1):
                continue
            wasfirst = -1
            for y in range(df.shape[1]):
                if df[x,y] == c1:
                    continue
                if wasfirst == -1:
                    res[x,:y+1] = df[x,y]
                    wasfirst = y
                else:
                    res[x,y:] = df[x,y]
                    res[x,wasfirst+1:y] = newc
                    wasfirst = -2
            if wasfirst != -2:
                res[x,wasfirst+1:] = newc
                if not sol1:
                    res[x,wasfirst:] = res[x,0]
                    res[x,:wasfirst] = newc
        return res
    res1 = _solve(xin1, newc)
    res2 = np.rot90(_solve(np.rot90(np.copy(xin1),3), newc))
    res3 = _solve(xin1, newc, False)
    return [res1, res2, res3]

def solveDrawHorVerLinesNewC2(xin1, newc):
    def _fill(row, col, bg, mode, inrow):
        res = []
        for x in range(row.shape[0]):
            if mode == 1:
                if row[x] == bg or row[x] == col:
                    res.append(col)
                else:
                    res.append(bg)
            if mode == 2:
                if row[x] == bg:
                    res.append(col)
                else:
                    res.append(bg)
            if mode == 3:
                if row[x] == bg:
                    res.append(col)
                else:
                    res.append(newc)
        return np.asarray(res)
    def _solvex(df, newc, res, sol1=True, mode=1):
        topcnt, topcols = lib_color_counts2(df)
        c1 = topcols[0]
        for x in range(df.shape[0]):
            if np.all(df[x,:] == c1):
                continue
            if np.where(df[x,:] == c1, 0, 1).sum() < 2:
                continue
            wasfirst = -1
            for y in range(df.shape[1]):
                if df[x,y] == c1:
                    continue
                if wasfirst == -1:
                    res[x,:y+1] = _fill(res[x,:y+1], df[x,y], c1, mode, df[x,:y+1])
                    wasfirst = y
                else:
                    res[x,y:] = _fill(res[x,y:], df[x,y], c1, mode, df[x,y:])
                    res[x,wasfirst+1:y] = _fill(res[x,wasfirst+1:y], newc, c1, mode, df[x,wasfirst+1:y])
                    wasfirst = -2
        return res
    def _solve(df, newc, mode):
        topcnt, topcols = lib_color_counts2(df)
        c1 = topcols[0]
        res = np.ones((df.shape[0],df.shape[1])) * c1
        res = _solvex(df, newc, res, mode=mode)
        res = np.rot90(_solvex(np.rot90(np.copy(df)), newc, np.rot90(res), mode=mode), 3)
        return res
    res1 = _solve(xin1, newc, 1)
    res2 = _solve(xin1, newc, 2)
    res3 = _solve(xin1, newc, 3)
    return [res1, res2, res3]

def solveDrawHorVerLinesNewC3(xin1, newc):
    def _fill(row, col, bg, mode, inrow):
        res = []
        for x in range(row.shape[0]):
            if mode == 1:
                if row[x] == bg or row[x] == col:
                    res.append(col)
                else:
                    if col == newc:
                        res.append(row[x])
                    else:
                        res.append(bg)
            if mode == 2:
                if row[x] == bg:
                    res.append(col)
                else:
                    if col == newc:
                        res.append(row[x])
                    else:
                        res.append(newc)
            if mode == 3:
                if row[x] == bg:
                    res.append(col)
                else:
                    if col == newc:
                        res.append(row[x])
                    else:
                        res.append(bg)
        return np.asarray(res)
    def _solvex(df, newc, res, sol1=True, mode=1):
        topcnt, topcols = lib_color_counts2(df)
        c1 = topcols[0]
        for x in range(df.shape[0]):
            if np.all(df[x,:] == c1):
                continue
            if np.where(df[x,:] == c1, 0, 1).sum() < 2:
                continue
            wasfirst = -1
            for y in range(df.shape[1]):
                if df[x,y] == c1:
                    continue
                if wasfirst == -1:
                    res[x,:y+1] = _fill(res[x,:y+1], df[x,y], c1, mode, df[x,:y+1])
                    wasfirst = y
                else:
                    res[x,y:] = _fill(res[x,y:], df[x,y], c1, mode, df[x,y:])
                    res[x,wasfirst+1:y] = _fill(res[x,wasfirst+1:y], newc, c1, mode, df[x,wasfirst+1:y])
                    wasfirst = -2
        return res
    def _solve(df, newc, mode):
        topcnt, topcols = lib_color_counts2(df)
        c1 = topcols[0]
        res = np.ones((df.shape[0],df.shape[1])) * c1
        res = _solvex(df, newc, res, mode=mode)
        res = np.rot90(_solvex(np.rot90(np.copy(df)), newc, np.rot90(res), mode=mode), 3)
        return res
    res1 = _solve(xin1, newc, 1)
    res2 = _solve(xin1, newc, 2)
    res3 = _solve(xin1, newc, 3)
    return [res1, res2, res3]

def solveDrawHorVerLinesNewC4(xin1, newc):
    def _fill(row, col, bg, mode, inrow):
        res = []
        for x in range(row.shape[0]):
            if mode == 1:
                res.append(col)
            if mode == 2:
                if row[x] == bg:
                    res.append(col)
                else:
                    res.append(row[x])
            if mode == 3:
                if row[x] == bg or row[x] == newc:
                    res.append(col)
                else:
                    res.append(row[x])
        return np.asarray(res)
    def _solvex(df, newc, res, sol1=True, mode=1):
        topcnt, topcols = lib_color_counts2(df)
        c1 = topcols[0]
        for x in range(df.shape[0]):
            if np.all(df[x,:] == c1):
                continue
            if np.where(df[x,:] == c1, 0, 1).sum() < 2:
                continue
            wasfirst = -1
            for y in range(df.shape[1]):
                if df[x,y] == c1:
                    continue
                if wasfirst == -1:
                    res[x,:y+1] = _fill(res[x,:y+1], df[x,y], c1, mode, df[x,:y+1])
                    wasfirst = y
                else:
                    res[x,y:] = _fill(res[x,y:], df[x,y], c1, mode, df[x,y:])
                    res[x,wasfirst+1:y] = _fill(res[x,wasfirst+1:y], newc, c1, mode, df[x,wasfirst+1:y])
                    wasfirst = -2
        return res
    def _solve(df, newc, mode):
        topcnt, topcols = lib_color_counts2(df)
        c1 = topcols[0]
        res = np.ones((df.shape[0],df.shape[1])) * c1
        res = _solvex(df, newc, res, mode=mode)
        res = np.rot90(_solvex(np.rot90(np.copy(df)), newc, np.rot90(res), mode=mode), 3)
        return res
    res1 = _solve(xin1, newc, 1)
    res2 = _solve(xin1, newc, 2)
    res3 = _solve(xin1, newc, 3)
    return [res1, res2, res3]

def solveDrawHorVerLinesNewC5(xin1, newc):
    def _fill(row, col, bg, mode, inrow, frc=False):
        res = []
        for x in range(row.shape[0]):
            if mode == 1:
                if row[x] == bg or not frc:
                    res.append(col)
                else:
                    res.append(row[x])
            if mode == 2:
                if row[x] == bg or frc:
                    res.append(col)
                else:
                    res.append(row[x])
            if mode == 3:
                if row[x] == bg or not frc:
                    res.append(col)
                else:
                    res.append(row[x])
        return np.asarray(res)
    def _solvex(df, newc, res, sol1=True, mode=1):
        topcnt, topcols = lib_color_counts2(df)
        c1 = topcols[0]
        for x in range(df.shape[0]):
            if np.all(df[x,:] == c1):
                continue
            if np.where(df[x,:] == c1, 0, 1).sum() < 2:
                continue
            wasfirst = -1
            for y in range(df.shape[1]):
                if df[x,y] == c1:
                    continue
                if wasfirst == -1:
                    res[x,:y+1] = _fill(res[x,:y+1], df[x,y], c1, mode, df[x,:y+1])
                    wasfirst = y
                else:
                    res[x,y:] = _fill(res[x,y:], df[x,y], c1, mode, df[x,y:], frc=True)
                    res[x,wasfirst+1:y] = _fill(res[x,wasfirst+1:y], newc, c1, mode, df[x,wasfirst+1:y], frc=True if mode==3 else False)
                    wasfirst = -2
        return res
    def _solve(df, newc, mode):
        topcnt, topcols = lib_color_counts2(df)
        c1 = topcols[0]
        res = np.ones((df.shape[0],df.shape[1])) * c1
        res = _solvex(df, newc, res, mode=mode)
        res = np.rot90(_solvex(np.rot90(np.copy(df)), newc, np.rot90(res), mode=mode), 3)
        return res
    res1 = _solve(xin1, newc, 1)
    res2 = _solve(xin1, newc, 2)
    res3 = _solve(xin1, newc, 3)
    return [res1, res2, res3]

def solveDrawHorVerLinesNewC6(xin1, newc):
    def _fill(row, col, bg, mode, inrow):
        res = []
        for x in range(row.shape[0]):
            if mode == 1:
                if row[x] == bg:
                    res.append(col)
                else:
                    if col == newc:
                        res.append(row[x])
                    elif col != newc and row[x] == newc:
                        res.append(col)
                    else:
                        res.append(newc)
            if mode == 2:
                if row[x] == bg:
                    res.append(col)
                else:
                    res.append(newc)
            if mode == 3:
                if row[x] == bg:
                    res.append(col)
                else:
                    if col == newc:
                        res.append(row[x])
                    else:
                        res.append(newc)
        return np.asarray(res)
    def _solvex(df, newc, res, sol1=True, mode=1):
        topcnt, topcols = lib_color_counts2(df)
        c1 = topcols[0]
        for x in range(df.shape[0]):
            if np.all(df[x,:] == c1):
                continue
            if np.where(df[x,:] == c1, 0, 1).sum() < 2:
                continue
            wasfirst = -1
            for y in range(df.shape[1]):
                if df[x,y] == c1:
                    continue
                if wasfirst == -1:
                    res[x,:y+1] = _fill(res[x,:y+1], df[x,y], c1, mode, df[x,:y+1])
                    wasfirst = y
                else:
                    res[x,y:] = _fill(res[x,y:], df[x,y], c1, mode, df[x,y:])
                    res[x,wasfirst+1:y] = _fill(res[x,wasfirst+1:y], newc, c1, mode, df[x,wasfirst+1:y])
                    wasfirst = -2
        return res
    def _solve(df, newc, mode):
        topcnt, topcols = lib_color_counts2(df)
        c1 = topcols[0]
        res = np.ones((df.shape[0],df.shape[1])) * c1
        res = _solvex(df, newc, res, mode=mode)
        res = np.rot90(_solvex(np.rot90(np.copy(df)), newc, np.rot90(res), mode=mode), 3)
        return res
    res1 = _solve(xin1, newc, 1)
    res2 = _solve(xin1, newc, 2)
    res3 = _solve(xin1, newc, 3)
    return [res1, res2, res3]

def solveDrawHorVerLinesNewC8(xin1, newc):
    def _fill(row, col, bg, mode, inrow):
        res = []
        for x in range(row.shape[0]):
            if mode == 1:
                if row[x] == bg:
                    res.append(col)
                else:
                    if col == newc:
                        res.append(row[x])
                    elif col != newc and row[x] == newc:
                        res.append(col)
                    else:
                        res.append(newc)
            if mode == 2:
                if row[x] == bg:
                    res.append(col)
                else:
                    res.append(newc)
            if mode == 3:
                if row[x] == bg:
                    res.append(col)
                else:
                    if col == newc:
                        res.append(row[x])
                    else:
                        res.append(newc)
        return np.asarray(res)
    def _solvex(df, newc, res, sol1=True, mode=1):
        topcnt, topcols = lib_color_counts2(df)
        c1 = topcols[0]
        for x in range(df.shape[0]):
            if np.all(df[x,:] == c1):
                continue
            if np.where(df[x,:] == c1, 0, 1).sum() < 2:
                continue
            wasfirst = -1
            for y in range(df.shape[1]):
                if df[x,y] == c1:
                    continue
                if wasfirst == -1:
                    res[x,:y+1] = _fill(res[x,:y+1], df[x,y], c1, mode, df[x,:y+1])
                    wasfirst = y
                else:
                    res[x,y:] = _fill(res[x,y:], df[x,y], c1, mode, df[x,y:])
                    if mode in [2,3] and df[x,wasfirst] == df[x,y]:
                        res[x,wasfirst+1:y] = _fill(res[x,wasfirst+1:y], df[x,y], c1, mode, df[x,wasfirst+1:y])
                    else:
                        res[x,wasfirst+1:y] = _fill(res[x,wasfirst+1:y], newc, c1, mode, df[x,wasfirst+1:y])
                    
                    wasfirst = -2
        return res
    def _solve(df, newc, mode):
        topcnt, topcols = lib_color_counts2(df)
        c1 = topcols[0]
        res = np.ones((df.shape[0],df.shape[1])) * c1
        res = _solvex(df, newc, res, mode=mode)
        res = np.rot90(_solvex(np.rot90(np.copy(df)), newc, np.rot90(res), mode=mode), 3)
        return res
    res1 = _solve(xin1, newc, 1)
    res2 = _solve(xin1, newc, 2)
    res3 = _solve(xin1, newc, 3)
    return [res1, res2, res3]

def solveDrawHorVerLinesNewC9(xin1, newc):
    def _fill(row, col, bg, mode, inrow):
        res = []
        for x in range(row.shape[0]):
            if mode == 3:
                if row[x] == bg:
                    res.append(col)
                else:
                    if col == newc:
                        res.append(row[x])
                    else:
                        res.append(newc)
            if mode == 2:
                if row[x] == bg:
                    res.append(col)
                else:
                    if inrow[x] != bg:
                        res.append(inrow[x])
                    else:
                        res.append(newc)
            if mode == 1:
                if row[x] == bg:
                    res.append(col)
                else:
                    if inrow[x] != bg:
                        res.append(inrow[x])
                    elif col == newc:
                        res.append(row[x])
                    else:
                        res.append(newc)
        return np.asarray(res)
    def _solvex(df, newc, res, sol1=True, mode=1):
        topcnt, topcols = lib_color_counts2(df)
        c1 = topcols[0]
        for x in range(df.shape[0]):
            if np.all(df[x,:] == c1):
                continue
            if np.where(df[x,:] == c1, 0, 1).sum() < 2:
                continue
            wasfirst = -1
            for y in range(df.shape[1]):
                if df[x,y] == c1:
                    continue
                if wasfirst == -1:
                    res[x,:y+1] = _fill(res[x,:y+1], df[x,y], c1, mode, df[x,:y+1])
                    wasfirst = y
                else:
                    res[x,y:] = _fill(res[x,y:], df[x,y], c1, mode, df[x,y:])
                    if df[x,wasfirst] == df[x,y]:
                        res[x,wasfirst+1:y] = _fill(res[x,wasfirst+1:y], df[x,y], c1, mode, df[x,wasfirst+1:y])
                    else:
                        res[x,wasfirst+1:y] = _fill(res[x,wasfirst+1:y], newc, c1, mode, df[x,wasfirst+1:y])
                    
                    wasfirst = -2
        return res
    def _solve(df, newc, mode):
        topcnt, topcols = lib_color_counts2(df)
        c1 = topcols[0]
        res = np.ones((df.shape[0],df.shape[1])) * c1
        res = _solvex(df, newc, res, mode=mode)
        res = np.rot90(_solvex(np.rot90(np.copy(df)), newc, np.rot90(res), mode=mode), 3)
        return res
    res1 = _solve(xin1, newc, 1)
    res2 = _solve(xin1, newc, 2)
    res3 = _solve(xin1, newc, 3)
    return [res1, res2, res3]

def solveDrawHorVerLines(xin1, task):
    # Solves tasks where horizontal and vertical lines must be drawn at positions defined by points

    # Check if some new color appears in outputs
    unseen_color = None
    task_train = task['train']
    for item in task_train:
        i, o = item['input'], item['output']
        ia, oa = np.asarray(i), np.asarray(o)
        icc = lib_get_colors_all(ia)
        occ = lib_get_colors_all(oa)
        if len(occ) > len(icc):
            notsame = [x for x in occ if x not in icc]
            unseen_color = notsame[0]
            return solveDrawHorVerLinesNewC9(xin1, unseen_color)
    return solveDrawHorVerLinesNewC9(xin1, 0)

def solveColorFlow(xin, mode=1):
    # Solves tasks where specific color flows in a given direction until meets another color
    
    def _solve(xin):
        cols = lib_color_counts_dict(xin)
        res = np.copy(xin)
        for i in range(xin.shape[0]):
            if xin[i,0] == cols[1][0] and xin[i,1] == cols[1][0]:
                for j in range(2, xin.shape[1]):
                    if xin[i,j] == cols[0][0]:
                        res[i,j] = cols[1][0]
                    if xin[i,j] == cols[2][0]:
                        break
        return res
    
    def _solve2(xin):
        cols = lib_color_counts_dict(xin)
        res = np.copy(xin)
        for i in range(xin.shape[0]):
            if xin[i,0] == cols[2][0] and xin[i,1] == cols[2][0]:
                for j in range(2, xin.shape[1]):
                    if xin[i,j] == cols[0][0]:
                        res[i,j] = cols[2][0]
                    if xin[i,j] == cols[1][0]:
                        break
        return res

    xxres = np.copy(xin)
    for i in range(4):
        xxin = np.rot90(np.copy(xin), i)
        xres = _solve2(np.copy(xxin))
        xres = np.rot90(xres, 4-i)
        xxres = np.where(xxres == xin, xres, xxres)
    return [xxres, xxres, xxres]

def lib_crop_bg(xin, bg):
    res = np.copy(xin)
    if np.all(res == bg):
        return None
    for i in range(4):
        for j in range(res.shape[0]):
            if not np.all(res[j,:] == bg):
                res = res[j:,:]
                break
        res = np.rot90(res)
    return res

def lib_objs_split_by_bg(xin, bg):
    res = lib_crop_bg(xin, bg)
    if res is None:
        return []
    objs = []
    for i in range(1,res.shape[0]-1):
        if np.all(res[i,:] == bg):
            obj1 = lib_crop_bg(res[:i+1,:], bg)
            obj2 = lib_crop_bg(res[i+1:,:], bg)
            if obj1 is not None:
                objs.append(obj1)
            if obj2 is not None:
                objs.append(obj2)
            break
    if objs:
        return objs
    for i in range(1,res.shape[1]-1):
        if np.all(res[:,i] == bg):
            obj1 = lib_crop_bg(res[:,:i+1], bg)
            obj2 = lib_crop_bg(res[:,i+1:], bg)
            if obj1 is not None:
                objs.append(obj1)
            if obj2 is not None:
                objs.append(obj2)
            break
    return objs

def lib_objs_split_by_bg2(xin, bg):
    res = lib_crop_bg(xin, bg)
    if res is None:
        return []
    objs = lib_objs_split_by_bg(xin, bg)
    if objs:
        return objs
    
    mask = np.zeros(res.shape)
    for z in range(0, res.shape[0]):
        if res[z,0] != bg:
            mask[z,0] = 1
            break
    oldmask = np.zeros(mask.shape)
    for iterx in range(20):
        if np.all(mask == oldmask): #no changes in mask
            obj = np.ones(mask.shape) * bg
            obj = np.where(mask == 1, res, obj)
            obj = lib_crop_bg(obj, bg)
            objs.append(np.copy(obj))

            res = np.where(mask == 1, bg, res)
            res = lib_crop_bg(res, bg)
            if res is None:
                return objs
            mask = np.zeros(res.shape)
            for z in range(0, res.shape[0]):
                if res[z,0] != bg:
                    mask[z,0] = 1
                    break
        oldmask = np.copy(mask)
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                if mask[i,j] == 0:
                    continue
                if i < res.shape[0]-1 and mask[i+1,j] == 0 and res[i+1,j] != bg:
                    mask[i+1,j] = 1
                if j < res.shape[1]-1 and mask[i,j+1] == 0 and res[i,j+1] != bg:
                    mask[i,j+1] = 1
                if i > 0 and mask[i-1,j] == 0 and res[i-1,j] != bg:
                    mask[i-1,j] = 1
                if j > 0 and mask[i,j-1] == 0 and res[i,j-1] != bg:
                    mask[i,j-1] = 1
                if i < res.shape[0]-1 and j < res.shape[1]-1 and mask[i+1,j+1] == 0 and res[i+1,j+1] != bg:
                    mask[i+1,j+1] = 1
                if i > 0 and j < res.shape[1]-1 and mask[i-1,j+1] == 0 and res[i-1,j+1] != bg:
                    mask[i-1,j+1] = 1
                if i > 0 and j > 0 and mask[i-1,j-1] == 0 and res[i-1,j-1] != bg:
                    mask[i-1,j-1] = 1
                if j > 0 and i < res.shape[0]-1 and mask[i+1,j-1] == 0 and res[i+1,j-1] != bg:
                    mask[i+1,j-1] = 1
                    
def solveSimetrical(xin):
    # Solve tasks where symetrical pattern must be composed from 2 given objects, ensuring symetry horizontally and vertically

    def _mk(hor, ver, bg):
        horskip = 0
        verskip = 0
        for i in range(int(ver.shape[1]//2)):
            if ver[-1,i] == bg:
                verskip += 1
        for i in range(int(hor.shape[0]//2)):
            if hor[i,-1] == bg:
                horskip += 1
        res = np.ones((ver.shape[0]*2 + hor.shape[0] - 2*horskip, hor.shape[1]*2 + ver.shape[1] - 2*verskip))*bg
        res[0:ver.shape[0],hor.shape[1]-verskip:hor.shape[1]-verskip+ver.shape[1]] = np.copy(ver)
        res[ver.shape[0]+hor.shape[0]-horskip*2:ver.shape[0]+hor.shape[0]-horskip*2+ver.shape[0],hor.shape[1]-verskip:hor.shape[1]-verskip+ver.shape[1]] = np.flipud(np.copy(ver))
        res[ver.shape[0]-horskip:ver.shape[0]-horskip+hor.shape[0],0:hor.shape[1]] = np.copy(hor)
        res[ver.shape[0]-horskip:ver.shape[0]-horskip+hor.shape[0],hor.shape[1]+ver.shape[1]-verskip*2:hor.shape[1]+ver.shape[1]-verskip*2+hor.shape[1]] = np.fliplr(np.copy(hor))
        return res
    cols = lib_color_counts_dict(xin)
    bg = cols[0][0]
    objs = lib_objs_split_by_bg2(xin, bg)
    ver = objs[0] if np.all(objs[0] == np.fliplr(objs[0])) else objs[1]
    hor = objs[0] if np.all(objs[0] == np.flipud(objs[0])) else objs[1]
    over = np.copy(ver)
    ohor = np.copy(hor)
    if np.where(ver[0,:] == bg, 1, 0).sum() < np.where(ver[-1,:] == bg, 1, 0).sum():
        ver = np.flipud(ver)
    if np.where(hor[:,0] == bg, 1, 0).sum() < np.where(hor[:,-1] == bg, 1, 0).sum():
        hor = np.fliplr(hor)
    res1 = _mk(hor, ver, bg)
    res2 = _mk(ohor, over, bg)
    return [res1, res2, res1]

def skip_bg(xin, bg):
    try:
        mask = np.ones(xin.shape)
        for i in range(xin.shape[0]):
            if np.all(xin[i,:] == bg):
                mask[i,:] = 0
        for i in range(xin.shape[1]):
            if np.all(xin[:,i] == bg):
                mask[:,i] = 0
        res = xin[mask == 1]
        shape0 = int(mask.sum(axis=1).max())
        res = res.reshape((shape0, -1))
        return res
    except Exception as ex:
        print(ex)
    return None

def is_square(apositiveint):
    x = apositiveint // 2
    seen = set([x])
    while x * x != apositiveint:
        x = (x + (apositiveint // x)) // 2
        if x in seen: return False
        seen.add(x)
    return True

def solvePatternOfPatterns(xin, bg1, bg2=None):
    if not is_square(xin.shape[0]) or not is_square(xin.shape[1]):
        return None
    s0 = int(np.sqrt(xin.shape[0]))
    s1 = int(np.sqrt(xin.shape[1]))
    res = np.ones((s0,s1)) * bg1
    for i in range(s0):
        for j in range(s1):
            one = xin[i*s0:(i+1)*s0,j*s1:(j+1)*s1]
            if not np.all(one == one[0,0]):
                res[i,j] = one[i,j]
    return res
    
def solveRepeatingPattern(xin, xout=None, info={}):
    # Solve tasks where global pattern should be extracted, consisting of objects each having pattern themselves

    # Check if skipping background helps
    cols = lib_color_counts_dict(xin)
    bg = cols[0][0]
    r1 = skip_bg(xin, bg)
    cols2 = lib_color_counts_dict(r1)
    bg2 = cols2[0][0]
    res = solvePatternOfPatterns(r1, bg, bg2)
    return [res, res, res]

def cleanLinearBackground(xin):
    res = np.copy(xin)
    fgc = []
    direction = None
    for i in range(xin.shape[1]-1):
        cols1 = lib_get_colors_all(res[:,[i]])
        cols2 = lib_get_colors_all(res[:,[i+1]])
        if len(cols2) > len(cols1) and len([x for x in cols1 if x in cols2]) == len(cols1):
            fgc.extend([x for x in cols2 if x not in cols1 and x not in fgc])
            res[:,i+1] = res[:,i]
            if len(cols1) == 1:
                direction = 'V'
    if fgc and direction is None:
        direction = 'H'
    for i in range(xin.shape[0]-1):
        cols1 = lib_get_colors_all(res[[i],:])
        cols2 = lib_get_colors_all(res[[i+1],:])
        if len(cols2) > len(cols1) and len([x for x in cols1 if x in cols2]) == len(cols1):
            fgc.extend([x for x in cols2 if x not in cols1 and x not in fgc])
            res[i+1,:] = res[i,:]
    return res, fgc, direction

def lib_objs_split_by_bg3(xin, bg):
    res = lib_crop_bg(xin, bg)
    if res is None:
        return []
    objs = []
    mask = np.zeros(res.shape)
    for z in range(0, res.shape[0]):
        if res[z,0] != bg:
            mask[z,0] = 1
            break
    oldmask = np.zeros(mask.shape)
    for iter in range(100):
        if np.all(mask == oldmask): #no changes in mask
            obj = np.ones(mask.shape) * bg
            obj = np.where(mask == 1, res, obj)
            obj = lib_crop_bg(obj, bg)
            objs.append(np.copy(obj))

            res = np.where(mask == 1, bg, res)
            res = lib_crop_bg(res, bg)
            if res is None:
                return objs
            mask = np.zeros(res.shape)
            for z in range(0, res.shape[0]):
                if res[z,0] != bg:
                    mask[z,0] = 1
                    break
        oldmask = np.copy(mask)
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                if mask[i,j] == 0:
                    continue
                if i < res.shape[0]-1 and mask[i+1,j] == 0 and res[i+1,j] != bg:
                    mask[i+1,j] = 1
                if j < res.shape[1]-1 and mask[i,j+1] == 0 and res[i,j+1] != bg:
                    mask[i,j+1] = 1
                if i > 0 and mask[i-1,j] == 0 and res[i-1,j] != bg:
                    mask[i-1,j] = 1
                if j > 0 and mask[i,j-1] == 0 and res[i,j-1] != bg:
                    mask[i,j-1] = 1
                if i < res.shape[0]-1 and j < res.shape[1]-1 and mask[i+1,j+1] == 0 and res[i+1,j+1] != bg:
                    mask[i+1,j+1] = 1
                if i > 0 and j < res.shape[1]-1 and mask[i-1,j+1] == 0 and res[i-1,j+1] != bg:
                    mask[i-1,j+1] = 1
                if i > 0 and j > 0 and mask[i-1,j-1] == 0 and res[i-1,j-1] != bg:
                    mask[i-1,j-1] = 1
                if j > 0 and i < res.shape[0]-1 and mask[i+1,j-1] == 0 and res[i+1,j-1] != bg:
                    mask[i+1,j-1] = 1

def lib_obj_in_grid(xin, obj):
    if obj.shape[0] > xin.shape[0] or obj.shape[1] > xin.shape[1]:
        return None
    for i in range(xin.shape[0] - obj.shape[0] + 1):
        for j in range(xin.shape[1] - obj.shape[1] + 1):
            if np.all(xin[i:i+obj.shape[0],j:j+obj.shape[1]] == obj):
                return (i,j)
    return None

def solveSymetry4(xin, xout=None, info={}):
    bg, fgc, direction = cleanLinearBackground(xin)
    if not fgc:
        return None, info
    fg = np.ones(xin.shape) * -1
    fgcc = []
    for c in lib_color_counts_dict(xin):
        if c[0] in fgc:
            fg = np.where(xin == c[0], xin, fg)
            fgcc.append(c[0])
    obj = lib_crop_bg(fg, -1)
    objs = lib_objs_split_by_bg3(obj, -1)
    objs = [o for o in objs if o.shape[0] > 1 and o.shape[1] > 1]
    odirection = 'N'
    if np.all(objs[0] == np.flipud(objs[0])):
        odirection = 'H'
    if np.all(objs[0] == np.fliplr(objs[0])):
        odirection = 'V'
    kernel = None
    xobj = np.copy(objs[0])
    opos = 'N'
    if odirection == 'V' or odirection == 'N':
        robj = np.flipud(objs[0])
        for i in range(1, robj.shape[0]):
            sel = lib_crop_bg(robj[:i,:], -1)
            if sel.shape[0] > 2 and np.where(sel[0,:] != -1, 1, 0).sum() <= 1:
                sel=sel[1:,:]
            if sel.shape[1] > 2 and np.where(sel[:,0] != -1, 1, 0).sum() <= 1:
                sel=sel[:,1:]
            if sel.shape[1] > 2 and np.where(sel[:,-1] != -1, 1, 0).sum() <= 1:
                sel=sel[:,:-1]
            if sel.shape[0] > 1 and sel.shape[1] == sel.shape[0] and np.all(sel == np.flipud(sel)) and np.all(sel == np.fliplr(sel)):
                kernel = np.copy(sel)
                xobj = np.flipud(np.copy(robj[i:,:]))
                opos = 'B'

    pos = lib_obj_in_grid(objs[0], kernel)
    kpos = lib_obj_in_grid(xin, kernel)
    xres = np.copy(xin)
    if np.any(xin[:kpos[0],kpos[1]:kpos[1]+kernel.shape[1]] == fgcc[0]):
        origobj = np.copy(xres[kpos[0]-xobj.shape[0]:kpos[0],kpos[1]-int((xobj.shape[1]-kernel.shape[1])/2):kpos[1]-int((xobj.shape[1]-kernel.shape[1])/2)+xobj.shape[1]])
        xres[kpos[0]-xobj.shape[0]:kpos[0],kpos[1]-int((xobj.shape[1]-kernel.shape[1])/2):kpos[1]-int((xobj.shape[1]-kernel.shape[1])/2)+xobj.shape[1]] = np.where(xobj!=-1, np.copy(xobj), origobj)
    if np.any(xin[kpos[0]+kernel.shape[0]:,kpos[1]:kpos[1]+kernel.shape[1]] == fgcc[0]):
        origobj = np.copy(xres[kpos[0]+kernel.shape[0]:kpos[0]+kernel.shape[0]+xobj.shape[0],kpos[1]-int((xobj.shape[1]-kernel.shape[1])/2):kpos[1]-int((xobj.shape[1]-kernel.shape[1])/2)+xobj.shape[1]])
        nobj = np.flipud(xobj)
        xres[kpos[0]+kernel.shape[0]:kpos[0]+kernel.shape[0]+xobj.shape[0],kpos[1]-int((xobj.shape[1]-kernel.shape[1])/2):kpos[1]-int((xobj.shape[1]-kernel.shape[1])/2)+xobj.shape[1]] = np.where(nobj!=-1, np.copy(nobj), origobj)
    if np.any(xin[kpos[0]:kpos[0]+kernel.shape[0],:kpos[1]] == fgcc[0]):
        nobj = np.rot90(xobj,1)
        origobj = np.copy(xres[kpos[0]-int((nobj.shape[0]-kernel.shape[0])/2):kpos[0]-int((nobj.shape[0]-kernel.shape[0])/2)+nobj.shape[0],kpos[1]-nobj.shape[1]:kpos[1]])
        xres[kpos[0]-int((nobj.shape[0]-kernel.shape[0])/2):kpos[0]-int((nobj.shape[0]-kernel.shape[0])/2)+nobj.shape[0],kpos[1]-nobj.shape[1]:kpos[1]] = np.where(nobj!=-1, np.copy(nobj), origobj)
    if np.any(xin[kpos[0]:kpos[0]+kernel.shape[0],kpos[1]+kernel.shape[1]:] == fgcc[0]):
        nobj = np.rot90(xobj,3)
        origobj = np.copy(xres[kpos[0]-int((nobj.shape[0]-kernel.shape[0])/2):kpos[0]-int((nobj.shape[0]-kernel.shape[0])/2)+nobj.shape[0],kpos[1]+kernel.shape[1]:kpos[1]+kernel.shape[1]+nobj.shape[1]])
        xres[kpos[0]-int((nobj.shape[0]-kernel.shape[0])/2):kpos[0]-int((nobj.shape[0]-kernel.shape[0])/2)+nobj.shape[0],kpos[1]+kernel.shape[1]:kpos[1]+kernel.shape[1]+nobj.shape[1]] = np.where(nobj!=-1, np.copy(nobj), origobj)
    
    return xres

def solveSymetry(xin, xout=None, info={}):
    # Solve tasks where symetry should be restored in all directions
    
    xxres = np.copy(xin)
    for i in range(4):
        xxin = np.rot90(np.copy(xin), i)
        xxout = np.rot90(np.copy(xout), i) if xout is not None else None
        try:
            xres = solveSymetry4(np.copy(xxin), xxout, info)
            if xres is not None:
                xres = np.rot90(xres, 4-i)
                xxres = np.where(xxres == xin, xres, xxres)
        except:
            pass
    return [xxres, xxres, xxres]

def solveWithAlgo(task, algo, metadata=None):
    res = None
    i = np.copy(np.asarray(task['input']))
    if algo == 1:
        return solvePatchEveryElementByPattern(i)
    if algo == 4:
        return solveBicolorMaze(i)
    if algo == 7:
        return solvePutObjsInSquareByPattern(i)
    if algo == 9:
        return solveColorVerHorByColor(i)
    if algo == 10:
        return solveAppendMissingFigure(i)
    if algo == 12:
        return solveColoredMirror(i)
    if algo == 13:
        return solveFixObjsSymetry(i)
    if algo == 14:
        return solveSimetrical(i)
    if algo == 15:
        return solveShooting(i)
    if algo == 17:
        return solveDrawBordersAroundShapes(i)
    if algo == 18:
        return solveDrawHorVerLines(i, metadata)
    if algo == 19:
        return solveColorFlow(i)
    if algo == 20:
        return solveRepeatingPattern(i)
    if algo == 21:
        return solveSymetry(i)
    if algo == 999:  # echo input
        return [i, i, i]

# json_file_path = '/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json'  
# with open(json_file_path, 'r') as f:
#     data = json.load(f)

def add_solution(sub, task_id, task_nr, attempt_1, attempt_2):
    def _valid(attempt):
        if attempt is None:
            return False
        if not isinstance(attempt, np.ndarray):
            return False
        if len(attempt.shape) != 2:
            return False
        if attempt.shape[0] <= 1 or attempt.shape[1] <= 1:
            return False
        if attempt.shape[0] > 30 or attempt.shape[1] > 30:
            return False
        return True
    
    attempts = {
        "attempt_1": attempt_1.astype(int).tolist() if _valid(attempt_1) else [[0, 0], [0, 0]],
        "attempt_2": attempt_2.astype(int).tolist() if _valid(attempt_2) else [[0, 0], [0, 0]],
    }
    
    if task_nr == 0 or task_nr == '0':
        sub[task_id] = [attempts]
    else:
        sub[task_id].append(attempts)    

def solve():
    sub = {}
    for task_id, task_data in data.items():
        algo_solved = None
        for train_nr, train_task in enumerate(task_data["train"]):
            train_solution = np.asarray(train_task["output"])
                
            for algo in [1, 4, 7, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21]: 
                try:
                    dbg(f" - trying algo {algo} for task {task_id}_{train_nr}")
                    train_res = solveWithAlgo(train_task, algo, task_data)
                    for one in train_res:
                        if train_solution.shape == one.shape and np.all(one == train_solution):
                            algo_solved = algo
                            print(f" - algo {algo} got CORRECT solution for train task {task_id}_{train_nr}: {one}")
                            break
                        else:
                            dbg(f" - algo {algo} got incorrect solution for train task {task_id}_{train_nr}: {one}")
                    if algo_solved:
                        break
                except Exception as ex:
                    dbg(f" - algo {algo} failed for train task {task_id}_{train_nr}: {ex}")
            if algo_solved:
                break  # one solution is enough to try
            
        if algo_solved is not None:
            dbg(f" - algo {algo_solved} solved task {task_id}")

            for test_nr, test_task in enumerate(task_data["test"]):
                try:
                    test_res = solveWithAlgo(test_task, algo_solved, task_data)

                    attempt_1 = test_res[0]
                    attempt_2 = test_res[1]
                    add_solution(sub, task_id, test_nr, attempt_1, attempt_2)
                    dbg(f"   * added test solution for {task_id}_{test_nr} with algo {algo_solved}")
                except Exception as ex:
                    add_solution(sub, task_id, test_nr, None, None)
                    dbg(f" - algo {algo} failed for test task {task_id}_{test_nr}: {ex}")
        
        if task_id not in sub:
            for test_nr, test_task in enumerate(task_data["test"]):
                add_solution(sub, task_id, test_nr, None, None)
                dbg(f"   * added empty test solution for {task_id}_{test_nr}")
        
    print(sub)

    with open("submission3.json", 'r') as f:
        sample = json.load(f)

    for task_id, task_data in sample.items():
        if task_id in sub:
            for task_nr, task_attempts in enumerate(task_data):
                if len(sub[task_id]) > task_nr:
                    if "attempt_1" in sub[task_id][task_nr]:
                        sub_attempt_1 = sub[task_id][task_nr]["attempt_1"]
                        sub_attempt_1 = [[int(x) for x in xx] for xx in sub_attempt_1]
                        max_sub_col = max(max(sub_attempt_1))
                        if sub_attempt_1 and max_sub_col > 0 and max_sub_col <= 9:
                            sample[task_id][task_nr]["attempt_1"] = sub_attempt_1
                    if "attempt_2" in sub[task_id][task_nr]:
                        sub_attempt_2 = sub[task_id][task_nr]["attempt_2"]
                        sub_attempt_2 = [[int(x) for x in xx] for xx in sub_attempt_2]
                        max_sub_col = max(max(sub_attempt_2))
                        if sub_attempt_2 and max_sub_col > 0 and max_sub_col <= 9:
                            sample[task_id][task_nr]["attempt_2"] = sub_attempt_2

    with open('submission.json', 'w') as f:
        json.dump(sample, f, indent=4)
