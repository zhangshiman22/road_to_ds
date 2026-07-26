
import torch
import matplotlib.pyplot as plt

def use_svg_display():
    """完美适配最新版 Jupyter 的 SVG 渲染"""
    try:
        from matplotlib_inline import backend_inline
        backend_inline.set_matplotlib_formats('svg')
    except ImportError:
        pass

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置坐标轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def set_figsize(figsize=(3.5, 2.5)):
    """设置图表大小"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', '--', 'g-'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []
    set_figsize(figsize)
    axes = axes if axes else plt.gca()
    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim ==1 or isinstance(X, list) and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(x,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

def synthetic_data(w, b, num_examples): #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w))) #len(w) 代表了你这个回归模型中“输入特征的个数”（即因变量的维度）
    y = torch.matmul(X, w) + b #矩阵乘法，维度会发生变化
    y += torch.normal(0, 0.01, y.shape) #加噪音
    return X, y.reshape((-1, 1)) #-1 代表让系统自动去算行数（系统自动填入 1000）,1 代表固定为 1 列，为了防止反向传播疯狂复制导致计算爆炸



# 为了让你能用 d2l.plt.plot
from types import ModuleType
import sys
current_module = sys.modules[__name__]
current_module.plt = plt

fake_torch = ModuleType('torch')
fake_torch.plot = plot
fake_torch.set_figsize = set_figsize
fake_torch.plt = plt
sys.modules['d2l.torch'] = fake_torch
