import matplotlib.pyplot as plt
import numpy as np

from knitwork.common.base import format_readable_num
from knitwork.common.torch import to_numpy
from knitwork.common.utils import dont_throw


@dont_throw('EVAL context window plot')
def plot_bpc_by_context_pos(cw_ix_bpc, *, step, fig_id=0):
    fig, ax = plt.subplots(figsize=(8, 4))

    cw_ix_bpc = to_numpy(cw_ix_bpc, copy=False)
    cw = len(cw_ix_bpc)

    fig = plt.figure(num=fig_id+1, clear=True)
    ax = fig.subplots()

    ax.plot(np.arange(cw), cw_ix_bpc, lw=1.5)
    ax.set_xlabel('Position in context window (tokens)')
    ax.set_ylabel('BPC')
    ax.set_title(f'BPC by context window ({cw}) position [step={format_readable_num(step)}]')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig