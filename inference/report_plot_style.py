"""Shared restrained figure style; captions live in the report, not the image."""

import matplotlib.pyplot as plt


PANEL_LABELS = {
    'TEXT8: обучение (EMA)': 'Text8 / Train',
    'SDQ: обучение (EMA)': 'SDQ / Train',
    'TEXT8: независимая validation': 'Text8 / Validation',
    'SDQ: независимая validation': 'SDQ / Validation',
    'Text8: финальный checkpoint': 'Text8',
    'SDQ: финальный checkpoint': 'SDQ',
    'Обычный SDQ: расстояние от сохранения до запроса': 'SDQ',
    'Отдельная проверка: нужная запись не перезаписывается': 'Fixed gaps',
    'Ошибка на неизменной held-out задаче': 'Validation',
    'Норма градиента до clipping (EMA)': 'Train / EMA',
    'TEXT8': 'Text8',
    'SDQ': 'SDQ',
    'Sdq': 'SDQ',
    'Seed 42': '',
    'Transformer · separate horizon': 'Transformer',
    'Validation': '',
}


def prepare_figure(fig):
    if fig._suptitle is not None:
        fig._suptitle.remove()
        fig._suptitle = None
    for ax in fig.axes:
        title = ax.get_title()
        if 'Training loop' in title:
            ax.set_xlabel('Train time (h)')
            title = title.split(' · ')[0].title()
        elif 'Reported CUDA peak' in title:
            ax.set_xlabel('CUDA peak (GiB)')
            title = title.split(' · ')[0].title()
        elif 'положительное = первый вариант лучше' in title:
            title = title.split(' · ')[0].title()
        elif title.startswith('Δ vs '):
            title = ''
        title = PANEL_LABELS.get(title, title)
        ax.set_title(title, fontsize=10, fontweight='normal', loc='center', pad=9)
        if ax.get_xlabel() in ('Tokens (M)', 'Slots (M)'):
            ax.set_xlim(left=0)
        if ax.get_ylabel().startswith('Skipped'):
            ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=9, width=0.65, length=3)
        ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_fontsize(10)
        for name, spine in ax.spines.items():
            spine.set_linewidth(0.65)
            spine.set_color('#333333')
            if name in ('top', 'right'):
                spine.set_visible(False)
        for line in ax.get_xgridlines() + ax.get_ygridlines():
            line.set_color('#e1e5e8')
            line.set_linewidth(0.55)
            line.set_alpha(0.65)
        old = ax.get_legend()
        if old is not None:
            handles, labels = ax.get_legend_handles_labels()
            old.remove()
            ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.015, 1),
                borderaxespad=0, frameon=False, fontsize=8, handlelength=2.1,
                labelspacing=0.5, ncol=1)
    fig.canvas.draw()


def text_overlaps(fig):
    """Find material text collisions after layout, excluding shared tick corners."""
    renderer = fig.canvas.get_renderer()
    texts = []
    for ax in fig.axes:
        candidates = list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        xlo, xhi = sorted(ax.get_xlim())
        ylo, yhi = sorted(ax.get_ylim())
        candidates += [label for label in ax.get_xticklabels() if xlo <= label.get_position()[0] <= xhi]
        candidates += [label for label in ax.get_yticklabels() if ylo <= label.get_position()[1] <= yhi]
        legend = ax.get_legend()
        if legend is not None:
            candidates += legend.get_texts()
        for text in candidates:
            if text.get_visible() and text.get_text().strip():
                box = text.get_window_extent(renderer)
                if box.width > 0 and box.height > 0:
                    texts.append((text.get_text(), box))
    collisions = []
    for i, (a, x) in enumerate(texts):
        for b, y in texts[i+1:]:
            width = min(x.x1, y.x1) - max(x.x0, y.x0)
            height = min(x.y1, y.y1) - max(x.y0, y.y0)
            if width > 2 and height > 2 and width*height > 0.12*min(x.width*x.height, y.width*y.height):
                collisions.append([a, b])
    return collisions
