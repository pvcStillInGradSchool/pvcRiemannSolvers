from matplotlib import pyplot as plt
from matplotlib.figure import Figure


line_styles = [
    ('dotted',                (0, (1, 1))),
    ('loosely dotted',        (0, (1, 4))),
    ('dashed',                (0, (4, 4))),
    ('densely dashed',        (0, (4, 1))),
    ('loosely dashed',        (0, (4, 8))),
    ('long dash with offset', (4, (8, 2))),
    ('dashdotted',            (0, (2, 4, 1, 4))),
    ('densely dashdotted',    (0, (2, 1, 1, 1))),
    ('loosely dashdotted',    (0, (2, 8, 1, 8))),
    ('dashdotdotted',         (0, (2, 4, 1, 4, 1, 4))),
    ('densely dashdotdotted', (0, (2, 1, 1, 1, 1, 1))),
    ('loosely dashdotdotted', (0, (2, 8, 1, 8, 1, 8))),
]


def savefig(file_name: str, folder_name: str = '../build/output',
        fig: Figure = None):
    svg_name = f'{folder_name}/{file_name}.svg'
    pdf_name = f'{folder_name}/{file_name}.pdf'
    if fig:
      fig.savefig(svg_name)
      fig.savefig(pdf_name)
    else:
      plt.savefig(svg_name)
      plt.savefig(pdf_name)


if __name__ == '__main__':
    pass
