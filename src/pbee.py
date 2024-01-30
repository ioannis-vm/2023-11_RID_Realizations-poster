"""
Loss estimation with FEMA P-58, using pelicun
"""

import pickle
import matplotlib.pyplot as plt
from src.util import store_info
from extra.improved_methodology_demo.src.p58_assessment import sa_t1
from doc.poster.src import common


def main():
    with open('extra/improved_methodology_demo/results/out.pcl', 'rb') as f:
        df = pickle.load(f)

    _, ax = plt.subplots(
        figsize=(9.00 / common.FIG_SCALE, 5.5 / common.FIG_SCALE),
    )

    def add_lines(model, color, linestyle='solid'):
        df_cost = df[model].xs('repair_cost', axis=1, level=1) / 1e6  # million
        df_cost_describe = df_cost.describe()
        mean = df_cost_describe.loc['mean', :]

        ax.plot(
            sa_t1.values(),
            mean.values,
            color=color,
            label=model,
            linestyle=linestyle,
        )

    add_lines('Conditional Weibull', 'C0')
    add_lines('FEMA P-58', 'C1')
    add_lines('Empirical', 'black', 'dashed')
    ax.legend()

    ax.grid(which='both', linewidth=0.30, linestyle='dashed')
    ax.set(xlabel='$Sa(T_1)$ (g)', ylabel='Repair Cost\n(million USD)')
    plt.tight_layout()
    plt.savefig(store_info('doc/poster/figures/loss_estimation.svg'))
    # plt.show()


if __name__ == '__main__':
    main()
