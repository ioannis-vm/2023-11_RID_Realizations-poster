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

    num_hz = len(df.columns.get_level_values(1).unique())

    _, ax = plt.subplots(
        figsize=(9.00 / common.FIG_SCALE, 5.5 / common.FIG_SCALE),
    )

    def add_lines(model, color, linestyle='solid'):
        df_cost = df[model].xs('repair_cost', axis=1, level=1) / 1e6  # million
        df_cost_describe = df_cost.describe()
        mean = df_cost_describe.loc['mean', :]
        std = df_cost_describe.loc['std', :]
        mean_plus = mean + std
        mean_minus = mean - std

        for i in range(num_hz):
            ax.scatter(
                [sa_t1[f'{i + 1}']],
                [df_cost.iloc[:, i].mean()],
                edgecolors=color,
                facecolor='white',
                s=8,
            )
        ax.plot(
            sa_t1.values(),
            mean_plus.values,
            color=color,
            linestyle='dashed',
            linewidth=0.1,
        )
        ax.plot(
            sa_t1.values(),
            mean_minus.values,
            color=color,
            linestyle='dashed',
            linewidth=0.1,
        )
        if model != 'FEMA P-58 optimized':
            label = model
        else:
            label = None
        ax.plot(
            sa_t1.values(),
            mean.values,
            color=color,
            label=label,
            linestyle=linestyle,
        )

    add_lines('Conditional Weibull', 'C0')
    add_lines('FEMA P-58', 'C1')
    add_lines('FEMA P-58 optimized', 'lightgrey')
    add_lines('Empirical', 'black', 'dashed')
    ax.legend()

    ax.grid(which='both', linewidth=0.30, linestyle='dashed')
    ax.set(xlabel='$Sa(T_1)$ (g)', ylabel='Repair Cost\n(million USD)')
    plt.tight_layout()
    # plt.savefig(store_info('doc/poster/figures/loss_estimation.svg'))
    plt.show()


if __name__ == '__main__':
    main()
