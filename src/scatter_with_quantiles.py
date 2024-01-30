"""
Scatter plots with fitted distributions
   - [single archetype], [single story]
   - histogram of marginals on the sides
   - a single distribution (Weibull)

"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.models import Model_0_P58
from src.models import Model_1_Weibull
from src.handle_data import load_dataset
from src.handle_data import only_drifts
from src.handle_data import remove_collapse
from doc.poster.src import common
from src.util import store_info



def load_data():
    data = only_drifts(remove_collapse(load_dataset('data/edp_extra.parquet')[0]))
    return data


def fit_models(data):
    models = {}

    story = '3'
    the_case = ("scbf", "9", "ii", story)
    case_df = data[the_case].dropna().stack(level=0)

    rid_vals = case_df.dropna()["RID"].to_numpy().reshape(-1)
    pid_vals = case_df.dropna()["PID"].to_numpy().reshape(-1)

    model = Model_1_Weibull()
    model.add_data(pid_vals, rid_vals)
    model.censoring_limit = 0.0005
    model.fit(method='quantiles')

    num_realizations = 10000
    pid_sim = np.random.choice(pid_vals, size=num_realizations, replace=True)
    model.generate_rid_samples(pid_sim)

    model_fema = Model_0_P58()
    model_fema.add_data(pid_vals, rid_vals)
    model_fema.fit(delta_y=0.4 / 100.00, beta=0.80)
    model_fema.generate_rid_samples(pid_sim)

    models['Weibull'] = model
    models['FEMA P-58'] = model_fema

    return models


def make_plot(models):
    xlim = (-0.002, 0.022)
    ylim = (-0.005, 0.058)

    model = models['Weibull']
    model_p58 = models['FEMA P-58']

    plt.close()
    fig, axs = plt.subplots(
        3,
        2,
        figsize=(9.00 / common.FIG_SCALE, 9.0 / common.FIG_SCALE),
        gridspec_kw={'width_ratios': [5, 1], 'height_ratios': [3, 1, 5]},
    )

    # main plot
    ax = axs[2, 0]

    data = load_data()[("scbf", "9", "ii", common.story)].dropna().stack(level=0)
    hz = common.hazard_level_of_focus
    ax.scatter(
        data.loc[hz, 'RID'],
        data.loc[hz, 'PID'],
        color='green',
        marker='.',
        s=4
    )
    ax.scatter(
        data['RID'],
        data['PID'],
        color='black',
        marker='.',
        s=0.1
    )

    model.calculate_rolling_quantiles()

    ax.plot(model.rolling_rid_50, model.rolling_pid, 'k', label='Empirical')
    ax.plot(model.rolling_rid_20, model.rolling_pid, 'k', linestyle='dashed')
    ax.plot(model.rolling_rid_80, model.rolling_pid, 'k', linestyle='dashed')

    model_pid = np.linspace(0.00, 0.06, 1000)
    model_rid_50 = model.evaluate_inverse_cdf(0.50, model_pid)
    model_rid_20 = model.evaluate_inverse_cdf(0.20, model_pid)
    model_rid_80 = model.evaluate_inverse_cdf(0.80, model_pid)

    ax.plot(model_rid_50, model_pid, 'C0', label='Conditional Weibull', zorder=10)
    ax.plot(model_rid_20, model_pid, 'C0', linestyle='dashed', zorder=10)
    ax.plot(model_rid_80, model_pid, 'C0', linestyle='dashed', zorder=10)

    model_rid_50 = model_p58.evaluate_inverse_cdf(0.50, model_pid)
    model_rid_20 = model_p58.evaluate_inverse_cdf(0.20, model_pid)
    model_rid_80 = model_p58.evaluate_inverse_cdf(0.80, model_pid)

    ax.plot(model_rid_50, model_pid, 'C1', label='FEMA P-58', zorder=5)
    ax.plot(model_rid_20, model_pid, 'C1', linestyle='dashed', zorder=5)
    ax.plot(model_rid_80, model_pid, 'C1', linestyle='dashed', zorder=5)

    ax.legend(fontsize='6')

    ax.axvline(x=model.censoring_limit, color='black', linestyle='dotted')

    ax.set(xlim=xlim, ylim=ylim, xlabel='RID', ylabel='PID')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # transform axis labels
    ax.set_xticklabels([f'{x*100.00:.1f}%' for x in ax.get_xticks().tolist()])
    ax.set_yticklabels([f'{x*100.00:.1f}%' for x in ax.get_yticks().tolist()])
    ax.grid(which='both', linestyle='dashed', linewidth=0.30)

    # histogram of PIDs
    ax = axs[2, 1]

    ax.hist(
        model.raw_pid,
        bins=25,
        orientation='horizontal',
        density=True,
        edgecolor='black',
        facecolor='lightgrey',
        zorder=10,
        range=(0, ylim[1]),
    )

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # remove axis labels
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set(ylim=ylim)
    ax.grid(which='both', linestyle='dashed', linewidth=0.30, zorder=-100)

    # histogram of RIDs
    ax = axs[1, 0]

    bins = np.concatenate(
        (
            np.array((0.00, model.censoring_limit)),
            np.linspace(model.censoring_limit, xlim[1], 24),
        )
    )

    ax.hist(
        model.raw_rid,
        bins=bins,
        density=True,
        edgecolor='black',
        facecolor='lightgrey',
        zorder=10,
        range=(0, xlim[1]),
    )
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # remove axis labels
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set(xlim=xlim)
    ax.set(ylim=(0.00, 250.00))
    ax.grid(which='both', linestyle='dashed', linewidth=0.30, zorder=-100)

    # ecdf plot
    ax = axs[0, 0]

    vals = np.random.choice(data.loc[hz, 'PID'].values, size=10000, replace=True)
    models['Weibull'].generate_rid_samples(vals)
    models['FEMA P-58'].generate_rid_samples(vals)

    sns.ecdfplot(data.loc[hz, 'RID'], ax=ax, label='Empirical', color='black')
    sns.ecdfplot(
        models['Weibull'].sim_rid, ax=ax, label='Conditional Weibull', color='C0'
    )
    sns.ecdfplot(models['FEMA P-58'].sim_rid, ax=ax, label='FEMA P-58', color='C1')
    ax.set(xlim=xlim, ylabel='CDF', xlabel=None)
    ax.set_xticklabels([f'{x*100.00:.1f}%' for x in ax.get_xticks().tolist()])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(which='both', linestyle='dashed', linewidth=0.30)
    ax.set_xticklabels([])

    # remove other axs
    fig.delaxes(axs[0, 1])
    fig.delaxes(axs[1, 1])

    fig.tight_layout()
    plt.savefig(store_info('doc/poster/figures/scatter_with_quantiles.svg'))
    # plt.show()


def main():
    data = load_data()
    models = fit_models(data)
    make_plot(models)


if __name__ == '__main__':
    main()
