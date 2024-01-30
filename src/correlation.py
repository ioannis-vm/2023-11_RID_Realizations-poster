"""
Visualize correlations of variables with different model fitting
approaches
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pelicun.assessment import Assessment
from src import models
from src.handle_data import load_dataset
from src.handle_data import remove_collapse
from src.handle_data import only_drifts
from doc.poster.src import common
from src.util import store_info


def main():
    story_dict = {'i': '2', 'j': '3'}
    model_dict = {}

    num_realizations = 80

    df_all_edps = remove_collapse(load_dataset('data/edp_extra.parquet')[0])
    df = only_drifts(df_all_edps)

    df_all_edps = (
        df_all_edps.unstack(0)
        .unstack(0)
        .unstack(0)
        .unstack(2)
        .unstack(2)
        .unstack(1)
        .stack(4)
        .dropna(axis=1)
        .drop('PVb', level='edp', axis=1)
        # .drop('RID', level='edp', axis=1)
    )

    for index, story in story_dict.items():
        the_case = ("scbf", "9", "ii", story)  # we combine the two directions
        case_df = df[the_case].dropna().stack(level=0)
        analysis_rid_vals = case_df.dropna()["RID"].to_numpy().reshape(-1)
        analysis_pid_vals = case_df.dropna()["PID"].to_numpy().reshape(-1)

        model = models.Model_1_Weibull()
        model.add_data(analysis_pid_vals, analysis_rid_vals)
        model.censoring_limit = 0.005
        model.fit(method='quantiles')

        model_dict[index] = model

    # simulate data
    demand_sample_dict = {}
    rid_sample_dfs = []
    for hz in df_all_edps.index.get_level_values(0).unique():
        df_hz = df_all_edps.loc[hz, :]
        # re-index
        df_hz.index = pd.RangeIndex(len(df_hz.index))
        df_hz.columns = pd.MultiIndex.from_tuples(
            [(x[-1], x[-2], '1') for x in df_hz.columns], names=('edp', 'loc', 'dir')
        )
        df_hz.loc['Units', :] = [
            {'PFA': 'inps2', 'PFV': 'inps', 'PID': 'rad', 'RID': 'rad'}[x]
            for x in df_hz.columns.get_level_values('edp')
        ]

        asmt = Assessment({"PrintLog": False})  # no seed, debug
        asmt.stories = 9
        asmt.demand.load_sample(df_hz)
        asmt.demand.calibrate_model(
            {
                "ALL": {
                    "DistributionFamily": "lognormal",
                    "AddUncertainty": 0.00,
                },
                "PID": {
                    "DistributionFamily": "lognormal",
                    "TruncateLower": "",
                    "TruncateUpper": "0.10",
                    "AddUncertainty": 0.00,
                },
            }
        )
        asmt.demand.generate_sample({"SampleSize": num_realizations})
        demand_sample = asmt.demand.save_sample()
        demand_sample_dict[hz] = demand_sample

        # generate RIDs
        num_samples = len(demand_sample['PID', story_dict['i'], '1'].values)
        assert num_samples == len(demand_sample['PID', story_dict['j'], '1'].values)
        uniform_sample = np.random.uniform(0.00, 1.00, num_samples)
        # uniform_sample = None --> No RID-RID correlation
        for key in story_dict.keys():
            model = model_dict[key]
            model.uniform_sample = uniform_sample
            pids = demand_sample['PID', story_dict[key], '1'].values
            rids = model.generate_rid_samples(pids)
            rid_sample_dfs.append(pd.Series(rids, name=(hz, story_dict[key])))

    demand_sample_df = pd.concat(
        demand_sample_dict.values(), keys=demand_sample_dict.keys(), names=['hz']
    )

    rid_sample_df = pd.concat(rid_sample_dfs, axis=1)
    rid_sample_df.columns.names = ('hz', 'story')

    plt.close()
    fig, ax = plt.subplots(figsize=(9.00 / common.FIG_SCALE, 5.5 / common.FIG_SCALE))
    ax.scatter(
        demand_sample_df.loc[:, ('RID', story_dict['i'], '1')].values[::3],
        demand_sample_df.loc[:, ('RID', story_dict['j'], '1')].values[::3],
        color='C1',
        label='Direct fit',
        marker='.',
        s=1,
    )
    ax.scatter(
        df.stack(level=4)
        .loc[:, ('scbf', '9', 'ii', story_dict['i'], 'RID')]
        .values[::3],
        df.stack(level=4)
        .loc[:, ('scbf', '9', 'ii', story_dict['j'], 'RID')]
        .values[::3],
        color='black',
        label='Empirical',
        marker='.',
        s=1,
    )
    ax.scatter(
        rid_sample_df.xs(story_dict['i'], level='story', axis=1).stack().values[::3],
        rid_sample_df.xs(story_dict['j'], level='story', axis=1).stack().values[::3],
        color='C0',
        label='Conditional Weibull',
        marker='.',
        s=1,
    )
    ax.plot([0.00, 1.00], [0.00, 1.00], linestyle='dashed', color='black')
    ax.set(xlim=(-0.001, 0.026), ylim=(-0.001, 0.026))
    ax.set(xlabel=f'RID, story {story_dict["i"]}')
    ax.set(ylabel=f'RID, story {story_dict["j"]}')
    ax.grid(which='both', linewidth=0.30, linestyle='dashed')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticklabels([f'{x*100.00:.1f}%' for x in ax.get_xticks().tolist()])
    ax.set_yticklabels([f'{x*100.00:.1f}%' for x in ax.get_yticks().tolist()])
    plt.legend()
    plt.subplots_adjust(
        left=0.208, bottom=0.303, right=0.958, top=0.932, wspace=0.2, hspace=0.2
    )
    plt.savefig(store_info('doc/poster/figures/correlation.svg'))
    # plt.show()


if __name__ == '__main__':
    main()
