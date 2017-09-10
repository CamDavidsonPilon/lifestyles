from itertools import groupby, chain
from theano import tensor as tt
import pandas as pd
import pymc3 as pm
import numpy as np


def flatten(iterable):
    return list(chain.from_iterable(iterable))


def model(profiles, comparisons, selections, sample=2500):
    all_attributes = pd.get_dummies(profiles).columns
    profiles_dummies = pd.get_dummies(profiles, drop_first=True)

    choices = [profiles_dummies.loc[comparisons[profile]] for profile in comparisons.columns]

    respondants = selections.columns
    n_attributes_in_model = profiles_dummies.shape[1]
    n_participants = selections.shape[1]

    # https://www.sawtoothsoftware.com/download/ssiweb/CBCHB_Manual.pdf
    with pm.Model() as model:

        alpha = pm.Normal('alpha', 0, sd=10, shape=n_attributes_in_model, testval=np.random.randn(n_attributes_in_model))
        partsworth = pm.MvNormal("partsworth", alpha, tau=np.eye(n_attributes_in_model), shape=(n_participants, n_attributes_in_model))

        cs = [pm.Categorical("Obs%d" % (i+1),
                             tt.nnet.softmax(tt.stack([
                              tt.dot(choice.values, partsworth[i, :]) for choice in choices
                              ], axis=0).T),
                             observed=(selections[selections.columns[i]] - 1).values) for i in xrange(n_participants)]

        trace = pm.sample(sample)
    return transform_trace_to_summary_statistics(trace, respondants, profiles_dummies.columns, all_attributes)


def transform_trace_to_summary_statistics(trace, respondants, attributes_in_model, all_attributes):
    def create_linear_combination(df):
        name = df.name
        cols_to_impute_from = [c for c in attributes_in_model if c.startswith(name)]
        col_to_impute = [c for c in all_attributes if (c.startswith(name) and (c not in cols_to_impute_from))][0]
        df.loc[col_to_impute] = -df.loc[cols_to_impute_from].groupby(level=1).sum().values
        return df

    partsworth_trace = trace.get_values("partsworth")
    N, _, _ = partsworth_trace.shape
    sample_axis_index = range(N)
    df = pd.concat(
        [
            pd.DataFrame(partsworth_trace[:, :, i],
                         columns=respondants,
                         index=pd.MultiIndex.from_product([[attr], sample_axis_index])
                         )
            for i, attr in enumerate(attributes_in_model)
        ]
    )

    df = df.reindex(pd.MultiIndex.from_product([all_attributes, sample_axis_index]))
    df = df.groupby(lambda i: i.split("_")[0], level=0, group_keys=False).apply(create_linear_combination)
    return df.groupby(level=0).describe().swaplevel(axis=1)


def calculate_importance(summary_stats):
    pass


if __name__ == "__main__":
    # data from https://help.xlstat.com/customer/en/portal/articles/2062399-choice-based-conjoint-analysis-with-hierarchical-bayes-cbc-hb-
    profiles = pd.read_csv("data/lemonade/profiles.tsv", sep="\s+").set_index('Profile')
    comparisons = pd.read_csv("data/lemonade/comparisons.tsv", sep="\s+").set_index('Comparisons')
    selections = pd.read_csv("data/lemonade/selections.tsv", sep="\s+").set_index("Comparisons")
    summary_stats = model(profiles, comparisons, selections)

