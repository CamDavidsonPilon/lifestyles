from theano import tensor as tt
import pandas as pd
import pymc3 as pm
import numpy as np


def _create_observation_variable(individual_selections, choices, partsworth):
    """
    This function handles creating the PyMC3 observation variables.  It also gracefully handles missing observations in individual selections.

    `individual_selections` is a Series of the individuals selections made, starting from 0. It can contain NaNs which represent answer was not provided.

    `choices` is a DataFrame with a hierarchical index: level=0 enumerates the choices, and level=1 displays the profile at a specific choice.
    It's size is (n_questions, n_choices_per_question).

    `partsworth` is a slice of PyMC3 matrix. It represents the partsworth variables of a individual. Size is (n_profiles,)

    This computes the values exp(partsworth * profile_j) / sum[ exp(partsworth * profile_k ] for all j.
    """
    nan_mask = pd.notnull(individual_selections)
    return pm.Categorical("Obs_%s" % individual_selections.name,
                          tt.nnet.softmax(tt.stack([
                            tt.dot(choice.values, partsworth) for _, choice in choices[nan_mask.values].groupby(axis=1, level=0)
                          ], axis=0).T),
                          observed=individual_selections[nan_mask.values].values)


def model(profiles, comparisons, selections, sample=2500, alpha_prior_std=10):
    all_attributes = pd.get_dummies(profiles).columns
    profiles_dummies = pd.get_dummies(profiles, drop_first=True)
    choices = pd.concat({profile: profiles_dummies.loc[comparisons[profile]].reset_index(drop=True) for profile in comparisons.columns}, axis=1)

    respondants = selections.columns
    n_attributes_in_model = profiles_dummies.shape[1]
    n_participants = selections.shape[1]

    with pm.Model():

        # https://www.sawtoothsoftware.com/download/ssiweb/CBCHB_Manual.pdf
        # need to include the covariance matrix as a parent of `partsworth`
        alpha = pm.Normal('alpha', 0, sd=alpha_prior_std, shape=n_attributes_in_model, testval=np.random.randn(n_attributes_in_model))
        partsworth = pm.MvNormal("partsworth", alpha, tau=np.eye(n_attributes_in_model), shape=(n_participants, n_attributes_in_model))

        cs = [_create_observation_variable(selection, choices, partsworth[i, :]) for i, (_, selection) in enumerate(selections.iteritems())]

        trace = pm.sample(sample)
    return transform_trace_to_individual_summary_statistics(trace, respondants, profiles_dummies.columns, all_attributes)


def transform_trace_to_individual_summary_statistics(trace, respondants, attributes_in_model, all_attributes):

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


def calculate_individual_importance(individual_summary_stats):
    pass


if __name__ == "__main__":
    # data from https://help.xlstat.com/customer/en/portal/articles/2062399-choice-based-conjoint-analysis-with-hierarchical-bayes-cbc-hb-
    profiles = pd.read_csv("data/lemonade/profiles.tsv", sep="\s+").set_index('Profile')
    comparisons = pd.read_csv("data/lemonade/comparisons.tsv", sep="\s+").set_index('Comparisons')
    selections = pd.read_csv("data/lemonade/selections.tsv", sep="\s+").set_index("Comparisons") - 1
    ind_summary_stats = model(profiles, comparisons, selections)
    print ind_summary_stats
