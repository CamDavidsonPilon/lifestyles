from theano import tensor as tt
import pandas as pd
import pymc3 as pm
import numpy as np

# data fromhttps://help.xlstat.com/customer/en/portal/articles/2062399-choice-based-conjoint-analysis-with-hierarchical-bayes-cbc-hb-
profiles = pd.get_dummies(pd.read_csv("data/lemonade/profiles.tsv", sep="\s+").set_index('Profile'), drop_first=True)
comparisons = pd.read_csv("data/lemonade/comparisons.tsv", sep="\s+").set_index('Comparisons')
selections = pd.read_csv("data/lemonade/selections.tsv", sep="\s+").set_index("Comparisons")

first_choice = profiles.loc[comparisons['Profile1']]
second_choice = profiles.loc[comparisons['Profile2']]
third_choice = profiles.loc[comparisons['Profile3']]

p = profiles.shape[1]
participants = selections.shape[1]

# https://www.sawtoothsoftware.com/download/ssiweb/CBCHB_Manual.pdf
with pm.Model() as hierarchical_model:
    alpha = pm.Normal('alpha', 0, sd=10, shape=p, testval=np.random.randn(p))

    partsworth = pm.MvNormal("partsworth", alpha, tau=np.eye(p), shape=(participants, p))

    cs = [pm.Categorical("Obs%d" % (i+1),
            tt.nnet.softmax(tt.stack([
             tt.dot(first_choice.values, partsworth[i, :]),
             tt.dot(second_choice.values, partsworth[i, :]),
             tt.dot(third_choice.values, partsworth[i, :])
            ], axis=0).T),
        observed=(selections[selections.columns[i]] - 1).values) for i in xrange(participants)]

    trace = pm.sample(2500)
