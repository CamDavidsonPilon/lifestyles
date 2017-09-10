from theano import tensor as tt
import pandas as pd
import pymc3 as pm

"""
Looking at what xlstat does, they make sure that all weights in a level sum to 0
"""
profiles = pd.get_dummies(pd.read_csv("data/lemonade/profiles.tsv", sep="\s+").set_index('Profile'), drop_first=True)
comparisons = pd.read_csv("data/lemonade/comparisons.tsv", sep="\s+").set_index('Comparisons')
selections = pd.read_csv("data/lemonade/selections.tsv", sep="\s+").set_index("Comparisons")


first_choice = profiles.loc[comparisons['Profile1']]
second_choice = profiles.loc[comparisons['Profile2']]
third_choice = profiles.loc[comparisons['Profile3']]


with pm.Model() as hierarchical_model:

    weights = pm.Normal("weights", 0, sd=10., shape=(profiles.shape[1], 1))

    probs = tt.nnet.softmax(tt.stack([
             tt.dot(first_choice, weights),
             tt.dot(second_choice, weights),
             tt.dot(third_choice, weights)
            ], axis=0).T)

    cs = [pm.Categorical("Obs%d" % i, probs, observed=(selections['Individual%i' % i] - 1).values) for i in xrange(1, 11)]

with hierarchical_model:

    hierarchical_trace = pm.sample(40000, pm.Metropolis(), tune=2000)


pm.plots.traceplot(hierarchical_trace)
