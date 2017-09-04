from theano import tensor as tt
import pandas as pd
import pymc3 as pm
from StringIO import StringIO
import numpy as np


_ = StringIO("""
Profile Temperature Sugar   Lemon   Intensity
1    Ice 1sugar no  Low
2    Warm    NoSugar    no  Low
3    Ice 2sugar no  Medium
4    VeryWarm   1sugar no  Strong
5    Warm    1sugar yes Medium
6    VeryWarm   2sugar no  Medium
7    Ice 2sugar yes Strong
8    Warm    2sugar no  Strong
9    Warm    2sugar yes Low
10   VeryWarm   2sugar yes Low
11   VeryWarm   NoSugar    yes Strong
12   Ice NoSugar    yes Medium
""")
profiles = pd.get_dummies(pd.read_csv(_, sep="\s+").set_index('Profile'), drop_first=True)

_ = StringIO("""
Comparisons Profile1    Profile2    Profile3
1    8   9   7
2    10  3   12
3    11  2   6
4    1   8   4
5    5   10  2
6    3   11  9
7    12  6   7
8    7   1   4
9    12  10  11
10   8   6   3
11   9   2   5
12   2   12  8
13   1   9   6
14   7   5   3
15   11  4   10
16   3   1   2
17   4   12  9
18   10  7   8
19   5   11  1
20   6   4   5
""")
choices = pd.read_csv(_, sep="\s+").set_index('Comparisons')


first_choice = profiles.loc[choices['Profile1']]
second_choice = profiles.loc[choices['Profile2']]
third_choice = profiles.loc[choices['Profile3']]



_ = StringIO("""
Comparisons  Individual1    Individual2    Individual3    Individual4    Individual5    Individual6    Individual7    Individual8    Individual9    Individual10
1  1   2   3   2   2   3   3   2   2   1
2  1   3   2   3   3   1   2   1   1   3
3  1   1   1   1   1   2   3   1   1   1
4  2   1   3   3   1   3   1   1   2   1
5  3   2   1   1   3   1   3   2   3   3
6  2   2   3   2   1   1   2   3   2   1
7  3   1   2   1   1   1   1   2   2   1
8  3   3   1   1   3   3   2   2   2   1
9  1   3   1   1   2   1   1   2   2   1
10 1   1   3   3   3   3   1   1   3   1
11 3   2   1   3   2   2   1   2   3   2
12 1   2   1   2   1   2   3   3   1   1
13 3   2   2   3   2   1   2   2   3   3
14 3   1   1   1   3   3   3   2   3   1
15 3   2   2   3   1   1   3   3   2   3
16 1   3   3   1   1   2   1   3   2   2
17 1   3   1   3   2   1   1   2   2   1
18 3   3   3   3   3   2   1   2   1   3
19 1   3   1   3   1   1   1   2   3   3
20 3   3   1   2   1   3   1   3   2   2
""")
selections = pd.read_csv(_, sep="\s+").set_index("Comparisons")


with pm.Model() as hierarchical_model:

    weights = pm.Normal("weights", 0, sd=10.0, shape=(7, 1))

    probs = tt.nnet.softmax(tt.stack([
             tt.dot(first_choice, weights),
             tt.dot(second_choice, weights),
             tt.dot(third_choice, weights)
            ], axis=0).reshape((20, 3)))

    cs = [pm.Categorical("Obs%d" % i, probs, observed=(selections['Individual%i' % i] - 1).values) for i in xrange(1, 11)]

with hierarchical_model:
    hierarchical_trace = pm.sample(40000, pm.Metropolis(), tune=5000)


pm.plots.traceplot(hierarchical_trace)
