"""
http://webuser.bus.umich.edu/plenk/HB%20Conjoint%20Lenk%20DeSarbo%20Green%20Young%20MS%201996.pdf

"""
from theano import tensor as tt
import pandas as pd
import pymc3 as pm

personal_characteristics = pd.read_csv("data/computer_buyers/personal_characteristics.csv")
personal_characteristics['constant'] = 1

likelihood_to_buy = pd.read_csv("data/computer_buyers/likelihood_to_buy.csv")   # (190, 20)

design_matrix = pd.read_csv("data/computer_buyers/design_matrix.csv")
design_matrix['constant'] = 1.

N_PART_CHRC = personal_characteristics.shape[1]
N_PARTICIPANTS = likelihood_to_buy.shape[0]
N_COMP_CHRC = design_matrix.shape[1]
N_PROFILES = likelihood_to_buy.shape[1]


with pm.Model() as hierarchical_model:

    Sigma = pm.Normal("Sigma", 0, sd=10, shape=(N_PART_CHRC, N_COMP_CHRC))  # (7, 14)
    Beta = pm.Normal("Beta", tt.dot(personal_characteristics, Sigma), sd=1, shape=(N_PARTICIPANTS, N_COMP_CHRC))  # (190, 14)

    mean = tt.dot(Beta, design_matrix.T)  # (190, 20)
    sd = pm.Uniform("Individual_variance", 1e-5, 1e2)
    y = pm.Normal("observations", mean, sd=sd, shape=(N_PARTICIPANTS, N_PROFILES), observed=likelihood_to_buy)  # (190, 20)


with hierarchical_model:
    hierarchical_trace = pm.sample(draws=2000, tune=1000, n_init=50000)


df = pd.DataFrame(hierarchical_trace.get_values("Sigma").mean(0), columns=design_matrix.columns, index=personal_characteristics.columns)
