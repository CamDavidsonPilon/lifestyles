"""

This model looks like the following:

Y_i = X_i \beta_i + \epsilon_i   for i=1..n

\beta_i = \Sigma z_i + \delta_i

where z_i are individual characteristics of the participants and X_i and the characteristic of the
objects (profiles) shown to user i. Y_i encode the responses.

The matrix \Sigma represents the unobserved preferences, by participant characterstic
and object characteristic. Thus, each participant has a specific preference vector, \beta_i

Example output of \Sigma:

               Hot line       RAM  Screen size  CPU speed  Hard disk  \
APPLY    mean -0.007746  0.044346     0.010688   0.019450   0.017453
         std   0.052518  0.051294     0.053317   0.054735   0.053340
EXPERT   mean  0.036692 -0.044133     0.010544   0.038605   0.048322
         std   0.048852  0.047047     0.048433   0.047586   0.049023
FEMALE   mean  0.301551 -0.007823    -0.116979  -0.009133  -0.059673
         std   0.188446  0.184779     0.189560   0.192246   0.189363
OWN      mean  0.025637 -0.014327     0.046543  -0.288823  -0.022155
         std   0.193098  0.194544     0.202576   0.191677   0.205398
TECH     mean -0.092104  0.216937    -0.037374   0.111902   0.039594
         std   0.219524  0.229970     0.221848   0.225858   0.223730
YEARS    mean -0.031091  0.153652     0.052316   0.206464  -0.004942
         std   0.180996  0.181811     0.183941   0.178567   0.178801
constant mean -0.171966  0.286664     0.071539   0.162412  -0.267776
         std   0.373027  0.364603     0.383818   0.370149   0.386035

                 CD ROM     Cache  Color of unit  Availability  Warranty  \
APPLY    mean  0.030005 -0.024019      -0.007257     -0.023260  0.003275
         std   0.052011  0.053753       0.054286      0.052572  0.052126
EXPERT   mean -0.014856  0.037270       0.004334     -0.018608 -0.001328
         std   0.047208  0.049132       0.049493      0.046833  0.047744
FEMALE   mean -0.176534 -0.050838       0.001616      0.017704  0.114448
         std   0.189728  0.189738       0.190505      0.181998  0.189794
OWN      mean  0.126685 -0.046220      -0.029179      0.039143 -0.039319
         std   0.185128  0.189676       0.196250      0.189792  0.196129
TECH     mean -0.015982  0.070722       0.028498      0.121370  0.012058
         std   0.229553  0.218412       0.227133      0.227576  0.233561
YEARS    mean -0.070300  0.011943      -0.117638     -0.125296 -0.028073
         std   0.183298  0.186200       0.182523      0.177328  0.188354
constant mean  0.475872 -0.166859       0.014595      0.215484  0.129247
         std   0.369968  0.386870       0.385288      0.366806  0.369530

               Software  Guarantee     Price  constant
APPLY    mean  0.015383   0.000679 -0.010254  0.177272
         std   0.052421   0.053174  0.052620  0.051181
EXPERT   mean -0.014422   0.003025  0.052377  0.123728
         std   0.048148   0.048165  0.048086  0.048768
FEMALE   mean -0.048464  -0.001087  0.345567  0.043020
         std   0.194091   0.188258  0.188386  0.186807
OWN      mean  0.130883  -0.083200 -0.005931 -0.037465
         std   0.184213   0.195836  0.189383  0.194186
TECH     mean -0.064105  -0.134026 -0.097655  0.006192
         std   0.222548   0.227758  0.230823  0.233296
YEARS    mean  0.109149  -0.057188 -0.125526 -0.111568
         std   0.188263   0.182854  0.186182  0.185695
constant mean  0.180477   0.286788 -1.506407  3.229457
         std   0.373614   0.362738  0.367813  0.383238



http://webuser.bus.umich.edu/plenk/HB%20Conjoint%20Lenk%20DeSarbo%20Green%20Young%20MS%201996.pdf

"""
from theano import tensor as tt
import pandas as pd
import pymc3 as pm

personal_characteristics = pd.read_csv("data/computer_buyers/personal_characteristics.csv") # (190, 6)
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
    hierarchical_trace = pm.sample(draws=2500, tune=1000, n_init=25000)


df_mean = pd.DataFrame(hierarchical_trace.get_values("Sigma").mean(0), columns=design_matrix.columns, index=personal_characteristics.columns)
df_std = pd.DataFrame(hierarchical_trace.get_values("Sigma").std(0), columns=design_matrix.columns, index=personal_characteristics.columns)

df = pd.concat([df_mean, df_std], keys=['mean', 'std']).swaplevel().sort_index()
