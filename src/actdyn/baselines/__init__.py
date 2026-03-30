"""
Empirical baselines wired in this repository (CSC415 actdyn).

policy_type in YAML ``model`` section:

- ``act_cvae`` — ACT-style chunk policy (Transformer CVAE); compare execution modes.
- ``bc_mlp`` — single-step BC MLP (``configs/lift_bc_mlp.yaml``).
- ``knn_bc`` — kNN over (obs, action) train pairs, VINN-style nonparametric BC (``configs/lift_knn_bc.yaml``).
- ``bet_dbc`` — discrete behavior tokens via k-means + MLP (``configs/lift_bet_style.yaml``).

RT-1-scale vision--language policies are not instantiated (state-only Lift); cite as related work.
"""
