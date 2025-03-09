# DeconFlow

This repository is the official code base accompanying our paper

<a href="https://arxiv.org/abs/2408.05647v1">Controlling for Discrete Unmeasured Confounding in Nonlinear Causal Models</a>.



main_deconflow.py is the main file from which ray tune experiments are started

dependendencies.py contains required packages


flow_analysis.py generates Figures for the synthetic experiments
flow_architecture.py contains the flow architecture

flow_auxiliary_fns.py contains auxiliary functions for the ray tune experiments
	and the function implementing the deconfounding (deconf_permutation)

toy_data.py generates synthetic data
get_toy_data.py defines data loaders

application_twins_data_preparation.py prepares the twins data
application_twins_analysis.py generates the Figure for the twins experiment
