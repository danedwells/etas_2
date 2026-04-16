#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# simulation of catalog continuation (for forecasting)
#
# as described by Mizrahi et al., 2021
# Leila Mizrahi, Shyam Nandan, Stefan Wiemer;
# Embracing Data Incompleteness for Better Earthquake Forecasting.
# Journal of Geophysical Research: Solid Earth.
# doi: https://doi.org/10.1029/2021JB022379
###############################################################################


import json
import logging

from etas import set_up_logger
from etas.inversion import ETASParameterCalculation
from etas.simulation import ETASSimulation
import pandas as pd

set_up_logger(level=logging.INFO)

if __name__ == '__main__':

    with open('./config/forecast_catalog_continuation_config.json', 'r') as f:
        forecast_config = json.load(f)

    fn_inversion_output = forecast_config['fn_inversion_output']
    fn_store_simulation = forecast_config['fn_store_simulation']
    forecast_duration = forecast_config['forecast_duration']
    n_simulations = forecast_config['n_simulations']

    # load output from inversion
    with open(fn_inversion_output, 'r') as f:
        inversion_output = json.load(f)
    print(inversion_output)
    etas_invert = ETASParameterCalculation.load_calculation(
        inversion_output)

    # initialize simulation

    m_max = forecast_config.get('m_max', None)
    simulation = ETASSimulation(etas_invert, m_max=m_max)
    simulation.prepare()


    # to store the forecast in a csv instead of just producting it,
    # do the following:
    # simulation.simulate_to_csv(fn_store_simulation, forecast_duration,
    #                            n_simulations)

    store = pd.DataFrame()
    
    for i,chunk in simulation.simulate(forecast_duration, n_simulations):
        store = pd.concat([store, chunk],
                          ignore_index=False)

    # Save to csv of some sort
    print(f"Number of events: {len(store)}")
    store.to_csv(f"./output_data/forecast_catalog_continuation.csv")
