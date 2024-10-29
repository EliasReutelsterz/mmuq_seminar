import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from hmg import HBV1D012A


def NSE(sim_dis, obs_dis):
    # obs are the "true" values, sim is our model output
    mean_obs_dis = np.mean(obs_dis)
    return 1 - (np.sum([(obs_dis[i] - sim_dis[i])**2 for i in range(len(sim_dis))])) / (np.sum([(obs_dis[i] - mean_obs_dis)**2 for i in range(len(sim_dis))]))


#TODO if log(0) appears use huge number as a replacement or something
# def LnNSE(sim_dis, obs_dis):
#     # take logs of values
#     sim_dis = np.array([np.log(x) for x in sim_dis ])




def objective_function_evaluation(prms, diso, modl_objt):

    modl_objt.set_parameters(prms)
    modl_objt.set_optimization_flag(0)

    # Run the model for the given inputs, constants and parameters.
    modl_objt.run_model()
    diss = modl_objt.get_discharge()

    # NSE calculation
    NSE_value = NSE(sim_dis=diss, obs_dis=diso)
    print(f"NSE: {NSE_value}")

    OBJ_value = 1 - NSE_value
    print(f"OBJ: {OBJ_value}")

    return OBJ_value



if __name__ == '__main__':

    #TODO implement how to choose between efficiency metrics

    main_dir = Path(r'/Users/elias/Desktop/HWS24/MMUQ Seminar/time_series__24163005')
    os.chdir(main_dir)
    inp_dfe = pd.read_csv(r'time_series___24163005.csv', sep=';', index_col=0)
    inp_dfe.index = pd.to_datetime(inp_dfe.index, format='%Y-%m-%d-%H')
    cca_srs = pd.read_csv(r'area___24163005.csv', sep=';', index_col=0)
    ccaa = cca_srs.values[0, 0]
    tems = inp_dfe.loc[:, 'tavg__ref'].values  # Temperature.
    ppts = inp_dfe.loc[:, 'pptn__ref'].values  # Preciptiation.
    pets = inp_dfe.loc[:, 'petn__ref'].values  # PET.
    diso = inp_dfe.loc[:, 'diso__ref'].values  # Observed discharge.
    modl_objt = HBV1D012A()
    tsps = tems.shape[0]  # Number of time steps.
    dslr = ccaa / (3600 * 1000)  # For daily res. multiply denominator with 24.
    modl_objt.set_inputs(tems, ppts, pets)
    modl_objt.set_outputs(tsps)
    modl_objt.set_discharge_scaler(dslr)

    bounds = [  (0.0, 0.0), # 'snw_dth'
                (-1.0, 1.0), # 'snw_ast'
                (0.0, 2.0), # 'snw_amt'
                (0.0, 0.0), # 'snw_amf'
                (0.0, 2.0), # 'snw_pmf'

                (0.0, 1e+2), # 'sl0_mse'
                (0.0, 2e+2), # 'sl1_mse'

                (0.0, 2e+2), # 'sl0_fcy'
                (0.0, 3.0), # 'sl0_bt0'

                (0.0, 4e+2), # 'sl1_pwp'
                (0.0, 4e+2), # 'sl1_fcy'
                (0.0, 4.0), # 'sl1_bt0'

                (0.0, 2e+1), # 'urr_dth'
                (0.0, 0.0), # 'lrr_dth'

                (0.0, 1.0), # 'urr_rsr'
                (3e+5, 3e+5), # 'urr_tdh'
                (0.0, 0.0), # 'urr_tdr'
                (0.0, 0.0), # 'urr_cst'
                (0.0, 0.0), # 'urr_dro'
                (0.0, 1.0), # 'urr_ulc'

                (0.0, 1e+4), # 'lrr_tdh'
                (0.0, 1.0), # 'lrr_cst'
                (0.0, 1.0), # 'lrr_dro'
              ]


    prms = np.array([
        0.00,  # 'snw_dth'
        -0.1,  # 'snw_ast'
        +0.1,  # 'snw_amt'
        0.01,  # 'snw_amf'
        0.00,  # 'snw_pmf'

        50.0,  # 'sl0_mse'
        300.,  # 'sl1_mse'

        70.0,  # 'sl0_fcy'
        2.50,  # 'sl0_bt0'

        300.,  # 'sl1_pwp'
        400.,  # 'sl1_fcy'
        2.50,  # 'sl1_bt0'

        0.00,  # 'urr_dth'
        0.00,  # 'lrr_dth'

        1.00,  # 'urr_rsr'
        30.0,  # 'urr_tdh'
        0.15,  # 'urr_tdr'
        1e-4,  # 'urr_cst'
        1.00,  # 'urr_dro'
        0.00,  # 'urr_ulc'

        0.00,  # 'lrr_tdh'
        0.00,  # 'lrr_cst'
        0.00,  # 'lrr_dro'
        ], dtype=np.float32)

    #TODO give inputs to the model object already, then just use the object as input
    objective_function_evaluation(prms, diso, modl_objt)

    result = differential_evolution(objective_function_evaluation, bounds, args=(diso, modl_objt))

    print(f"result.x: {result.x}")
    print(f"result.fun: {result.fun}")

    result_prms = result.x

    # for default maxiter=1000 we get
    # result.x: [ 0.00000000e+00 -8.50456759e-01  1.55656933e+00  0.00000000e+00
    #   1.23171246e+00  7.28789541e+01  1.84629921e+02  1.44196315e+02
    #   2.15978555e+00  1.56338210e+02  7.51584389e+01  8.02963602e-01
    #   2.42125374e+00  0.00000000e+00  9.34173302e-01  3.00000000e+05
    #   0.00000000e+00  0.00000000e+00  0.00000000e+00  9.37866241e-03
    #   6.71475300e+02  1.66273918e-01  9.97923827e-01]
    # result.fun: 0.09209441373643845