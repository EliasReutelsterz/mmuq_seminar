import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import time

from hmg import HBV1D012A

bounds = {
    # Snow.
    'snw_dth': (0.0, 10.0), # Initial depth [L].
    'snw_ast': (-1.0, 1.0), # Air snow TEM [K].
    'snw_amt': (0.0, 2.0), # Air melt TEM [K].
    'snw_amf': (0.0, 2.0), # Air melt factor [L/TK].
    'snw_pmf': (0.0, 2.0), # PPT melt factor [L/LTK].

    # Soils.
    'sl0_mse': (0.0, 1e+2), # Soil 0 initial depth [L].
    'sl1_mse': (0.0, 2e+2), # Soil 1 initial depth [L].

    # Soil 0.
    'sl0_fcy': (5.0, 4e+1), # Field capacity [L].
    'sl0_bt0': (1.0, 6.0), # Beta [-].

    # Soil 1.
    'sl1_pwp': (1.0, 2e+2), # PWP [L].
    'sl1_fcy': (1e+2, 4e+2), # Field capacity [L].
    'sl1_bt0': (1.0, 3.0), # Beta [-].

    # Routing reservoirs.
    'urr_dth': (0.0, 2e+1), # URR initial depth [L].
    'lrr_dth': (0.0, 5.0), # LRR initial depth [L].

    # Upper reservoir.
    'urr_rsr': (0.0, 1.0), # Runoff split ratio [-].
    'urr_tdh': (0.0, 1e+2), # Threshold depth [L].
    'urr_tdr': (0.0, 1.0), # Threshold DIS const. [1/T].
    'urr_cst': (1e-4, 1.0), # RNF const. [1/T].
    'urr_dro': (1.0, 1.0), # DIS ratio [-].
    'urr_ulc': (0.0, 1.0), # URR-to-LRR const. [1/T].

    # Lower reservoir.
    'lrr_tdh': (5e+2, 1e+4), # Threshold depth [L].
    'lrr_cst': (0.0, 1.0), # Runoff const. [1/T].
    'lrr_dro': (0.0, 1.0) # Discharge ratio [-].
}



opt_prms = np.array([6.70180268e-01,
    -4.73635541e-01,
    1.39573083e+00,
    2.85948067e-01,
    2.96108258e-01,
    9.06437515e+01,
    1.06834473e+01,
    5.04459911e+00,
    1.09250502e+00,
    1.90613268e+02,
    1.55454530e+02,
    1.73108194e+00,
    4.43894918e-01,
    5.25192818e-01,
    9.90426386e-01,
    2.82355194e+01,
    8.14094054e-01,
    1.03801474e-04,
    1.00000000e+00,
    1.96578290e-01,
    7.39459909e+02,
    8.81377843e-03,
    9.99256261e-01], dtype=np.float32)


def NSE(sim_dis, obs_dis):
    # obs are the "true" values, sim is our model output
    mean_obs_dis = np.mean(obs_dis)
    return 1 - (np.sum((obs_dis - sim_dis) ** 2)) / (np.sum((obs_dis - mean_obs_dis) ** 2))

def LnNSE(sim_dis, obs_dis):
    sim_dis[sim_dis <= 0] = 0.00001
    obs_dis[obs_dis <= 0] = 0.00001
    ln_sim_dis = np.log(sim_dis)
    ln_obs_dis = np.log(obs_dis)
    mean_ln_obs_dis = np.mean(ln_obs_dis)
    return 1 - (np.sum((ln_obs_dis - ln_sim_dis) ** 2)) / (np.sum((ln_obs_dis - mean_ln_obs_dis) ** 2))

def KGE(sim_dis, obs_dis):
    raise ValueError("Not implemented yet")


def objective_function(prms, diso, modl_objt, efficieny_metric, parameter_in_optimization = [], obj_values= []):

    allowed_metrics = {NSE, LnNSE, KGE}
    if efficieny_metric not in allowed_metrics:
        raise ValueError(f"efficieny_metric must be one of {allowed_metrics}")

    modl_objt.set_parameters(prms)
    modl_objt.set_optimization_flag(0)

    # Run the model for the given inputs, constants and parameters.
    modl_objt.run_model()
    diss = modl_objt.get_discharge()

    # NSE calculation
    efficiency_value = efficieny_metric(sim_dis=diss, obs_dis=diso)

    obj_value = 1 - efficiency_value

    parameter_in_optimization.append(prms.copy())
    obj_values.append(obj_value)
    return obj_value

def setup_object_with_data(tems: np.array, ppts: np.array):
    main_dir = Path(r'/Users/elias/Desktop/HWS24/MMUQ Seminar/time_series__24163005')
    os.chdir(main_dir)
    inp_dfe = pd.read_csv(r'time_series___24163005.csv', sep=';', index_col=0)
    inp_dfe.index = pd.to_datetime(inp_dfe.index, format='%Y-%m-%d-%H')
    cca_srs = pd.read_csv(r'area___24163005.csv', sep=';', index_col=0)
    ccaa = cca_srs.values[0, 0]
    pets = inp_dfe.loc[:, 'petn__ref'].values  # PET.
    modl_objt = HBV1D012A()
    tsps = tems.shape[0]  # Number of time steps.
    dslr = ccaa / (3600 * 1000)  # For daily res. multiply denominator with 24.
    modl_objt.set_inputs(tems, ppts, pets)
    modl_objt.set_outputs(tsps)
    modl_objt.set_discharge_scaler(dslr)
    return modl_objt

def get_diso():
    main_dir = Path(r'/Users/elias/Desktop/HWS24/MMUQ Seminar/time_series__24163005')
    os.chdir(main_dir)
    inp_dfe = pd.read_csv(r'time_series___24163005.csv', sep=';', index_col=0)
    inp_dfe.index = pd.to_datetime(inp_dfe.index, format='%Y-%m-%d-%H')
    diso = inp_dfe.loc[:, 'diso__ref'].values  # Observed discharge.
    return diso


def perturb_inputs(num: float = 2000):
    # diso = get_diso() # Observed discharge
    perturbation_factors = np.random.uniform(0.75, 1.25, num)
    main_dir = Path(r'/Users/elias/Desktop/HWS24/MMUQ Seminar/time_series__24163005')
    os.chdir(main_dir)
    inp_dfe = pd.read_csv(r'time_series___24163005.csv', sep=';', index_col=0)
    inp_dfe.index = pd.to_datetime(inp_dfe.index, format='%Y-%m-%d-%H')
    ref_tem = inp_dfe.loc[:, 'tavg__ref'].values # reference temperature
    ref_ppt = inp_dfe.loc[:, 'pptn__ref'].values # reference preciptiation

    tems = []
    ppts = []
    for perturbation_factor in perturbation_factors:
        tems.append(perturbation_factor * ref_tem)
        ppts.append(perturbation_factor * ref_ppt)
    return ref_tem, np.array(tems), ref_ppt, np.array(ppts), perturbation_factors

def run_model_with_perturbed_inputs(tems, ppts, ref_tem, ref_ppt, maxiter_differential_evolution=20):
    ref_params = opt_prms
    metric = NSE
    recalib_params = []
    recalib_obj_values = []
    for i in range(tems.shape[0]):
        tic = time.time()
        modl_objt = setup_object_with_data(tems=tems[i], ppts=ppts[i])
        diso = get_diso() # Observed discharge
        result = differential_evolution(func=objective_function,
                                        bounds=list(bounds.values()),
                                        maxiter=maxiter_differential_evolution,
                                        polish=False,
                                        x0=ref_params,
                                        args=(diso, modl_objt, metric))
        recalib_params.append(result.x)
        recalib_obj_values.append(result.fun)
        toc = time.time()
        print(f"Iteration {i + 1}/{tems.shape[0]}, Time elapsed: {toc - tic} seconds")

    modl_objt = setup_object_with_data(tems=ref_tem, ppts=ref_ppt)
    diso = get_diso()
    ref_obj_value = objective_function(ref_params, diso, modl_objt, efficieny_metric=NSE)

    return np.array(recalib_params), np.array(recalib_obj_values), ref_params, ref_obj_value 


def run_model_with_perturbed_inputs_save_results():
    maxiter_differential_evolution=20
    num = 2000 #! later 2000

    ref_tem, tems, ref_ppt, ppts, perturbation_factors = perturb_inputs(num=num)
    x_labels = pd.read_csv(r'time_series___24163005.csv', sep=';', index_col=0)
    x_labels.index = pd.to_datetime(x_labels.index, format='%Y-%m-%d-%H')
    recalib_params, recalib_obj_values, ref_params, ref_obj_value = run_model_with_perturbed_inputs(tems, ppts, ref_tem, ref_ppt, maxiter_differential_evolution)
    script_directory = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_directory, "assignment4_data")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "recalib_params.npy"), recalib_params)
    np.save(os.path.join(save_dir, "recalib_obj_values.npy"), recalib_obj_values)
    np.save(os.path.join(save_dir, "ref_params.npy"), ref_params)
    np.save(os.path.join(save_dir, "ref_obj_value.npy"), ref_obj_value)
    np.save(os.path.join(save_dir, "perturbation_factors.npy"), perturbation_factors)
    np.save(os.path.join(save_dir, 'tems.npy'), tems)
    np.save(os.path.join(save_dir, 'ppts.npy'), ppts)
    print("Results saved successfully")