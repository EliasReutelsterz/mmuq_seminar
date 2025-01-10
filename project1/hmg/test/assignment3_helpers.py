import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path
from multiprocessing import Manager, Pool
from multiprocessing.pool import ThreadPool
from queue import Queue

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hmg import HBV1D012A

DEBUG_FLAG = False

# Some monospaced font for plots.
plt.rcParams['font.family'] = 'sans-serif'


def sobol_algorithm(metric, narrow_range, sobl_smpl_size):

    main_dir = Path(r'/Users/elias/Desktop/HWS24/MMUQ Seminar/time_series__24163005')
    os.chdir(main_dir)

    prms_buds_dict = {
        'snw_dth': (0.00, 10.0),
        'snw_ast': (-1.0, +1.0),
        'snw_amt': (-0.0, +2.0),
        'snw_amf': (0.00, 2.00),
        'snw_pmf': (0.00, 2.00),

        'sl0_mse': (0.00, 1e+2),
        'sl1_mse': (0.00, 2e+2),

        'sl0_fcy': (5.00, 4e+1),
        'sl0_bt0': (1.00, 6.00),

        'sl1_pwp': (1.00, 2e+2),
        'sl1_fcy': (1e+2, 4e+2),
        'sl1_bt0': (1.00, 3.00),

        'urr_dth': (0.00, 2e+1),
        'lrr_dth': (0.00, 5.00),

        'urr_rsr': (0.00, 1.00),
        'urr_tdh': (0.00, 1e+2),
        'urr_tdr': (0.00, 1.00),
        'urr_cst': (1e-4, 1.00),
        'urr_dro': (1.00, 1.00),
        'urr_ulc': (0.00, 1.00),

        'lrr_tdh': (5e+2, 1e+4),
        'lrr_cst': (0.00, 1.00),
        'lrr_dro': (0.00, 1.00),
        }

    inp_dfe = pd.read_csv(r'time_series___24163005.csv', sep=';', index_col=0).iloc[:365 * 24 * 2,:]
    inp_dfe.index = pd.to_datetime(inp_dfe.index, format='%Y-%m-%d-%H')

    cca_srs = pd.read_csv(r'area___24163005.csv', sep=';', index_col=0)

    ccaa = cca_srs.values[0, 0]

    tems = inp_dfe.loc[:, 'tavg__ref'].values
    ppts = inp_dfe.loc[:, 'pptn__ref'].values
    pets = inp_dfe.loc[:, 'petn__ref'].values

    diso = inp_dfe.loc[:, 'diso__ref'].values  # Observed discharge.

    tsps = tems.shape[0]

    dslr = ccaa / (3600 * 1000)  # For daily res. multiply denominator with 24.

    sobl_text_name = f'sobl_inds_mprg_{metric}_narrow_{narrow_range}.csv'
    sobl_figr_name = f'sobl_inds_mprg_{metric}_narrow_{narrow_range}.png'

    dtyp = np.float64

    mprg_pool_size = 8 #! 8 for my macbook, 4 should be taken! 
    #==========================================================================

    if (sobl_smpl_size % mprg_pool_size):

        sobl_smpl_size += (mprg_pool_size - (sobl_smpl_size % mprg_pool_size))

        assert (sobl_smpl_size % mprg_pool_size) == 0

        print('Adjusted sobl_smpl_size to', sobl_smpl_size)

    modl_objt = HBV1D012A()

    prms_buds = modl_objt.get_parameter_bounds_in_correct_order(prms_buds_dict)

    prms_buds = prms_buds.astype(dtyp)

    prms_cont = len(prms_buds)

    modl_objt.set_inputs(tems, ppts, pets)
    modl_objt.set_outputs(tsps)

    modl_objt.set_discharge_scaler(dslr)
    #==========================================================================

    optn_args = OPTNARGS()

    optn_args.cntr = 0
    optn_args.oflg = 0
    optn_args.vbse = False
    optn_args.diso = diso
    optn_args.modl_objt = modl_objt
    optn_args.take_idxs = np.isfinite(diso)
    optn_args.metric = metric

    modl_objt.set_optimization_flag(1)
    #==========================================================================

    #==========================================================================
    # Queuing and multiprocessing.
    #==========================================================================

    if mprg_pool_size == 1:
        ques = [(Queue(), Queue()) for _ in range(mprg_pool_size)]

    else:
        mngr = Manager()

        ques = [(mngr.Queue(), mngr.Queue()) for _ in range(mprg_pool_size)]

    # Manager Pool cannot work with a generator!
    mprg_args = [(
        tdid,
        optn_args,
        get_objv_fntn_vlue,
        ques[tdid][0],
        ques[tdid][1],
        )
        for tdid in range(mprg_pool_size)]

    if mprg_pool_size == 1:
        mprg_pool = ThreadPool(1)

    else:
        mprg_pool = Pool(mprg_pool_size)
        wait_for_pool_init(mprg_pool, mprg_pool_size, 'SOBOL', True)

    mprg_pool.map_async(get_objv_fntn_vlue_mprg, mprg_args)

    assert all([ques[tdid][1].get() == 1 for tdid in range(mprg_pool_size)])
    #==========================================================================

    bgn_tmr = timeit.default_timer()

    #==========================================================================
    # Sobol Matrix A (left) computations.
    #==========================================================================

    A = np.full((sobl_smpl_size, prms_cont), np.nan, dtype=dtyp)
    f_A = np.full(sobl_smpl_size, np.nan, dtype=dtyp)

    fill_sobl_mtrx(
        A,
        prms_buds,
        prms_cont,
        sobl_smpl_size,
        narrow_range)

    fill_ofvs_arry(
        ques,
        A,
        f_A,
        mprg_pool_size,
        sobl_smpl_size)

    #==========================================================================
    # Sobol Matrix B (middle) computations.
    #==========================================================================

    B = np.full((sobl_smpl_size, prms_cont), np.nan, dtype=dtyp)
    f_B = np.full(sobl_smpl_size, np.nan, dtype=dtyp)

    fill_sobl_mtrx(
        B,
        prms_buds,
        prms_cont,
        sobl_smpl_size,
        narrow_range)

    fill_ofvs_arry(
        ques,
        B,
        f_B,
        mprg_pool_size,
        sobl_smpl_size)

    #==========================================================================
    # Sobol f0^2.
    #==========================================================================
    
    f_0_squared = np.mean(f_A) * np.mean(f_B)

    #==========================================================================
    # Sobol Matrices Cs (right) computations.
    #==========================================================================

    ssis = np.full(prms_cont, np.nan, dtype=dtyp)
    stis = np.full(prms_cont, np.nan, dtype=dtyp)

    for i in range(prms_cont):
        A_B_i = np.zeros((sobl_smpl_size, prms_cont))
        for param_index in range(prms_cont):
            if param_index == i:
                A_B_i[:, param_index] = B[:, param_index]
            else:
                A_B_i[:, param_index] = A[:, param_index]
        
        f_A_B_i = np.full(sobl_smpl_size, np.nan, dtype=dtyp)
        fill_ofvs_arry(
        ques,
        A_B_i,
        f_A_B_i,
        mprg_pool_size,
        sobl_smpl_size)
        

        denominator = 1/sobl_smpl_size * np.sum([f_A[j]**2 for j in range(sobl_smpl_size)]) - f_0_squared
        ssis[i] = (1/sobl_smpl_size * np.sum([f_B[j] * (f_A_B_i[j] - f_A[j]) for j in range(sobl_smpl_size)])) / denominator
        stis[i] = (1/sobl_smpl_size * np.sum([f_A[j] * (f_A[j] - f_A_B_i[j]) for j in range(sobl_smpl_size)])) / denominator

    print('S:')
    print([f'{sis:.2E}' for sis in ssis])

    print(f'Sum S:{ssis.sum():.2E}')

    print('S_T:')
    print([f'{sti:.2E}' for sti in stis])
    print(f'Sum S_T:{stis.sum():.2E}')
    #==========================================================================

    [ques[tdid][0].put(None) for tdid in range(mprg_pool_size)]

    end_tmr = timeit.default_timer()

    print(f'Took {end_tmr - bgn_tmr:0.2E} seconds to complete.')
    #==========================================================================

    inds_dtfe = pd.DataFrame(
        index=list(modl_objt.get_parameter_labels()),
        data={'ssis': ssis, 'stis': stis})

    inds_dtfe.to_csv(sobl_text_name, sep=';', float_format='%.4E')

    figr, axes = plt.subplots(
        1,
        1,
        sharex=True,
        squeeze=False,
        figsize=(6.3, 3.5),
        layout='constrained')

    axes = axes.ravel()

    axis_idxs, = axes

    wdth = 0.3

    inds_dtfe['ssis'].plot.bar(
        ax=axis_idxs,
        alpha=0.95,
        color='C0',
        label='First Order',
        align='edge',
        width=-wdth)

    inds_dtfe['stis'].plot.bar(
        ax=axis_idxs,
        alpha=0.95,
        color='C1',
        label='Total Effect',
        align='edge',
        width=+wdth)

    axis_idxs.set_ylabel('Sensitivity [-]')

    axis_idxs.grid()
    axis_idxs.set_axisbelow(True)

    axis_idxs.legend()

    if narrow_range:
        plt.suptitle(f'Sample Size: {sobl_smpl_size} Narrow Range')
    else:
        plt.suptitle(f'Sample Size: {sobl_smpl_size} Full Range')

    plt.show()

    main_dir = Path(r'/Users/elias/Desktop/HWS24/MMUQ Seminar/mmuq_seminar/project1/hmg/test/assignment3_images')
    os.chdir(main_dir)

    plt.savefig(sobl_figr_name, bbox_inches='tight')

    plt.close(figr)

    return

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

def sample_narrow_range(prm_index, sobl_smpl_size, prms_buds):
    opt_param_value = opt_prms[prm_index]
    range_factor = 0.2
    if opt_param_value == 0:
        print("For the parameter index: ", prm_index, " the optimal value is 0")
        return None
    elif opt_param_value < 0:
        low = np.max([prms_buds[prm_index, 0], opt_param_value * (1 + range_factor)])
        high = np.min([prms_buds[prm_index, 1], opt_param_value * (1 - range_factor)]) 
    else:
        low = np.max([prms_buds[prm_index, 0], opt_param_value * (1 - range_factor)])
        high = np.min([prms_buds[prm_index, 1], opt_param_value * (1 + range_factor)])
    return np.random.uniform(low=low, high=high, size=sobl_smpl_size)


def fill_sobl_mtrx(
        mtrx_sobl,
        prms_buds,
        prms_cont,
        sobl_smpl_size,
        narrow_range):

    if narrow_range:
        for prm_index in range(prms_cont):
            mtrx_sobl[:, prm_index] = sample_narrow_range(prm_index, sobl_smpl_size, prms_buds)
    else:
        for prm_index in range(prms_cont):
            mtrx_sobl[:, prm_index] = np.random.uniform(low=prms_buds[prm_index, 0], high=prms_buds[prm_index, 1], size=sobl_smpl_size)
    return


def fill_ofvs_arry(
        ques,
        mtrx_sobl,
        ofvs_sobl,
        mprg_pool_size,
        sobl_smpl_size):

    vecs_perr_thrd = sobl_smpl_size // mprg_pool_size

    for k in range(mprg_pool_size):
        ques[k][0].put(
            mtrx_sobl[k * vecs_perr_thrd:((k + 1) * vecs_perr_thrd),:])

    for k in range(mprg_pool_size):
        objt_vles = ques[k][1].get()

        assert np.all(np.isfinite(objt_vles))

        ofvs_sobl[k * vecs_perr_thrd:((k + 1) * vecs_perr_thrd)] = objt_vles

    return


def get_objv_fntn_vlue_mprg(args):

    tdid, optn_args, objt_fntn, queu_recv, queu_send = args

    _ = tdid

    queu_send.put(1)

    while True:

        recv_objt = queu_recv.get()

        if recv_objt is None:
            break

        try:
            prms_vecs = recv_objt

            ofvs = np.full(prms_vecs.shape[0], np.nan)

            for i in range(prms_vecs.shape[0]):
                ofvs[i] = objt_fntn(prms_vecs[i,:].copy(), optn_args)

            queu_send.put(ofvs)

        except:
            queu_send.put(None)

            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

    return


def obj_function_by_NSE(sim_dis, obs_dis):
    # obs are the "true" values, sim is our model output
    mean_obs_dis = np.mean(obs_dis)
    # Objective function is 1 - NSE(...)
    return (np.sum((obs_dis - sim_dis) ** 2)) / (np.sum((obs_dis - mean_obs_dis) ** 2))


# Global variable
count_zeros = []
mean_count_zeros = 0

def obj_function_by_LnNSE(sim_dis, obs_dis):

    global count_zeros
    zero_count = np.sum(sim_dis == 0)
    count_zeros.append(zero_count)
    mean_count_zeros = np.mean(count_zeros)
    print(f"Mean count of zeros: {mean_count_zeros} of total {len(sim_dis)}")

    sim_dis[sim_dis <= 0] = 0.00001
    obs_dis[obs_dis <= 0] = 0.00001
    ln_sim_dis = np.log(sim_dis)
    ln_obs_dis = np.log(obs_dis)
    mean_ln_obs_dis = np.mean(ln_obs_dis)
    # Objective function is 1 - LnNSE(...)
    obj_func_value = (np.sum((ln_obs_dis - ln_sim_dis) ** 2)) / (np.sum((ln_obs_dis - mean_ln_obs_dis) ** 2))
    return obj_func_value

def obj_function_by_LnNSE_random_pen(sim_dis, obs_dis):

    global count_zeros
    zero_count = np.sum(sim_dis == 0)
    count_zeros.append(zero_count)
    mean_count_zeros = np.mean(count_zeros)
    print(f"Mean count of zeros: {mean_count_zeros} of total {len(sim_dis)}")

    for i in range(len(sim_dis)):
        if sim_dis[i] == 0:
            sim_dis[i] = 10**(- np.random.uniform(1, 7))
    obs_dis[obs_dis <= 0] = 0.00001 #! shouldnt be necessary
    ln_sim_dis = np.log(sim_dis)
    ln_obs_dis = np.log(obs_dis)
    mean_ln_obs_dis = np.mean(ln_obs_dis)
    # Objective function is 1 - LnNSE(...)
    obj_func_value = (np.sum((ln_obs_dis - ln_sim_dis) ** 2)) / (np.sum((ln_obs_dis - mean_ln_obs_dis) ** 2))
    return obj_func_value

def get_objv_fntn_vlue(prms, args):

    diso = args.diso[args.take_idxs]
    modl_objt = args.modl_objt

    modl_objt.set_parameters(prms)
    modl_objt.run_model()

    diss = modl_objt.get_discharge()[args.take_idxs]

    if args.metric == 'NSE':
        objv_fntn_vlue = obj_function_by_NSE(sim_dis=diss, obs_dis=diso)
    elif args.metric == 'LnNSE':
        objv_fntn_vlue = obj_function_by_LnNSE(sim_dis=diss, obs_dis=diso)
    elif args.metric == 'LnNSE-random_pen':
        objv_fntn_vlue = obj_function_by_LnNSE_random_pen(sim_dis=diss, obs_dis=diso)
    else:
        raise ValueError("Invalid metric")

    args.cntr += 1

    if args.vbse:
        print(
            args.cntr,
            objv_fntn_vlue)

    assert np.isfinite(objv_fntn_vlue)

    return objv_fntn_vlue


def wait_for_pool_init(mp_pool, pol_sze, pol_nme, vb):

    trs = 0
    tmr_bgn = timeit.default_timer()
    while True:
        pol_res = list(mp_pool.map(get_pid, range(pol_sze), chunksize=1))

        time.sleep(0.2)

        trs += 1

        if len(pol_res) == len(set(pol_res)): break

    tmr_end = timeit.default_timer()

    if vb: print(
        f'{pol_nme}', '|',
        os.getppid(), '|',
        pol_res, '|',
        trs, '|',
        f'{tmr_end - tmr_bgn:0.2f} secs')
    return


def get_pid(args):

    _ = args

    return os.getpid()


class OPTNARGS: pass
    


if __name__ == '__main__':


    metric = "NSE"
    narrow_range = False
    sobl_smpl_size = int(100)

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack,
    # 2. "up" move the stack up to an older frame,
    # 3. "down" move the stack down to a newer frame, and
    # 4. "interact" start an interactive interpreter.
    #==========================================================================

    if DEBUG_FLAG:
        try:
            sobol_algorithm(metric=metric, narrow_range=narrow_range, sobl_smpl_size = sobl_smpl_size)

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

            import pdb
            pdb.post_mortem()
    else:
        sobol_algorithm(metric=metric, narrow_range=narrow_range, sobl_smpl_size = sobl_smpl_size)

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))