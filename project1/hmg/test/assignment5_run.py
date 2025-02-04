from assignment5_helpers import *

if __name__ == "__main__":
    maxiter_differential_evolution = 400
    num = 250
    script_directory = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.join(script_directory, "assignment5_data")
    rating_curve_params = np.load(os.path.join(main_dir, 'rating_curve_params.npy'))
    hs_perturbed, h_perturbation_factors = perturb_depth(num=num)
    run_model_with_perturbed_depths(hs_perturbed, h_perturbation_factors, rating_curve_params, maxiter_differential_evolution=maxiter_differential_evolution)
