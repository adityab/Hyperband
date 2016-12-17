import math
import numpy as np

# you need to write the following hooks for your custom problem

def get_random_hyperparameter_configuration():

    t = np.random.uniform()

    return t

def run_then_return_val_loss(nepochs, hyperparameters, noiselevel):

    xcur = hyperparameters                      # hyperparameter value to be evaluated
    xopt = 0.8                                  # true optimum location on x-axis when infinite number of epochs
    xshift = 0.2                                # shift of the optimum in the decision space, i.e., x1
    xopt = xopt - xshift/math.sqrt(nepochs)     # denoised suggestion when running for nepochs

    yvalue = math.pow( math.fabs(xopt - xcur), 0.5)     # actual objective function = distance to the optimum
    yvalue = yvalue + 0.5/nepochs               # plus additive term
    yvalue = yvalue * (1 + math.fabs(np.random.normal(0, noiselevel)))    # multiplicative noise

    return yvalue

iscenario = 1

if (iscenario == 1):
    stat_filename = "fprofile.txt"              # output filename
    stat_file = open(stat_filename, 'w+', 0)

    nx = 1001           # number of function evaluations / solutions
    noiselevel = 0.2    # noise level of the objective function
    for nepochs in [1, 3, 9, 27, 81]:   # with different resolution / number of epochs
        for i in range(0, nx):                  # for different hyperparameter values
            x_i = i * 1.0 / (nx-1)              # which are equispaced in [0, 1]
            y_i = run_then_return_val_loss(nepochs, x_i, noiselevel)        # we evaluate them
            if (i < nx-1):      stat_file.write("{:.15g}\t".format(y_i))    # and save to the file
            else:               stat_file.write("{:.15g}\n".format(y_i))
    stat_file.close()

    nruns = 100
    for irun in range(0, 100):
        stat_filename = "randomsearch_{}.txt".format(irun)
        stat_file = open(stat_filename, 'w+', 0)
        nrandom = 100
        for i in range(nrandom):
            x_i = get_random_hyperparameter_configuration()
            y_i = run_then_return_val_loss(81, x_i, noiselevel)
            if (i == 0):                    x_best_observed = x_i;     y_best_observed = y_i
            if (y_i < y_best_observed):     y_best_observed = y_i;     x_best_observed = x_i
            # now compute what would be denoised loss values of the best observed solution
            # use stddev 1e-10 to set noise to ~0 similar to averaging over tons of runs
            y_best_observed_denoised = run_then_return_val_loss(81, x_best_observed, 1e-10)
            y_i = run_then_return_val_loss(81, x_best_observed, y_best_observed_denoised)
            stat_file.write("{}\t{:.15g}\n".format(i, y_i))
        stat_file.close()

if (iscenario == 2):

    max_iter = 81   # maximum iterations/epochs per configuration
    eta = 3         # defines downsampling rate (default=3)
    logeta = lambda x: math.log(x)/math.log(eta)
    s_max = int(logeta(max_iter))   # number of unique executions of Successive Halving (minus one)
    B = (s_max+1)*max_iter          # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

    noiselevel = 0.2  # noise level of the objective function
    # Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.

    nruns = 1       # set it to e.g. 10 when testing hyperband against randomsearch
    for irun in range(0, 100):
        hband_results_filename = "hyperband_{}.txt".format(irun)
        hband_file = open(hband_results_filename, 'w+', 0)

        x_best_observed = []
        x_best_observed_nep = 0

        nevals = 0       # total number of full (with max_iter epochs) evaluations used so far

        for s in reversed(range(s_max+1)):

            stat_filename = "hband_benchmark_{}.txt".format(s)
            stat_file = open(stat_filename, 'w+', 0)

            n = int(math.ceil(B/max_iter/(s+1)*eta**s)) # initial number of configurations
            r = max_iter*eta**(-s)      # initial number of iterations to run configurations for

            # Begin Finite Horizon Successive Halving with (n,r)
            T = [ get_random_hyperparameter_configuration() for i in range(n) ]
            for i in range(s+1):
                # Run each of the n_i configs for r_i iterations and keep best n_i/eta
                n_i = n*eta**(-i)
                r_i = r*eta**(i)
                val_losses = [ run_then_return_val_loss(nepochs=r_i, hyperparameters=t,
                                                        noiselevel=noiselevel) for t in T ]
                nevals = nevals + len(T) * r_i / 81
                argsortidx = np.argsort(val_losses)

                if (x_best_observed == []):
                    x_best_observed = T[argsortidx[0]]
                    y_best_observed = val_losses[argsortidx[0]]
                    x_best_observed_nep = r_i
                # only if better AND based on >= number of epochs, the latter is optional
                if (val_losses[argsortidx[0]] < y_best_observed):# and (r_i >= x_best_observed_nep):
                    x_best_observed_nep = r_i
                    y_best_observed = val_losses[argsortidx[0]]
                    x_best_observed = T[argsortidx[0]]

                for j in range(0, len(T)):
                    stat_file.write("{:.15g}\t{:.15g}\n".format(T[j], val_losses[j]))
                T = [ T[i] for i in argsortidx[0:int( n_i/eta )] ]

                # suppose the current best solution w.r.t. validation loss is our recommendation
                # then let's evaluate it in noiseless settings (~= averaging over tons of runs)
                if (len(T)):
                    f_recommendation = run_then_return_val_loss(81, x_best_observed, 1e-10) # 81 epochs and 1e-10 ~= zero noise
                hband_file.write("{:.15g}\t{:.15g}\n".format(nevals, f_recommendation))
            # End Finite Horizon Successive Halving with (n,r)

            stat_file.close()
        hband_file.close()