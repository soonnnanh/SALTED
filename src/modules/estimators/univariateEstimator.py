import os, sys
import numpy as np
import pandas as pd

os.chdir("./src")
from modules.helpers.plotters import plotCustomiser

class univariateEstimator:
    def __init__(self, times=None, events=None, times_enter = None, type = 'kaplan-meier'):


        self.times = times
        self.events = events
        self.times_enter = times_enter
        self.type = type

        assert len(times) == len(events)
        if times_enter is not None: 
            assert (len(times) - len(times_enter)) == 0


    def fit(self): 
        try:
            """
            Fits the data provided to either kaplan-meier estimator for default or survival function, 
            or nelson-aalen if you're looking for the cumulative hazard function instead. 

            Args: 
            self(class object)

            Returns: 
            self(class object): adds the relevant survival analysis attributes upon completion
            """
            self.timepoints, self.n_at_risk, self.n_events = self.get_counts(self.times, self.events, self.times_enter)

            if self.type == 'kaplan-meier': 
                self.surv_func, self.surv_variance, self.surv_se = self.kaplanmeier(self.timepoints, self.n_at_risk, self.n_events)
            elif self.type == 'nelson-aalen': 
                self.cumhaz, self.cumhaz_variance, self.cumhaz_se = self.nelsonaalen(self.timepoints, self.n_at_risk, self.n_events)

            return

        except AssertionError as AssError1:
            print("{}. Survival analysis requires time as an independent variable or there's nothing to survive".format(AssError1))
        except Exception as e :
            print("fitting of estimator failed due to {}".format(e))

    def fit_interval_censored(self): 
        try: 
            """
            Takes the time difference of times & times-entered and treats the data as a non-interval censored data for fitting 
            Why? Because stack overflow said it's the best way to do so. QED

            Args: 
            self(class object)

            Returns: 
            self(class object): adds the relevant survival analysis attributes upon completion of fit()
            """
            if (self.times_enter is None) | (self.times is None): 
                raise ValueError("both time & times_enter, the upper and lower bound of the time interval respectively are required ")
            if (self.times_enter > self.times).any(): 
                raise ValueError("Upper bound of time interval needs to be less than lower bound of time interval")

            time_diff = np.asarray(self.times) - np.asarray(self.times_enter)
            self.times = time_diff
            self.times_enter = np.zeros(len(self.times))
            self.fit()

            return self

        except Exception as e :
            print("fitting of estimator using interval-censored data failed due to {}".format(e))

    def get_fits(self): 
        """
        Transformation of cumulative hazard s.e. to survival function s.e. is from 

        https://stats.stackexchange.com/questions/319957/is-there-a-straight-forward-way-to-transform-the-s-e-of-the-cumulative-hazard-i


        """
        try:
            if self.surv_func is None: 
                if self.cumhaz is not None: 
                    self.surv_func = np.exp(-self.cumhaz)
                    self.surv_var = self.cumhaz_variance/ np.square((self.surv_func))
                    self.surv_se = np.asarray(self.cumhaz_se) / np.asarray(self.surv_func)
                else:
                    print("Survival time for probability of interest cannot be estimated if there is no survival function")
                    print("Fitting Kaplan-Meier (Thank me later)...")
                    self.fit()

        except Exception as e: 
            print("Unknown error occured while trying to get fitted estimators : {}".format(e))
    
    def get_counts(self, times, events, times_enter):
        try:
            """
            Calculates the number of the population at risk and those who are dead at each unique time point. 
            Note that for left-censored data, times_enter indicates the times the patients enter the data. 

            Args: 
            times(array-like int/float): 
            events(array-like boolean/int):
            times_enter(array-like int/float): 

            Returns 
            timepoints(array_like, int) : unique time points from the list of times provided 
            n_at_risk(array_like, int) : number of individuals who are at risk of the event at each unique timepoint
            n_death(array_like, int) :  number of individuals who have died(or have undergone event of interest) at each unique timepoint
            """
            # times, events, times_enter = np.asarray(times), np.asarray(events), np.asarray(times_enter)


            self.n_sample = len(times)
            if times_enter is not None: 
                times, events, times_enter = np.asarray(times), np.asarray(events), np.asarray(times_enter)
                timepoints = np.sort(np.unique(np.concatenate((times_enter, times))), kind="mergesort")
            else: 
                times, events = np.asarray(times), np.asarray(events)
                timepoints = np.sort(np.unique(times))

            n_at_risk = np.zeros(len(timepoints))
            n_events  = np.zeros(len(timepoints))

            t0 = timepoints[0]
            n_death = 0 
            sum_events = 0 
            sum_entry = 0 
            for i in range(len(timepoints)):
                n_at_risk[i] = self.n_sample - sum_events + sum_entry
                t_i_event_index = np.where(times == timepoints[i])[0]
                if (i != 0) & (times_enter is not None) : 
                    t_i_entry_index = np.where(times_enter == timepoints[i])[0]
                    try:
                        n_entry_t_i = len(t_i_entry_index)
                    except:
                        n_entry_t_i = 0 
                else: 
                    n_entry_t_i = 0
                t_i_n_events = events[t_i_event_index]
                # print(t_i_n_events)
                n_death = np.sum(t_i_n_events)
                n_event_i = len(t_i_n_events)
                sum_entry += n_entry_t_i
                sum_events += n_event_i
                # n_at_risk[i] = self.n_sample - sum_events + sum_entry
                n_events[i] = n_death

            return timepoints, n_at_risk, n_events

        def get_confidence_intervals_quantile(self, q, method = 'linear', alpha = 0.05): 
            """
            Adapted mostly from the statsmodel computation of confidence interval, 
            which is the approach used in SAS. 
            """
            try: 
                assert (q > 0) & (q <=1)

                if self.surv_func is None: 
                    self.get_fits()

                tr = norm.ppf(1 - alpha / 2)

                method = method.lower()
                
                if method == "linear":
                    g = lambda x: x
                    g_prime = lambda x: 1
                elif method == "log":
                    g = lambda x: np.log(x)
                    g_prime = lambda x: 1 / x
                elif method == "arcsinesqrt":
                    g = lambda x: np.arcsin(np.sqrt(x))
                    g_prime = lambda x: 1 / (2 * np.sqrt(x) * np.sqrt(1 - x))
                else:
                    raise ValueError("unknown method")

                r = g(self.surv_func) - g(1 - q)
                r /= (g_prime(self.surv_func) * self.surv_se)

                ii = np.flatnonzero(np.abs(r) <= tr)
                if len(ii) == 0:
                    return np.nan, np.nan

                lb = self.timepoints[ii[0]]

                if ii[-1] == len(self.timepoints) - 1:
                    ub = np.inf
                else:
                    ub = self.timepoints[ii[-1] + 1]

                return lb, ub

        except Exception as e: 
            print("failed helper function-> calculating confidence intervals for quantile {} :{}".format(q,e))

    def kaplanmeier(self, timepoints, n_at_risk, n_events, variance='greenwood'): 
        """
        Calculates the survival function & its associated variance using the Kaplan-Meier method. 
        Note that currently the variance & standard error calculation supported is the Greenwood's method

        Args: 
        timepoints(array-like int/float): 
        n_at_risk(array-like int): 
        n_event(array-like int):
        variance(str): 

        Returns: 
        surv_prob(array-like float): 
        surv_variance(array-like float):
        """
        try: 
            surv_func = np.ones(len(timepoints))
            surv_variance = np.zeros(len(timepoints))
            surv_se = np.zeros(len(timepoints))
            est_surv = 1
            surv_var_term = 1
            for i in range(1, len(timepoints)): 
                est_surv = est_surv * (1 - n_events[i]/n_at_risk[i])
                surv_func[i] = est_surv
                surv_var_term += n_events[i]/ ((n_at_risk[i]) * (n_at_risk[i] - n_events[i])) 
                surv_variance[i] = np.square(surv_func[i]) * surv_var_term
                surv_se[i] = np.sqrt(surv_variance[i])

            return surv_func, surv_variance, surv_se
        except Exception as e: 
            print("Failed estimating Kaplan-Meier:  {}".format(e))

    def nelsonaalen(self, timepoints, n_at_risk, n_events, variance='greenwood'): 
        """
        Calculates the cumulativehazard function & its associated variance using the Nelson-Aalen method. 
        Note that currently the variance calculation supported is the Greenwood's method

        Args: 
        timepoints(array-like int/float): 
        n_at_risk(array-like int): 
        n_event(array-like int):
        variance(str): 

        Returns: 
        cumhaz(array-like float): 
        cumhaz_var(array-like float):
        """
        try: 
            cumhaz = np.zeros(len(timepoints))
            cumhaz_var = np.zeros(len(timepoints))
            cumhaz_se = np.zeros(len(timepoints))

            est_cumhaz = 0
            cumhaz_var_term = 0

            for i in range(1, len(timepoints)): 
                est_cumhaz += n_events[i]/n_at_risk[i]
                cumhaz[i] = est_cumhaz
                cumhaz_var_term += n_events[i]/ (n_at_risk[i] * n_at_risk[i])
                cumhaz_var[i] = cumhaz_var_term
                cumhaz_se[i] = np.sqrt(cumhaz_var_term)

            return cumhaz, cumhaz_var, cumhaz_se
        except Exception as e: 
            print("Failed estimating using Nelson-Aalen's method: {}".format(e))

    def plot_survival_function(x_data, y_data,
                               plot_config = {}, 
                               log_time = False, log_prob = False ): 
        try: 
            plot_config["main"] = {
                "type" : "step", 
                "x_data" : np.asarray(x_data), 
                "y_data" : np.asarray(y_data)
            }
            plot_keys = list(plot_config.keys())
            
            if "multiple" not in plot_keys: 
                plot_config["multiple"] = None
            if "plt_functions" not in plot_keys:
                plot_config["plt_functions"] = {
                    "title" : "Survival function", 
                    "ylabel" : "Probability of survival",
                    "xlabel" : "Time"
                }
            
            if log_time is True: 
                x_data_log = np.log(x_data)
                plot_config["plt_functions"]["xlabel"] = "Log time"
                plot_config["main"]["x_data"] = x_data_log
            if log_prob is True: 
                y_data_log = np.log(y_data)
                plot_config["plt_functions"]["ylabel"] = "Log probability of survival"
                plot_config["main"]["y_data"] = y_data_log

            plotCustomiser(plot_config)

        except Exception as e: 
            print("Failed to plot survival function : {}".format(e))

        def survival_prob_time(p): 
            try: 
                assert isinstance(p, float) | isinstance(p, list)

                self.get_fits()

                # surv_probability = estimator.surv_prob
                if self.surv_prob[-1] > p: 
                    print("Survival function estimated does not go below the specified probability")
                    print("Tail correction method will be used to extrapolate (TO BE IMPLEMENTED")
                    
                    return np.inf
                else: 
                    idx_p = np.shape(estimator.surv_prob)[0] - np.searchsorted(np.flip(estimator.surv_prob), 0.75, side = 'left')
                    time_p = self.timepoints[idx_p]
                    
                    return time_p

            except AssertionError as AssError2:
                print("{}. Ensure that the quantile is a float between 0 and 1".format(AssError2))
            except Exception as e: 
                print("Failed to get time for survival probability: {}".format(e))

        def probability_at(times, tail_method = 'gill', interpolation_method = 'numpy', interpolation_argdict = None): 
            try: 
                self.get_fits()

                if tail_method == 'gill': 
                    prob_beyond_tmax = self.surv_func[-1]
                elif tail_method == 'efron': 
                    prob_beyond_tmax = 0
                else: 
                    raise ValueError

                if isinstance(times, int) | isinstance(times, float): 
                    if times > self.timepoints[-1]: 
                        return prob_beyond_tmax
                    else: 
                        if interpolation_method == 'numpy': 
                            return np.interp(times, self.timepoitns, self.surv_function)
                        elif interpolation_method == 'scipy.interp1d': 
                            from scipy import interpolate
                            
                            interp_function = interpolate.interp1d(self.timepoints, self.surv_function, interpolation_argdict)
                            return interp_function(times)

                elif isinstance(times, list) | len(times) > 1: 
                    prob = np.ones(len(times))
                    t_beyond_tmax_idx = np.where(np.asarray(times) > self.timepoints[-1])
                    t_below_tmax_idx = np.where(np.asarray(times) <= self.timepoints[-1])
                    interp_mask = np.full(len(times), True, dtype = bool)
                    interp_mask[t_beyond_tmax_idx] = False
                    t_interpolate = times[interp_mask]   

                    for arr_index, time_i in zip(t_below_tmax_idx, t_interpolate):
                        if interpolation_method == 'numpy': 
                            prob[arr_index] = np.interp(time_i, self.timepoitns, self.surv_function)
                        elif interpolation_method == 'scipy.interp1d': 
                            from scipy import interpolate
                            
                            interp_function = interpolate.interp1d(self.timepoints, self.surv_function, **interpolation_argdict)
                            prob[arr_index] = interp_function(time_i)

                            
                    for arr_index, times in zip(t_beyond_tmax_idx, times[~interp_mask]): 
                        prob[arr_index] = prob_beyond_tmax

                    return prob

                else:
                    print("Format not supported. Please check if the times provided are either in a list, array, or are singular floats/int")
                    raise ValueError

            except Exception as e: 
                print("Failed to get the probability at timepoints provided due to: {}".format(e))





