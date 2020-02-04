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
                self.surv_prob, self.surv_variance = self.kaplanmeier(self.timepoints, self.n_at_risk, self.n_events)
            elif self.type == 'nelson-aalen': 
                self.cumhaz, self.cumhaz_variance = self.nelsonaalen(self.timepoints, self.n_at_risk, self.n_events)

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

        except Exception as e: 
            print("failed helper function-> getting counts per timepoint :{}".format(e))

    def kaplanmeier(self, timepoints, n_at_risk, n_events, variance='greenwood'): 
        """
        Calculates the survival function & its associated variance using the Kaplan-Meier method. 
        Note that currently the variance calculation supported is the Greenwood's method

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
            surv_prob = np.ones(len(timepoints))
            surv_variance = np.zeros(len(timepoints))
            est_surv = 1
            surv_var_term = 1
            for i in range(1, len(timepoints)): 
                est_surv = est_surv * (1 - n_events[i]/n_at_risk[i])
                surv_prob[i] = est_surv
                surv_var_term += n_events[i]/ ((n_at_risk[i]) * (n_at_risk[i] - n_events[i])) 
                surv_variance[i] = np.square(surv_prob[i]) * surv_var_term

            return surv_prob, surv_variance
        except Exception as e: 
            print("Failed estimating Kaplan-Meier due to {}".format(e))

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
            est_cumhaz = 0
            cumhaz_var_term = 0

            for i in range(1, len(timepoints)): 
                est_cumhaz += n_events[i]/n_at_risk[i]
                cumhaz[i] = est_cumhaz
                cumhaz_var_term += n_events[i]/ (n_at_risk[i] * n_at_risk[i])
                cumhaz_var[i] = cumhaz_var_term

            return cumhaz, cumhaz_var
        except Exception as e: 
            print("Failed estimating using Nelson-Aalen's method due to {}".format(e))

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





