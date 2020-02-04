import matplotlib.pyplot as plt 
from inspect import getmembers, isfunction 

class notInPltException(Exception): 
    pass

class plotCustomiser: 
    def __init__(self, plot_config): 
        super(plotCustomiser, self).__init__()
        self.plt_main = plot_config["main"]
        self.plt_config = plot_config["plt_functions"]
        self.plot_multiple = plot_config["multiple"]
        self.plt_function_list = [o[0] for o in getmembers(plt) if isfunction(o[1])]
#         print('Init ok')
        self._plot()
        plt.show()
    
    def _plot(self): 
        if self.plt_main["type"] not in self.plt_function_list: 
            raise notInPltException("Plotting function {} not in matplotlib".format(self.plt_main["type"]))
        else:
            plt_mainEvalStr = "plt.{}({},{}, where = 'post')".format(\
                                                      self.plt_main["type"],
                                                      list(self.plt_main["x_data"]), 
                                                      list(self.plt_main["y_data"])
            )
            try: 
                eval(plt_mainEvalStr)
            except Exception as e:
                print("Couldn't evaluate main statement: {}".format(e))
        
        for config_key, config_value in self.plt_config.items(): 
            # print(config_key, config_value)
            if config_key not in self.plt_function_list: 
                raise notInPltException("Argument {} not in matplotlib. Check spelling".format(config_key))
            else: 
                eval("plt.{}('{}')".format(config_key, config_value))