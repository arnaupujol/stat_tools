import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def poisson_regression(x, y, verbose = False, show = True, c = None, ls = '--', lw = 2):
    """
    This method applies a Poisson regression model.

    Parameters:
    -----------
    x: np.array
        Array of x values.
    y: np.array
        Array of y values.
    verbose: bool
        It specifies if information is printed out.
    show: bool
        It specifies if plots are shown.
    c: str or RGB value
        The colour of the plot line.
    ls: str
        Linestyle of the plot.
    lw: str
        Linewidth of the plot.

    Returns:
    --------
    result_Poisson: statsmodels GLM model
        GLM model obtained
    """
    x_glm = sm.add_constant(x)
    result_Poisson=sm.GLM(y, x_glm, family=sm.families.Poisson()).fit()
    if verbose:
        print(result_Poisson.summary())
    if show:
        x_plot = np.linspace(np.min(x), np.max(x),100)
        x_plot_pred = sm.add_constant(x_plot)
        plt.plot(x_plot, result_Poisson.predict(x_plot_pred), c = c, linestyle = ls, lw = lw)
    return result_Poisson
