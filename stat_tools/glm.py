import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def regression(x, y, family = 'poisson', verbose = False, show = True, \
                       c = None, ls = '--', lw = 2):
    """
    This method applies a Poisson or binomial regression model.

    Parameters:
    -----------
    x: np.array
        Array of x values.
    y: np.array
        Array of y values.
    family: str {'poisson', 'binomial'}
        It specifies the family of the regression.
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
    result_glm: statsmodels GLM model
        GLM model obtained
    """
    x_glm = sm.add_constant(x)
    if family == 'poisson':
        result_glm=sm.GLM(y, x_glm, family=sm.families.Poisson()).fit()
    elif family == 'binomial':
        result_glm=sm.GLM(y, x_glm, family=sm.families.Binomial()).fit()
    if verbose:
        print(result_glm.summary())
    if show:
        x_plot = np.linspace(np.min(x), np.max(x),100)
        x_plot_pred = sm.add_constant(x_plot)
        plt.plot(x_plot, result_glm.predict(x_plot_pred), c = c, linestyle = ls, lw = lw)
    return result_glm
