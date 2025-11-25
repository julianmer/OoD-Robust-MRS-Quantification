####################################################################################################
#                                         plotFunctions.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 09/05/22                                                                                #
#                                                                                                  #
# Purpose: Defines functions to visualize results.                                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import matplotlib.colors as clt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, binned_statistic
from scipy.stats import binned_statistic

# own
from utils.auxiliary import delta, getCS


#**************#
#   bar plot   #
#**************#
def plotBars(error, xLables, yLabel='Error', label='', name='', color='tab:blue'):
    """
        Plots a bar pot with the mean and std error for each x-label.

        @param error -- The error to be represented as a bar.
        @param xLables -- The metabolite names.
        @params yLabel -- label of the y axis (default 'Error').
        @params label -- Name of the model (default '').
        @params name -- Title of the plot (default '').
    """
    fig, ax = plt.figure(figsize=(len(xLables) // 2, 4)), plt.gca()

    ax.bar(xLables, np.mean(error, axis=0), yerr=np.std(error, axis=0),
              color=color, align='center', alpha=0.7, ecolor='black', label=label, capsize=5)

    ax.plot(xLables, np.full(len(xLables), np.mean(error)), color=color, linestyle='--')

    ax.set_ylabel(yLabel)
    ax.set_xticklabels([elem.split()[0] for elem in xLables])
    ax.yaxis.grid(True)

    plt.legend()
    plt.title(name)

    plt.tight_layout()


#***************************#
#   bar plot of 2 metrics   #
#***************************#
def plotBars2Metrics(mapeM1, mapeM2, xLables, yLabel='Error', label1='', label2='', name='',
                     color1='tab:blue', color2='tab:green', alpha=1.0):
    """
        Plots a bar pot with the mean and std MAPE for each metabolite for 2 models.

        @param mapeM1 -- MAPE of model 1.
        @param mapeM2 -- MAPE of model 2.
        @param xLables -- The metabolite names.
        @params yLabel -- label of the y axis (default 'Error').
        @params label1 -- Name of the first model (default '').
        @params label2 -- Name of the second model (default '').
        @params name -- Title of the plot (default '').
        @params color1 -- Color of the first model (default 'tab:blue').
        @params color2 -- Color of the second model (default 'tab:green').
        @params alpha -- Alpha value of the bars (default 1.0).
    """
    fig, ax = plt.subplots(figsize=(len(xLables), 5))
    xPos = np.arange(mapeM1.shape[1])

    shift = 0.4
    ax.bar(xPos - shift / 2, np.mean(mapeM1, axis=0), shift, yerr=np.std(mapeM1, axis=0),
           color=color1, align='center', ecolor='black', capsize=5, label=label1, alpha=alpha)
    ax.bar(xPos + shift / 2, np.mean(mapeM2, axis=0), shift, yerr=np.std(mapeM2, axis=0),
           color=color2, align='center', ecolor='black', capsize=5, label=label2, alpha=alpha)

    # xPosL = xPos / mapeM1.shape[1] * (mapeM1.shape[1] + 2 * shift) - shift
    # ax.plot(xPosL, np.full(xPos.shape, np.mean(mapeM1)), color=color1, linestyle='--', alpha=alpha)
    # ax.plot(xPosL, np.full(xPos.shape, np.mean(mapeM2)), color=color2, linestyle='--', alpha=alpha)

    ax.set_ylabel(yLabel)
    ax.set_xticks(xPos)
    ax.set_xticklabels([elem.split()[0] for elem in xLables])
    ax.yaxis.grid(True)

    plt.legend()
    plt.title(name)
    # plt.tight_layout()


#******************#
#   plot scatter   #
#******************#
def plotScatter(x, y, c=None, xLabel='x', yLabel='y', cLabel='', name='',
                xLog=False, yLog=False, unc=False, simple=False):
    """
        Plots a scatter plot of x and y with regression line.

        @param x -- The x values.
        @param y -- The y values.
        @params c -- The color values (default None).
        @params xLabel -- label of the x axis (default 'x').
        @params yLabel -- label of the y axis (default 'y').
        @params cLabel -- label of the colorbar (default '').
        @params name -- Title of the plot (default '').
        @params xLog -- If True, the x axis is logarithmic (default False).
        @params yLog -- If True, the y axis is logarithmic (default False).
    """
    # scatter plot of true vs estimated concentrations
    plt.figure()
    if c is not None:
        # if unc:
        #     c = 1/2 * (np.log(y ** 2 + np.finfo(np.float32).eps) + \
        #                           x ** 2 / (y ** 2 + np.finfo(np.float32).eps))
        #     cLabel = '$L(e, \sigma)$'
        plt.scatter(x, y, c=c, cmap='viridis', s=10)

        # add label for colorbar
        cbar = plt.colorbar()
        cbar.set_label(cLabel)

    else: plt.scatter(x, y, s=10)

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(name)

    if not simple:
        # plot the x = y line dashed
        plt.plot([np.min(x), np.max(x)], [np.min(x), np.max(x)], 'k--', label='x = y')

        # plot line through lower percentile of data
        if unc:
            from utils.avail import fit_lower_quantile
            a, b = fit_lower_quantile(x, y, 0.05)

        else:
            # plot a line y = ax + b by fitting a linear regression
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            x, y = x[..., np.newaxis], y[..., np.newaxis]
            model.fit(x, y)

            # calculate R-squared of regression model
            r_squared = model.score(x, y)
            a, b = model.coef_.squeeze(), model.intercept_.squeeze()

            plt.text(0.65, 1.025, f'$R^2$ = {r_squared:.2f}', transform=plt.gca().transAxes)

        plt.plot([np.min(x), np.max(x)], [a * np.min(x) + b, a * np.max(x) + b],
                 color='yellowgreen', linestyle='--', label='y = ax + b')
        plt.text(0.0, 1.025, f'a = {a:.2f}', transform=plt.gca().transAxes)
        plt.text(0.2, 1.025, f'b = {b:.2f}', transform=plt.gca().transAxes)

        rmse = np.sqrt(np.mean((x - y) ** 2))
        plt.text(0.85, 1.025, f'$\sigma$ = {rmse:.2f}', transform=plt.gca().transAxes)

        plt.legend()

    # plt.xlim([np.min(x), np.max(x)])
    # plt.ylim([np.min(x), np.max(x)])

    # log scales
    if xLog: plt.xscale('log')
    if yLog: plt.yscale('log')


#*****************************#
#   plot all in one scatter   #
#*****************************#
def plotAllInOneScatter(x, y, lables, c=None, cmin=None, cmax=None, xLabel='x', yLabel='y',
                        cLabel='', xplot=5, yplot=5, name='', xLog=False, yLog=False):
    """
        Plots a scatter plot of x and y for all idx in one plot.

        @param x -- The x values.
        @param y -- The y values.
        @param lables -- The lables for each idx.
        @params c -- The color values (default None).
        @params cmin -- The minimum value of the color (default None).
        @params cmax -- The maximum value of the color (default None).
        @params xLabel -- label of the x axis (default 'x').
        @params yLabel -- label of the y axis (default 'y').
        @params cLabel -- label of the colorbar (default '').
        @params xplot -- Number of plots in x direction (default 5).
        @params yplot -- Number of plots in y direction (default 5).
        @params name -- Title of the plot (default '').
        @params xLog -- If True, the x axis is logarithmic (default False).
        @params yLog -- If True, the y axis is logarithmic (default False).
    """
    # plot all together using subplots
    fig, axs = plt.subplots(xplot, yplot, figsize=(2.5 * xplot, 3.5 * yplot))
    fig.tight_layout(h_pad=2.25)

    for idx, ax in enumerate(axs.flat):
        if idx < len(lables):
            # scatter plot of true vs estimated concentrations
            if c is not None:
                sca = ax.scatter(x[:, idx], y[:, idx], c=c, vmin=cmin, vmax=cmax, cmap='viridis', s=5)
            else:
                sca = ax.scatter(x[:, idx], y[:, idx], s=5)

            # add labels
            if idx >= len(lables) - yplot: ax.set_xlabel(xLabel)
            if idx % yplot == 0: ax.set_ylabel(yLabel)
            ax.set_title(lables[idx])

            # # also compute correlation coefficient r and p-values
            # r, p_val = pearsonr(x[:, idx], y[:, idx])
            # ax.text(0.675, 1.025, f'r = {r:.3f}', transform=ax.transAxes)
            # # ax.text(0.85, 1.025, f'p = {p_val:.3f}', transform=ax.transAxes)
            #
            # # add linear regression line
            # if not (xLog or yLog):
            #     try:
            #         p = np.polyfit(x[:, idx], y[:, idx], 1)
            #         ax.plot(x[:, idx], np.polyval(p, x[:, idx]), 'k--')
            #     except:
            #         print(f'Error in polyfit for {lables[idx]}')

            # plot the x = y line dashed
            ax.plot([np.min(x[:, idx]), np.max(x[:, idx])], [np.min(x[:, idx]), np.max(x[:, idx])],
                    'k--', label='x = y', linewidth=1)

            # plot a line y = ax + b by fitting a linear regression
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()

            # ignore nan values
            mask = ~np.isnan(x[:, idx]) & ~np.isnan(y[:, idx])
            x_i, y_i = x[:, idx][mask], y[:, idx][mask]
            x_i, y_i = x_i[..., np.newaxis], y_i[..., np.newaxis]

            model.fit(x_i, y_i)

            # calculate R-squared of regression model
            r_squared = model.score(x_i, y_i)
            a, b = model.coef_.squeeze(), model.intercept_.squeeze()
            rmse = np.sqrt(np.mean((x_i - y_i) ** 2))

            ax.plot([np.min(x_i), np.max(x_i)], [a * np.min(x_i) + b, a * np.max(x_i) + b],
                    color='yellowgreen', linestyle='--', label='y = ax + b', linewidth=1)

            # left side: α and β
            ax.text(-0.025, 1.03, f'$\\alpha$={a:.2f}, $\\beta$={b:.2f}',
                    transform=ax.transAxes, ha='left', va='bottom', fontsize=9)

            # right side: R² and σ
            ax.text(1.03, 1.03, f'$R^2$={r_squared:.2f}, $\\sigma$={rmse:.2f}',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=9)

            # log scales
            if xLog: ax.set_xscale('log')
            if yLog: ax.set_yscale('log')
        else:
            ax.set_axis_off()  # hide unused subplots

    if c is not None:
        # get the minimum and maximum values from the color data
        c_min = np.min(c)
        c_max = np.max(c)

        # normalize the color values to 0-1 range
        norm = clt.Normalize(c_min, c_max)

        # create a colorbar using the normalized values
        sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
        sm.set_array([])  # Empty array since colorbar doesn't use actual data

        # # add a global colorbar
        # cax = fig.add_axes([1.0, 0.04, 0.015, 0.93])
        # fig.colorbar(sm, cax=cax, location='right', label=cLabel)


#***************************#
#   plot parameter ranges   #
#***************************#
def plotParamRanges(param, losses, xLabel, yLabel, c=None, cmin=None, cmax=None, uncs=None,
                    minR=None, maxR=None, bins=20, name=None, xLog=False, yLog=False, visR=True,
                    mode='sim'):
    """
    Plots the parameter ranges including mean and limits.

    @param param -- The parameter values.
    @param losses -- The loss values.
    @param xLabel -- The label of the x axis.
    @param yLabel -- The label of the y axis.
    @param c -- The color values (default None).
    @param cmin -- The minimum value of the color (default None).
    @param cmax -- The maximum value of the color (default None).
    @param uncs -- The uncertainties of the parameter values (default None).
    @param minR -- The minimum range of the parameter.
    @param maxR -- The maximum range of the parameter.
    @param bins -- The number of bins (default 20).
    @param name -- The name of the plot (default None).
    @param xLog -- If True, the x axis is logarithmic (default False).
    @param yLog -- If True, the y axis is logarithmic (default False).
    @param visR -- If True, the ranges are visualized (default True).
    @param mode -- The mode of the plot limits (default 'sim').
    """
    plt.figure(figsize=(3, 2.8))

    # define hard limits (to unify the plots)
    if mode == 'sim':
        if 'baseline' in xLabel.lower(): xmin, xmax, ymin, ymax = 700, 2500, None, 1.7
        elif 'frequency' in xLabel.lower(): xmin, xmax, ymin, ymax = None, None, None, 1.7
        elif 'gaussian' in xLabel.lower(): xmin, xmax, ymin, ymax = None, None, None, 3.1
        elif 'lorentzian' in xLabel.lower(): xmin, xmax, ymin, ymax = None, None, None, 3.1
        elif 'voigt' in xLabel.lower(): xmin, xmax, ymin, ymax = 4, None, None, 2.4
        elif 'phase' in xLabel.lower(): xmin, xmax, ymin, ymax = None, None, None, 1.7
        elif 'random' in xLabel.lower(): xmin, xmax, ymin, ymax = 1e3, 5e4, None, 1.7
        elif 'snr' in xLabel.lower(): xmin, xmax, ymin, ymax = -22, 62, None, 3.8
        elif 'MM' in xLabel: xmin, xmax, ymin, ymax = None, None, None, 1.7
        elif 'concentration' in xLabel.lower(): xmin, xmax, ymin, ymax = None, None, None, 2.4
        else: xmin, xmax, ymin, ymax = None, None, None, None
    elif mode == 'real':
        if 'baseline' in xLabel.lower(): xmin, xmax, ymin, ymax = 0.2, 0.45, None, 1.4
        elif 'frequency' in xLabel.lower(): xmin, xmax, ymin, ymax = -23, None, None, 1.4
        elif 'gaussian' in xLabel.lower(): xmin, xmax, ymin, ymax = None, None, None, 1.4
        elif 'lorentzian' in xLabel.lower(): xmin, xmax, ymin, ymax = None, None, None, 1.4
        elif 'voigt' in xLabel.lower(): xmin, xmax, ymin, ymax = 18, 33, None, 1.4
        elif 'phase' in xLabel.lower(): xmin, xmax, ymin, ymax = 0, 0.75, None, 1.4
        elif 'random' in xLabel.lower(): xmin, xmax, ymin, ymax = -1e-5, 5e4, None, 1.4
        elif 'snr' in xLabel.lower(): xmin, xmax, ymin, ymax = 12, 17 , None, 1.4
        elif 'MM' in xLabel: xmin, xmax, ymin, ymax = None, None, None, 1.4
        elif 'concentration' in xLabel.lower(): xmin, xmax, ymin, ymax = None, None, None, None
        else: xmin, xmax, ymin, ymax = None, None, None, None
    else:
        # default limits
        xmin, xmax, ymin, ymax = None, None, None, None
    
    # bin and calculate mean in each bin
    bin_means, bin_edges, _ = binned_statistic(param, losses, statistic='mean', bins=bins)

    # calculate bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # interpolate over empty bins (NaNs)
    bin_means_series = pd.Series(bin_means)
    bin_means_interp = bin_means_series.interpolate(method='linear').fillna(method='bfill').fillna(
        method='ffill').values
    bin_means = bin_centers
    y_bin_means = bin_means_interp

    # create color map from distance from mean
    colors = np.abs([y_val - np.interp(x_val, bin_means, y_bin_means)
                     for x_val, y_val in zip(param, losses)])

    # plot
    # plt.scatter(param, losses, c='#3a5e8cFF', cmap='viridis', s=5, vmin=0, vmax=5)

    # plot scatter with good color (single but matching viridis colormap)
    if c is not None:
        norm = clt.Normalize(vmin=cmin if cmin is not None else np.min(c),
                             vmax=cmax if cmax is not None else np.max(c))
        cmap = plt.get_cmap('viridis')
        plt.scatter(param, losses, c=c, cmap=cmap, s=9, vmin=cmin, vmax=cmax, alpha=0.5, edgecolors='none')
    else:
        # color  #3a5e8cFF
        plt.scatter(param, losses, c='#44087D', s=9, alpha=0.5, edgecolors='none')

    # get current y-limits from the active axis
    ax = plt.gca()
    if xmin is None: xmin, _ = ax.get_xlim()
    if xmax is None: _, xmax = ax.get_xlim()
    if ymin is None: ymin, _ = ax.get_ylim()
    if ymax is None: _, ymax = ax.get_ylim()

    # also add vertical line at min max of ranges
    if visR and minR is not None and maxR is not None:
        # define OoD region color and line style
        ood_color = '#ffcccc'  # light red
        ood_line_color = '#d62728'  # strong red
        ood_line_style = (0, (1, 2))  # dotted

        # draw vertical lines at minR and maxR spanning the visible y-range
        ax.plot([minR, minR], [ymin, ymax], color=ood_line_color, linestyle=ood_line_style, linewidth=1)
        ax.plot([maxR, maxR], [ymin, ymax], color=ood_line_color, linestyle=ood_line_style, linewidth=1)

        # # add shading for out-of-range regions
        # ax.fill_between([np.min(param), minR], ymin, ymax, color=ood_color, alpha=0.5, zorder=0, linewidth=0.0)
        # ax.fill_between([maxR, np.max(param)], ymin, ymax, color=ood_color, alpha=0.5, zorder=0, linewidth=0.0)

        plt.fill_betweenx([ymin, ymax], xmin, minR, color=ood_color, alpha=0.5, zorder=0, linewidth=0)
        plt.fill_betweenx([ymin, ymax], maxR, xmax, color=ood_color, alpha=0.5, zorder=0, linewidth=0)

    # plot the mean and standard deviation for each bin
    plt.plot(bin_means, y_bin_means, color='black', linewidth=1, linestyle='--', label='Mean')

    # add uncertainties
    if uncs is not None:
        bin_means = np.linspace(np.min(param), np.max(param), bins + 1)
        bin_width = bin_means[1] - bin_means[0]

        # calculate the mean y-value for each bin
        unc_bin_means = [np.mean(uncs[(param >= bin_start) & (param < bin_start + bin_width)])
                            for bin_start in bin_means]

        # remove second last element and add to last element
        unc_bin_means[-1] += unc_bin_means[-2]
        unc_bin_means = unc_bin_means[:-1]
        bin_means = np.delete(bin_means, -2)

        # scale to mean
        unc_bin_means /= np.mean(unc_bin_means)
        unc_bin_means *= np.mean(y_bin_means)

        # plot the mean and standard deviation for each bin
        plt.plot(bin_means, unc_bin_means, color='red', linewidth=1, label='Uncertainty (Scaled)')

    # legend
    # plt.legend(loc='upper right')

    # highlight some indices
    if mode == 'sim':
        if 'baseline' in xLabel.lower(): idx = [33, 11]
        elif 'frequency' in xLabel.lower(): idx = [6, 68]
        elif 'gaussian' in xLabel.lower(): idx = [0, 1]
        elif 'lorentzian' in xLabel.lower(): idx = [0, 1]
        elif 'voigt' in xLabel.lower(): idx = [28, 40]
        elif 'phase' in xLabel.lower(): idx = [4, 26]
        elif 'random' in xLabel.lower(): idx = [78, 173]
        elif 'snr' in xLabel.lower(): idx = [1, 9]
        elif 'mm' in xLabel.lower(): idx = [33, 75]
        else: idx = [0, 1]
        plt.scatter(param[idx], losses[idx], c='orange', s=6, label='Highlighted Points', zorder=10)
    elif mode == 'real':
        if 'baseline' in xLabel.lower(): idx = [157 + 4 * 342, 157]
        elif 'frequency' in xLabel.lower(): idx = [39, + 4 * 342, 39]
        elif 'gaussian' in xLabel.lower(): idx = [0, 1]
        elif 'lorentzian' in xLabel.lower(): idx = [0, 1]
        elif 'voigt' in xLabel.lower(): idx = [22 + 4 * 342, 22]
        elif 'phase' in xLabel.lower(): idx = [60 + 4 * 342, 60]
        elif 'random' in xLabel.lower(): idx = [0, 1]
        elif 'snr' in xLabel.lower(): idx = [64 + 4 * 342, 64]
        elif 'mm' in xLabel.lower(): idx = [172 + 4 * 342, 172]
        else: idx = [0, 1]
        plt.scatter(param[idx], losses[idx], c='orange', s=6, label='Highlighted Points', zorder=10)

    # limits
    ymin = max(ymin, 0.0)  # ensure y-minimum is at least 0

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    # log scales
    if xLog: plt.xscale('symlog')
    if yLog: plt.yscale('symlog')

    # make sure y axis has at least one decimal place
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(name)
    plt.tight_layout()


#****************************************#
#   plot scatterplot and distributions   #
#****************************************#
def scatterHist(x, y, xLabel='', yLabel='', c=None, cmin=None, cmax=None, cLabel='',
                minR=None, maxR=None, bins=100, name=None, simple=False, lines=True,
                colorHist=False):
    """
    Plots a scatter plot with distributions of x and y as marginal histograms.

    @param x -- The x values.
    @param y -- The y values.
    @param xLabel -- The label of the x axis.
    @param yLabel -- The label of the y axis.
    @param c -- The color values (default None).
    @param cmin -- The minimum value of the color (default None).
    @param cmax -- The maximum value of the color (default None).
    @param cLabel -- The label of the colorbar (default '').
    @param minR -- The minimum range of the parameter.
    @param maxR -- The maximum range of the parameter.
    @param bins -- The number of bins of the histograms (default 100).
    @param name -- The name of the plot (default None).
    @param simple -- If True, adds correlation lines and coefficients (default False).
    @param lines -- If True, adds liens for distributions (default True).
    @param colorHist -- If True, colors the histograms according to the color values (default True).
    """
    # start with a square Figure.
    fig = plt.figure(figsize=(3, 3))

    # make sure everything is positive
    x[x < 0] = 0
    y[y < 0] = 0

    # ignore nan values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if c is not None: c = c[mask]

    # filter y on huge outliers (mainly for noise scenario)
    if 'MM' not in name:
        idx = np.where(y < 20)[0]
        x, y = x[idx], y[idx]
        if c is not None: c = c[idx]

    # add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal axes and the main axes in both directions,
    # also adjust the subplot parameters for a square plot
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    # create the axes
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    if c is not None:
        ax.scatter(x, y, c=c, cmap='viridis', s=6, vmin=cmin, vmax=cmax, alpha=0.8, edgecolors='none')
    else:
        ax.scatter(x, y, s=6, alpha=0.8)

    # use provided cmin/cmax or infer from data
    if c is not None and colorHist:
        norm = clt.Normalize(vmin=cmin if cmin is not None else np.min(c),
                         vmax=cmax if cmax is not None else np.max(c))
        cmap = plt.get_cmap('viridis')

        # x histogram (top)
        hx_counts, hx_edges = np.histogram(x, bins=bins)
        for i in range(len(hx_edges) - 1):
            bin_mask = (x >= hx_edges[i]) & (x < hx_edges[i + 1])
            if not np.any(bin_mask): continue
            bin_color_val = np.mean(c[bin_mask])
            color = cmap(norm(bin_color_val))
            ax_histx.bar(hx_edges[i], hx_counts[i],
                         width=hx_edges[i + 1] - hx_edges[i],
                         color=color, edgecolor='none', align='edge')

        # y histogram (right)
        hy_counts, hy_edges = np.histogram(y, bins=bins)
        for i in range(len(hy_edges) - 1):
            bin_mask = (y >= hy_edges[i]) & (y < hy_edges[i + 1])
            if not np.any(bin_mask): continue
            bin_color_val = np.mean(c[bin_mask])
            color = cmap(norm(bin_color_val))
            ax_histy.barh(hy_edges[i], hy_counts[i],
                          height=hy_edges[i + 1] - hy_edges[i],
                          color=color, edgecolor='none', align='edge')
    else:
        hx_counts, hx_edges, _ = ax_histx.hist(x, bins=bins, color='indigo')
        hy_counts, hy_edges, _ = ax_histy.hist(y, bins=bins, color='indigo', orientation='horizontal')

    # set the limits of the histograms but ignore the first bin if it is zero
    # x_zero_bin = np.where((hx_edges[:-1] <= 0) & (hx_edges[1:] > 0))[0]
    # y_zero_bin = np.where((hy_edges[:-1] <= 0) & (hy_edges[1:] > 0))[0]
    #
    # if len(x_zero_bin) > 0:
    #     x_excl = np.delete(hx_counts, x_zero_bin[0])
    #     ax_histx.set_ylim([0, np.max(x_excl)])
    # if len(y_zero_bin) > 0:
    #     y_excl = np.delete(hy_counts, y_zero_bin[0])
    #     ax_histy.set_xlim([0, np.max(y_excl)])

    # adjust the limits of the histograms, ignoring the first 5 bins
    if len(hx_counts) > 5:
        x_excl = hx_counts[5:]
        ax_histx.set_ylim([0, np.max(x_excl)])

    if len(hy_counts) > 5:
        y_excl = hy_counts[5:]
        ax_histy.set_xlim([0, np.max(y_excl)])

    if lines:
        # add horizontal lines for min, max of x
        if minR is None: minR = np.min(x)
        if maxR is None: maxR = np.max(x)
        ax.plot([minR, maxR], [minR, minR], color='grey', linewidth=1)
        ax.plot([minR, maxR], [maxR, maxR], color='grey', linewidth=1)

        # get max of histogram (already adjusted above if needed)
        maxHist = ax_histy.get_xlim()[1]
        ax_histy.plot([0, maxHist], [minR, minR], color='grey', linewidth=1)
        ax_histy.plot([0, maxHist], [maxR, maxR], color='grey', linewidth=1)

    # label axes
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)

    # add correlation lines and coefficients
    if not simple:
        # plot the x = y line dashed
        ax.plot([np.min(x), np.max(x)], [np.min(x), np.max(x)], 'k--', label='x = y', linewidth=1)

        # plot a line y = ax + b by fitting a linear regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()

        # ignore nan values
        mask = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[mask], y[mask]
        x, y = x[..., np.newaxis], y[..., np.newaxis]

        model.fit(x, y)

        # calculate R-squared of regression model
        r_squared = model.score(x, y)
        a, b = model.coef_.squeeze(), model.intercept_.squeeze()

        ax.text(0.0, 1.100, f'$R^2$ = {r_squared:.2f}', transform=plt.gca().transAxes, size=8)

        ax.plot([np.min(x), np.max(x)], [a * np.min(x) + b, a * np.max(x) + b],
                 color='yellowgreen', linestyle='--', label='y = ax + b', linewidth=1)
        ax.text(0.0, 1.250, f'$\\alpha$ = {a:.2f}', transform=plt.gca().transAxes, size=8)
        ax.text(0.0, 1.175, f'$\\beta$ = {b:.2f}', transform=plt.gca().transAxes, size=8)

        rmse = np.sqrt(np.mean((x - y) ** 2))
        ax.text(0.0, 1.025, f'$\sigma$ = {rmse:.2f}', transform=plt.gca().transAxes, size=8)

    elif name is not None:
        ax_histy.text(0.5, 1.15, name, horizontalalignment='center', verticalalignment='center',
                      transform=ax_histy.transAxes, fontsize=12)

    # ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))

    # hide both ticks and labels on histogram axes only
    ax_histx.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_histx.tick_params(axis="y", left=False, labelleft=False)
    ax_histy.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_histy.tick_params(axis="y", left=False, labelleft=False)

    def custom_decimal_format(x, pos):
        integer_part = abs(int(x))
        if integer_part >= 10: return f"{x:.1f}"
        else: return f"{x:.2f}"

    formatter = ticker.FuncFormatter(custom_decimal_format)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    return ax, ax_histx, ax_histy


#****************************************#
#   plot scatterplot and distributions   #
#****************************************#
def scatterHistDensity(x, y, xLabel='', yLabel='', c=None, cLabel='', bins=100, name=None):
    """
    Plots a scatter plot with distributions of x and y as marginal histograms. Adding density for color.

    @param x -- The x values.
    @param y -- The y values.
    @param xLabel -- The label of the x axis.
    @param yLabel -- The label of the y axis.
    @param c -- The color values (default None).
    @param cLabel -- The label of the colorbar (default '').
    @param bins -- The number of bins of the histograms (default 100).
    @param name -- The name of the plot (default None).
    """
    # start with a square Figure.
    fig = plt.figure(figsize=(3, 3))
    # add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal axes and the main axes in both directions,
    # also adjust the subplot parameters for a square plot
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    # create the axes
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # calculate the point density.
    xy = np.vstack([x,y])
    z = np.histogram2d(x, y, bins=bins)[0].T
    z = z.ravel()# / z.max()
    idx = np.argsort(z)
    x, y, z = x[idx], y[idx], z[idx]

    # invert color map
    cmap = plt.cm.viridis
    cmap = cmap.reversed()

    # the scatter plot:
    ax.scatter(x, y, c=z, cmap=cmap, s=5, vmin=0, vmax=10)

    # plot x = y line
    ax.plot([np.min(x), np.max(x)], [np.min(x), np.max(x)], 'k--', linewidth=1)

    # # now determine nice limits by hand:
    # binwidth = 0.25
    # xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    # lim = (int(xymax / binwidth) + 1) * binwidth
    #
    # bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, color='indigo')
    ax_histy.hist(y, bins=bins, color='indigo', orientation='horizontal')

    # add horizontal lines for min, max of x
    ax.plot([np.min(x), np.max(x)], [np.min(x), np.min(x)], color='grey', linewidth=1)
    ax.plot([np.min(x), np.max(x)], [np.max(x), np.max(x)], color='grey', linewidth=1)

    # get max of histogram
    maxHist = np.max(ax_histy.get_xlim())
    ax_histy.plot([np.min(x), maxHist], [np.min(x), np.min(x)], color='grey', linewidth=1)
    ax_histy.plot([np.min(x), maxHist], [np.max(x), np.max(x)], color='grey', linewidth=1)

    # label axes
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)

    # limit y axis
    # if name == 'NAA': yLim = 24
    # elif name == 'GABA': yLim = 16
    # elif name == 'Cr': yLim = 14.5
    # elif name == 'GSH': yLim = 7
    # else: yLim = np.max(y)
    yLim = np.max(y) if np.max(y) > 0 else 1.0

    plt.ylim([np.min(y), min(yLim, np.max(y))])
    ax.axis(xmin=np.min(x), xmax=min(yLim, np.max(x)))
    ax.axis(ymin=np.min(y), ymax=min(yLim, np.max(y)))

    # add name between the two histograms
    if name is not None:
        ax_histy.text(0.5, 1.15, name, horizontalalignment='center', verticalalignment='center',
                      transform=ax_histy.transAxes, fontsize=12)



#*************************************#
#    plot scatter hists in one plot   #
#*************************************#
def plotAllInOneScatterHist(x, y, lables, c=None, xLabel='x', yLabel='y', cLabel='',
                            xplot=5, yplot=5, name='', bins=100):
    """
    @param x -- The x values.
    @param y -- The y values.
    @param lables -- The lables for each idx.
    @params c -- The color values (default None).
    @params xLabel -- label of the x axis (default 'x').
    @params yLabel -- label of the y axis (default 'y').
    @params cLabel -- label of the colorbar (default '').
    @params xplot -- Number of plots in x direction (default 5).
    @params yplot -- Number of plots in y direction (default 5).
    @params name -- Title of the plot (default '').
    @params bins -- The number of bins of the histograms (default 100).
    """
    # plot all together using subplots
    fig, axs = plt.subplots(xplot, yplot, figsize=(5 * yplot, 5 * xplot))
    # fig.tight_layout(pad=3.0)

    if c is not None:
        cmap = plt.get_cmap('viridis')

        # # get the minimum and maximum values from the color data
        # c_min = np.min(c)
        # c_max = np.max(c)
        c_min, c_max = 0, 10

        # normalize the color values to 0-1 range
        norm = clt.Normalize(c_min, c_max)

        # create a colorbar using the normalized values
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    for idx, ax in enumerate(axs.flat):
        if idx < len(lables):
            # add a gridspec with two rows and two columns and a ratio of 1 to 4 between
            # the size of the marginal axes and the main axes in both directions,
            # also adjust the subplot parameters for a square plot
            gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                                  left=0.1, right=0.9, bottom=0.1, top=0.9,
                                  wspace=0.05, hspace=0.05)
            # create the axes
            ax = fig.add_subplot(gs[1, 0])
            ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
            ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

            # no labels
            ax_histx.tick_params(axis="x", labelbottom=False)
            ax_histy.tick_params(axis="y", labelleft=False)

            # the scatter plot:
            ax.scatter(x, y, c=c, cmap='viridis', s=5)

            # # now determine nice limits by hand:
            # binwidth = 0.25
            # xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
            # lim = (int(xymax / binwidth) + 1) * binwidth
            #
            # bins = np.arange(-lim, lim + binwidth, binwidth)
            ax_histx.hist(x, bins=bins, color='indigo')
            ax_histy.hist(y, bins=bins, color='indigo', orientation='horizontal')

            # add horizontal lines for min, max of x
            ax.plot([np.min(x), np.max(x)], [np.min(x), np.min(x)], color='grey', linewidth=1)
            ax.plot([np.min(x), np.max(x)], [np.max(x), np.max(x)], color='grey', linewidth=1)

            # get max of histogram
            maxHist = np.max(ax_histy.get_xlim())
            ax_histy.plot([np.min(x), maxHist], [np.min(x), np.min(x)], color='grey', linewidth=1)
            ax_histy.plot([np.min(x), maxHist], [np.max(x), np.max(x)], color='grey', linewidth=1)

            # label axes
            ax.set_xlabel(xLabel)
            ax.set_ylabel(yLabel)

            # add name between the two histograms
            ax_histy.text(0.5, 1.15, lables[idx], horizontalalignment='center', verticalalignment='center',
                          transform=ax_histy.transAxes, fontsize=12)
        else:
            ax.set_axis_off()

    # add a global colorbar
    cax = fig.add_axes([1.0, 0.04, 0.015, 0.93])
    fig.colorbar(sm, cax=cax, location='right', label=cLabel)



#********************************#
#   plot spectra and residuals   #
#********************************#
def plotResiduals(specPred, specTrue, bw, cf, ppmAxis=None, ppmRange=(0, 6), shift=2.02, name='', figsize=(7, 4)):
    """
        Plots the given spectra and their residuals in ppm scale.

        @param specPred -- The spectrum prediction to plot.
        @param specTrue -- The true spectrum to plot.
        @param bw -- The bandwidth of the
        @param cf -- The central frequency.
        @param ppmAxis -- The ppm axis to plot (default None).
        @param ppmRange -- The ppm range to plot. If tuple, the ppm axis is computed,
                           if list, the indices are used (as [beg, end]).
        @param shift -- The reference is shift (default: 2 (NAA)).
        @params name -- Title of the plot (default '').
        @params figsize-- The size of the plot (default (7, 4)).
    """
    plt.figure(figsize=figsize)
    plt.xticks(range(ppmRange[0], ppmRange[1], 1))

    if ppmAxis is not None:
        ppmAxis = ppmAxis + 4.68   # shift by water

        beg = min(range(len(ppmAxis)), key=lambda i: abs(ppmAxis[i] - ppmRange[0]))
        end = min(range(len(ppmAxis)), key=lambda i: abs(ppmAxis[i] - ppmRange[1]))

        # frequency shift
        specTrue = np.fft.fftshift(specTrue)
        specPred = np.fft.fftshift(specPred)

        plt.plot(ppmAxis[beg:end], specTrue[beg:end], 'k', linewidth=1.0)
        plt.plot(ppmAxis[beg:end], specPred[beg:end], 'r--', linewidth=1.5)
        plt.plot(ppmAxis[beg:end], specTrue[beg:end] - specPred[beg:end] + 1.4,
                 'b--', linewidth=1.0)

        plt.xlabel('Chemical Shift [ppm]')

    else:
        if type(ppmRange) is tuple:
            l = specTrue.shape[0] // 2
            specTrue = np.concatenate((specTrue[l:], specTrue[:l]))
            specPred = np.concatenate((specPred[l:], specPred[:l]))

            numSamples = specTrue.shape[0]
            reference = np.argmax(specTrue) / numSamples * bw

            # compute ppm axis
            cs = np.array([delta(freq / numSamples * bw, reference, cf * 1e6)
                           for freq in range(numSamples)]) + shift

            beg = min(range(len(cs)), key=lambda i: abs(cs[i] - ppmRange[0]))
            end = min(range(len(cs)), key=lambda i: abs(cs[i] - ppmRange[1]))

            plt.plot(cs[beg:end], specTrue[beg:end], 'k', linewidth=1.0)
            plt.plot(cs[beg:end], specPred[beg:end], 'r--', linewidth=1.5)
            plt.plot(cs[beg:end], specTrue[beg:end] - specPred[beg:end] + 1.4,
                     'b--', linewidth=1.0)

            plt.xlabel('Chemical Shift [ppm]')

        elif type(ppmRange) is list:
            beg, end = ppmRange[0], ppmRange[1]
            plt.plot(specTrue[beg:end], 'k', linewidth=1.0)
            plt.plot(specPred[beg:end], 'r--', linewidth=1.5)
            plt.plot(specTrue[beg:end] - specPred[beg:end] + 1.4,
                     'b--', linewidth=1.0)

            plt.xlabel('Frequency [hz]')

    plt.legend(['Spectrum', 'Prediction', 'Residuals'], loc=2)
    plt.title(name)
    plt.gca().invert_xaxis()


