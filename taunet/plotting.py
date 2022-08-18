"""
Module containing functions for plotting. 

Authors: Miles Cochran-Branson, Quentin Buat
Date: Summer 2022
"""

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import os
import pickle
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman', 'Times']

from . import log; log = log.getChild(__name__)
from taunet.computation import chi_squared

# set size of figure for all plots
size = (4,4)

def nn_history(file, plotSaveLoc):
    """Plot loss, mse, and mae as a function of epoch

    Parameters:
    ----------

    file : .pickle file
        Contains history from learning

    plotSaveLoc : str
        Path to where plots should be saved
    """

    log.info('Plotting NN history info')
    data = pickle.load(open(file, "rb"))
    for k in data.keys():
        if 'val' in k:
            continue
        metric = k
        fig = plt.figure(figsize=size, dpi = 300)
        plt.plot(data[metric])
        plt.plot(data['val_' + metric])
        plt.ylabel(metric, loc='top')
        plt.xlabel('epoch', loc='right')
        plt.legend(['train', 'val'], loc='upper right')
        fig.savefig(os.path.join(plotSaveLoc, 'plots/nn_model_{}.pdf'.format(metric)), bbox_inches='tight')
        plt.close(fig)


def pt_lineshape(testing_data, plotSaveLoc, nbins=200, target_normalize_var='TauJetsAuxDyn.ptCombined'):
    """Plot lineshape with respect to truth, network result, combined, and final

    Parameters:
    ----------

    testing_data : numpy array of arrays
        Must contain at least variables truthPtVisDressed, ptCombined,
        ptFinalCalib, regressed_target

    plotSaveLoc : str
        Path to where plots should be saved  

    nbins=200 : int
        Number of bins for the histogram

    target_normalize_var='TauJetsAuxDyn.ptCombined' : str
        Variable used in ratio of target
    """

    log.info('Plotting the transverse momenta on the full dataset')
    truth = testing_data['TauJetsAuxDyn.truthPtVisDressed'] / 1000.
    combined = testing_data['TauJetsAuxDyn.ptCombined'] / 1000.
    final = testing_data['TauJetsAuxDyn.ptFinalCalib'] / 1000.
    regressed_target = testing_data['regressed_target'] * testing_data[target_normalize_var] / 1000.
    fig, (ax1, ax2) = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [3,1]}, figsize=(5,6), dpi=100)
    ax1.ticklabel_format(axis='y',style='sci',scilimits=(-3,3), useMathText=True)
    ax1.sharex(ax2)
    fig.subplots_adjust(hspace=0)
    counts_t, bins_t, bars_t = ax1.hist(
        truth,
        bins=nbins,
        range=(0, 200), 
        histtype='stepfilled',
        color='cyan')
    counts_b, bins_b, bars_b = ax1.hist(
        combined,
        bins=nbins,
        range=(0, 200), 
        histtype='step',
        color='black')
    counts_f, bins_f, bars_f = ax1.hist(
        final,
        bins=nbins,
        range=(0, 200), 
        histtype='step',
        color='red')
    counts_ts, bins_ts, bars_ts = ax1.hist(
        regressed_target,
        bins=nbins,
        range=(0, 200), 
        histtype='step',
        color='purple')
    ax1.set_ylabel('Number of $\\tau_{had-vis}$', loc='top')
    ax1.legend(['Truth', 
                'Combined, $\\chi^2$ / dof = {}'.format(round(chi_squared(counts_f, counts_b) / len(counts_t))), 
                'Final, $\\chi^2$ / dof = {}'.format(round(chi_squared(counts_f, counts_t) / len(counts_t))), 
                'This work, $\\chi^2$ / dof = {}'.format(round(chi_squared(counts_ts, counts_t) / len(counts_t)))])

    # get centers of bins and compute distance to each new bin to get errors
    xbins = (bins_t + (bins_t[0] + bins_t[1])/2)[0:len(bins_t)-1]
    x_errors = [(bins_t[0] + bins_t[1])/2 for _ in range(len(bins_t)-1)]

    # compute ratio and error of ratio
    def uncertainty(x, dx, y, dy, z):
        return z * np.sqrt((dx/x)**2 + (dy/y)**2)
    truth_error = np.sqrt(counts_t)
    comb_error = np.sqrt(counts_b)
    fin_error = np.sqrt(counts_f)
    tw_error = np.sqrt(counts_ts)
    ratio_comb = counts_b/counts_t
    ratio_comb_error = uncertainty(counts_b, comb_error, counts_t, truth_error, ratio_comb)
    ratio_fin = counts_f/counts_t
    ratio_fin_error = uncertainty(counts_f, fin_error, counts_t, truth_error, ratio_fin)
    ratio_tw = counts_ts/counts_t
    ratio_tw_error = uncertainty(counts_ts, tw_error, counts_t, truth_error, ratio_tw)

    # plot ratio with errorbars
    ax2.errorbar(xbins, ratio_comb, xerr=x_errors, yerr=ratio_comb_error, color='black', ls='none', marker='.')
    ax2.errorbar(xbins, ratio_fin, xerr=x_errors, yerr=ratio_fin_error, color='red', ls='none', marker='.')
    ax2.errorbar(xbins, ratio_tw, xerr=x_errors, yerr=ratio_tw_error, color='purple', ls='none', marker='.')
    ax2.grid()
    ax2.set_ylim([0.88, 1.12])
    ax2.set_ylabel('Ratio')
    ax2.set_xlabel('$p_{T}(\\tau_{had-vis})$ [GeV]', loc='right')
    
    plt.savefig(os.path.join(plotSaveLoc, 'plots/tes_pt_lineshape.pdf'), bbox_inches='tight')
    plt.close(fig)

def response_lineshape(testing_data, plotSaveLoc, 
            plotSaveName='plots/tes_response_lineshape.pdf', Range=(0,2), scale='log', 
            nbins=200, lineat1=False, target_normalize_var='TauJetsAuxDyn.ptCombined'):
    """Plot the response lineshape

    Parameters:
    ----------

    testing_data : numpy array of arrays
        Must contain at least variables truthPtVisDressed, ptCombined,
        ptFinalCalib, regressed_target

    plotSaveLoc : str
        Path to where plots should be saved  

    plotSaveName='plots/tes_response_lineshape.pdf' : str
        Name of plot to be saved

    Range=(0,2) : tuple
        Ratio (x-axis) range

    scale='log' : string
        Scale of y-axis. Options include 'log' and 'linear'

    nbins=200 : int
        Number of bins to be used

    lineat1=False : bool
        Include a dashed grey line at centered at 1 on the x-axis

    target_normalize_var='TauJetsAuxDyn.ptCombined' : str
        Variable used in ratio of target
    """

    log.info('Plotting the response lineshape on the dataset')
    fig = plt.figure(figsize=size, dpi = 300)
    plt.ticklabel_format(axis='y',style='sci',scilimits=(4,4), useMathText=True)
    plt.yscale(scale)
    plt.hist(
        testing_data['regressed_target'] * testing_data[target_normalize_var] / testing_data['TauJetsAuxDyn.truthPtVisDressed'],
        bins=nbins, 
        range=Range, 
        histtype='step', 
        color='purple', 
        label='This work')
    plt.hist(
        testing_data['TauJetsAuxDyn.ptFinalCalib'] / testing_data['TauJetsAuxDyn.truthPtVisDressed'],
        bins=nbins, 
        range=Range, 
        histtype='step', 
        color='red', 
        label='Final')
    plt.hist(
        testing_data['TauJetsAuxDyn.ptCombined'] / testing_data['TauJetsAuxDyn.truthPtVisDressed'],
        bins=nbins, 
        range=Range, 
        histtype='step', 
        color='black', 
        label='Combined')
    if lineat1:
        xmin, xmax, ymin, ymax = plt.axis()
        plt.plot([1.0, 1.0], [ymin, ymax], linestyle='dashed', color='grey')
    plt.ylabel('Number of $\\tau_{had-vis}$', loc = 'top')
    plt.xlabel('Predicted $p_{T}(\\tau_{had-vis})$ / True $p_{T}(\\tau_{had-vis})$', loc = 'right')
    plt.legend()
    plt.savefig(os.path.join(plotSaveLoc, plotSaveName), bbox_inches='tight')
    plt.yscale('linear')
    plt.close(fig)
    

def target_lineshape(testing_data, bins=100, range=(0, 10), basename='tes_target_lineshape', logy=True, plotSaveLoc='', target_normalize_var='TauJetsAuxDyn.ptCombined'):
    """Plot the regressed target lineshape

    Parameters:
    ----------

    testing_data : numpy array of arrays
        Must contain at least variables truthPtVisDressed, ptCombined,
        ptFinalCalib, regressed_target

    plotSaveLoc='' : str
        Path to where plots should be saved  

    basename='tes_target_lineshape' " str
        Base name for plot saving

    range=(0,10) : tuple
        Ratio (x-axis) range

    scale='log' : string
        Scale of y-axis. Options include 'log' and 'linear'

    bins=100 : int
        Number of bins to be used

    logy=True : bool
        Make y-scale log

    target_normalize_var='TauJetsAuxDyn.ptCombined' : str
        Variable used in ratio of target
    """

    log.info('Plotting the regressed target lineshape on the dataset')
    fig = plt.figure(figsize=size, dpi = 300)
    if logy:
        plt.yscale('log')
    if not logy:
        plt.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
    counts_t, bins_t, bars_t = plt.hist(
        testing_data['TauJetsAuxDyn.truthPtVisDressed'] / testing_data[target_normalize_var],
        bins=bins, 
        range=range, 
        histtype='stepfilled', 
        color='cyan')
    counts_f, bins_f, bars_f = plt.hist(
        testing_data['TauJetsAuxDyn.ptFinalCalib'] / testing_data[target_normalize_var],
        bins=bins, 
        range=range, 
        histtype='step', 
        color='red')
    counts_m, bins_m, bars_m = plt.hist(
        testing_data['regressed_target'],
        bins=bins, 
        range=range, 
        histtype='step', 
        color='purple')
    plt.ylabel('Number of $\\tau_{had-vis}$', loc = 'top')
    plt.xlabel('Regressed target', loc = 'right')
    if 'ptCombined' in target_normalize_var:
        plt.legend(['Truth / Comb.',
                    'Final / Comb., $\\chi^2$ / dof = {}'.format(round(chi_squared(counts_f, counts_t) / len(counts_f))), 
                    'This work, $\\chi^2$ / dof = {}'.format(round(chi_squared(counts_m, counts_t) / len(counts_f)))])
    else:
        plt.legend(['Truth / $p_T^{Calo}',
            'Final / $p_T^{Calo}, $\\chi^2$ / dof = '+str(round(chi_squared(counts_f, counts_t) / len(counts_f))), 
            'This work, $\\chi^2$ / dof = '+str(round(chi_squared(counts_m, counts_t) / len(counts_f)))])
    plt.savefig(os.path.join(plotSaveLoc, 'plots/{}.pdf'.format(basename)), bbox_inches='tight')
    plt.yscale('linear')
    plt.close(fig)
    

def response_and_resol_vs_var(testing_data, plotSaveLoc, xvar='pt', CL=0.68, nbins=15, pltText='', target_normalize_var='TauJetsAuxDyn.ptCombined'):
    """Plot response and resolution as a function of a given variable

    Parameters:
    ----------

    testing_data : numpy array of arrays
        Must contain at least variables truthPtVisDressed, ptCombined,
        ptFinalCalib, regressed_target

    plotSaveLoc : str
        Path to where plots should be saved  

    xvar='pt' : str
        Takes variables 'pt', 'eta', 'mu'

    CL=0.68 : float
        Confidence level at which to compute variable/truth distribution

    nbins=15 : int
        Number of bins for plotting

    pltText='' : str
        Optionally add text to plot. Supply string with desired text

    target_normalize_var='TauJetsAuxDyn.ptCombined' : str
        Variable used in ratio of target
    """

    log.info('plotting the response and resolution versus {}'.format(xvar))
    from .utils import response_curve, makeBins

    response_reg = testing_data['regressed_target'] * testing_data[target_normalize_var] / testing_data['TauJetsAuxDyn.truthPtVisDressed']
    response_ref = testing_data['TauJetsAuxDyn.ptFinalCalib'] / testing_data['TauJetsAuxDyn.truthPtVisDressed']
    response_comb = testing_data['TauJetsAuxDyn.ptCombined'] / testing_data['TauJetsAuxDyn.truthPtVisDressed']
    if xvar=='pt':
        truth = testing_data['TauJetsAuxDyn.truthPtVisDressed'] / 1000. 
        bins = makeBins(10, 200, nbins)
        xlab = 'True $p_{T}(\\tau_{had-vis})$ [GeV]'
        bins_reg, bin_errors_reg, means_reg, errs_reg, resol_reg = response_curve(response_reg, truth, bins, cl=CL)
        bins_ref, bin_errors_ref, means_ref, errs_ref, resol_ref = response_curve(response_ref, truth, bins, cl=CL)
        bins_comb, bin_errors_comb, means_comb, errs_comb, resol_comb = response_curve(response_comb, truth, bins, cl=CL)
        pltTextLoc = (125, 17)
    elif xvar=='eta':
        truth = testing_data['TauJetsAuxDyn.truthEtaVisDressed']
        bins = makeBins(-2.5, 2.5, nbins)
        xlab = 'True $\\eta (\\tau_{had-vis})$'
        bins_reg, bin_errors_reg, means_reg, errs_reg, resol_reg = response_curve(response_reg, truth, bins, cl=CL)
        bins_ref, bin_errors_ref, means_ref, errs_ref, resol_ref = response_curve(response_ref, truth, bins, cl=CL)
        bins_comb, bin_errors_comb, means_comb, errs_comb, resol_comb = response_curve(response_comb, truth, bins, cl=CL)
        pltTextLoc = (-1, 14.5)
    elif xvar=='mu':
        truth = testing_data['TauJetsAuxDyn.mu']
        bins = makeBins(0, 80, nbins)
        xlab = 'Average interaction per bunch crossing'
        bins_reg, bin_errors_reg, means_reg, errs_reg, resol_reg = response_curve(response_reg, truth, bins, cl=CL)
        bins_ref, bin_errors_ref, means_ref, errs_ref, resol_ref = response_curve(response_ref, truth, bins, cl=CL)
        bins_comb, bin_errors_comb, means_comb, errs_comb, resol_comb = response_curve(response_comb, truth, bins, cl=CL)
        pltTextLoc = (2, 13.75)
    else:
        raise ValueError('Possible variables are pt, eta, mu')

    fig = plt.figure(figsize=size, dpi = 100)
    plt.ticklabel_format(axis='y',style='sci', scilimits=(-3,3))
    plt.errorbar(bins_comb, means_comb, errs_comb, bin_errors_comb, fmt='o', color='black', label='Combined')
    plt.errorbar(bins_ref, means_ref, errs_ref, bin_errors_ref, fmt='o', color='red', label='Final')
    plt.errorbar(bins_reg, means_reg, errs_reg, bin_errors_reg, fmt='o', color='purple', label='This work')
    plt.grid(color='0.95')
    plt.ylabel('Predicted $p_{T}(\\tau_{had-vis})$ / True $p_{T}(\\tau_{had-vis})$', loc = 'top')
    plt.xlabel(xlab, loc = 'right')
    plt.legend()
    plt.savefig(os.path.join(plotSaveLoc, 
        'plots/tes_mdn_response_vs_truth_{}.pdf'.format(xvar)), bbox_inches='tight')
    plt.close(fig) 

    fig = plt.figure(figsize=size, dpi = 100)
    ytitle = '$p_T (\\tau_{had-vis})$ resolution, '+str(round(CL*100))+'\% CL [\%]'
    plt.plot(bins_ref, 100 * resol_ref, color='red', label='Final')
    plt.plot(bins_ref, 100 * resol_comb, color='black', label='Combined')
    plt.plot(bins_ref, 100 * resol_reg, color='purple', label='This work')
    if pltText != '':
        plt.text(pltTextLoc[0], pltTextLoc[1], pltText)
    plt.ylabel(ytitle, loc = 'top')
    plt.xlabel(xlab, loc = 'right')
    plt.legend()
    plt.savefig(os.path.join(plotSaveLoc, 
        'plots/tes_mdn_resolution_vs_truth_{}.pdf'.format(xvar)), bbox_inches='tight')
    plt.close(fig)