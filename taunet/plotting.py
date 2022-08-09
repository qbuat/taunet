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

# global plotting parameters
size = (4,4)
#target_normalize_var = 'TauJetsAuxDyn.ptCombined'
target_normalize_var = 'TauJetsAuxDyn.ptTauEnergyScale'

def nn_history(file, plotSaveLoc):
    log.info('Plotting NN history info')
    #open pickle file stored from training
    data = pickle.load(open(file, "rb"))
    #create plots
    for k in data.keys():
        if 'val' in k:
            continue
        metric = k
        fig = plt.figure(figsize=size, dpi = 300)
        # try to do some scientific notation formatting
        # formatter = ticker.ScalarFormatter(useMathText=True)
        # formatter.set_scientific(True) 
        plt.plot(data[metric])
        plt.plot(data['val_' + metric])
        plt.ylabel(metric, loc='top')
        plt.xlabel('epoch', loc='right')
        plt.legend(['train', 'val'], loc='upper right')
        #fig.axes[0].set_yscale('log')
        fig.savefig(os.path.join(plotSaveLoc, 'plots/nn_model_{}.pdf'.format(metric)), bbox_inches='tight')
        plt.close(fig)


def pt_lineshape(testing_data, plotSaveLoc):
    """
    """
    log.info('Plotting the transverse momenta on the full dataset')
    truth = testing_data['TauJetsAuxDyn.truthPtVisDressed'] / 1000.
    combined = testing_data['TauJetsAuxDyn.ptCombined'] / 1000.
    final = testing_data['TauJetsAuxDyn.ptFinalCalib'] / 1000.
    regressed_target = testing_data['regressed_target'] * testing_data['TauJetsAuxDyn.ptCombined'] / 1000.
    fig, (ax1, ax2) = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [3,1]}, figsize=(5,6), dpi=100)
    ax1.ticklabel_format(axis='y',style='sci',scilimits=(-3,3), useMathText=True)
    ax1.sharex(ax2)
    fig.subplots_adjust(hspace=0)
    counts_t, bins_t, bars_t = ax1.hist(
        truth,
        bins=200,
        range=(0, 200), 
        histtype='stepfilled',
        color='cyan')
    counts_b, bins_b, bars_b = ax1.hist(
        combined,
        bins=200,
        range=(0, 200), 
        histtype='step',
        color='black')
    counts_f, bins_f, bars_f = ax1.hist(
        final,
        bins=200,
        range=(0, 200), 
        histtype='step',
        color='red')
    counts_ts, bins_ts, bars_ts = ax1.hist(
        regressed_target,
        bins=200,
        range=(0, 200), 
        histtype='step',
        color='purple')
    ax1.set_ylabel('Number of $\\tau_{had-vis}$', loc='top')
    ax1.legend(['Truth', 
                'Combined, $\\chi^2 = ${}'.format(round(chi_squared(counts_f, counts_b))), 
                'Final, $\\chi^2 = ${}'.format(round(chi_squared(counts_f, counts_t))), 
                'This work, $\\chi^2 = ${}'.format(round(chi_squared(counts_ts, counts_t)))])

    # make ratio plot
    from .utils import makeBins, response_curve
    bins = makeBins(0, 200, 15)
    bins_reg, bin_errors_reg, means_reg, errs_reg, resol_reg = response_curve(regressed_target/truth, truth, bins)
    bins_ref, bin_errors_ref, means_ref, errs_ref, resol_ref = response_curve(final/truth, truth, bins)
    bins_comb, bin_errors_comb, means_comb, errs_comb, resol_comb = response_curve(combined/truth, truth, bins)
    ax2.errorbar(bins_comb, means_comb, errs_comb, bin_errors_comb, fmt='.', color='black', label='Combined')
    ax2.errorbar(bins_ref, means_ref, errs_ref, bin_errors_ref, fmt='.', color='red', label='Final')
    ax2.errorbar(bins_reg, means_reg, errs_reg, bin_errors_reg, fmt='.', color='purple', label='This work')
    ax2.grid()
    ax2.set_ylabel('Ratio')
    ax2.set_xlabel('$p_{T}(\\tau_{had-vis})$ [GeV]', loc='right')
    
    plt.savefig(os.path.join(plotSaveLoc, 'plots/tes_pt_lineshape.pdf'))
    plt.close(fig)

def response_lineshape(testing_data, plotSaveLoc, 
            plotSaveName='plots/tes_response_lineshape.pdf', Range=(0,2), scale='log', nbins=200, lineat1=False):
    """
    """
    log.info('Plotting the response lineshape on the dataset')
    fig = plt.figure(figsize=size, dpi = 300)
    plt.yscale(scale)
    plt.hist(
        testing_data['regressed_target'] * testing_data['TauJetsAuxDyn.ptCombined'] / testing_data['TauJetsAuxDyn.truthPtVisDressed'],
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
    

def target_lineshape(testing_data, bins=100, range=(0, 10), basename='tes_target_lineshape', logy=True, plotSaveLoc=''):
    """
    """
    log.info('Plotting the regressed target lineshape on the dataset')
    fig = plt.figure(figsize=size, dpi = 300)
    if logy:
        plt.yscale('log')
    if not logy:
        plt.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
    counts_t, bins_t, bars_t = plt.hist(
        testing_data['TauJetsAuxDyn.truthPtVisDressed'] / testing_data['TauJetsAuxDyn.ptCombined'],
        bins=bins, 
        range=range, 
        histtype='stepfilled', 
        color='cyan')
        #label='Truth / Combined')
    counts_f, bins_f, bars_f = plt.hist(
        testing_data['TauJetsAuxDyn.ptFinalCalib'] / testing_data['TauJetsAuxDyn.ptCombined'],
        bins=bins, 
        range=range, 
        histtype='step', 
        color='red')
        #label='Final / Combined')
    counts_m, bins_m, bars_m = plt.hist(
        testing_data['regressed_target'],
        bins=bins, 
        range=range, 
        histtype='step', 
        color='purple')
        #label='This work')
    plt.ylabel('Number of $\\tau_{had-vis}$', loc = 'top')
    plt.xlabel('Regressed target', loc = 'right')
    plt.legend(['Truth / Comb.',
                'Final / Comb., $\\chi^2 = {}$'.format(round(chi_squared(counts_f, counts_t))), 
                'This work, $\\chi^2 = {}$'.format(round(chi_squared(counts_m, counts_t)))])
    plt.savefig(os.path.join(plotSaveLoc, 'plots/{}.pdf'.format(basename)), bbox_inches='tight')
    plt.yscale('linear')
    plt.close(fig)
    

def response_and_resol_vs_var(testing_data, plotSaveLoc, xvar='pt', CL=0.68, nbins=25, pltText=''):
    """
    """
    log.info('plotting the response and resolution versus {}'.format(xvar))
    from .utils import response_curve, makeBins

    response_reg = testing_data['regressed_target'] * testing_data['TauJetsAuxDyn.ptCombined'] / testing_data['TauJetsAuxDyn.truthPtVisDressed']
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

    fig = plt.figure(figsize=size, dpi = 300)
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

    fig = plt.figure(figsize=size, dpi = 300)
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
