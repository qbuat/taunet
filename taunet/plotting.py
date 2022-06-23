import matplotlib.pyplot as plt
import numpy as np
import pickle
plt.rcParams['text.usetex'] = True

from . import log; log = log.getChild(__name__)

def nn_history(file):
    #open pickle file stored from training
    data = pickle.load(open(file, "rb"))
    #create plots
    for k in data.keys():
        if 'val' in k:
            continue
        metric = k
        fig = plt.figure()
        plt.plot(data[metric])
        plt.plot(data['val_' + metric])
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        fig.axes[0].set_yscale('log')
        fig.savefig('plots/nn_model_{}.pdf'.format(metric))
        plt.close(fig)


def pt_lineshape(testing_data):
    """
    """
    log.info('Plotting the transverse momenta on the full dataset')
    fig = plt.figure()
    plt.hist(
        testing_data['TauJetsAuxDyn.truthPtVisDressed'] / 1000.,
        bins=200,
        range=(0, 200), 
        histtype='stepfilled',
        color='cyan',
        label='Truth')
    plt.hist(
        testing_data['TauJetsAuxDyn.ptCombined'] / 1000.,
        bins=200,
        range=(0, 200), 
        histtype='step',
        color='black',
        label='Combined')
    plt.hist(
        testing_data['TauJetsAuxDyn.ptFinalCalib'] / 1000.,
        bins=200,
        range=(0, 200), 
        histtype='step',
        color='red',
        label='Final')
    plt.hist(
        testing_data['regressed_target'] * testing_data['TauJetsAuxDyn.ptCombined'] / 1000.,
        bins=200,
        range=(0, 200), 
        histtype='step',
        color='purple',
        label='This work')
    plt.ylabel('Number of $\\tau_{had-vis}$')
    plt.xlabel('$p_{T}(\\tau_{had-vis})$ [GeV]')
    plt.legend()
    plt.savefig('plots/tes_pt_lineshape.pdf')
    plt.close(fig)

def response_lineshape(testing_data):
    """
    """
    log.info('Plotting the response lineshape on the dataset')
    fig = plt.figure()
    plt.yscale('log')
    plt.hist(
        testing_data['regressed_target'] * testing_data['TauJetsAuxDyn.ptCombined'] / testing_data['TauJetsAuxDyn.truthPtVisDressed'],
        bins=200, 
        range=(0, 2), 
        histtype='step', 
        color='purple', 
        label='This work')
    plt.hist(
        testing_data['TauJetsAuxDyn.ptFinalCalib'] / testing_data['TauJetsAuxDyn.truthPtVisDressed'],
        bins=200, 
        range=(0, 2), 
        histtype='step', 
        color='red', 
        label='Final')
    plt.hist(
        testing_data['TauJetsAuxDyn.ptCombined'] / testing_data['TauJetsAuxDyn.truthPtVisDressed'],
        bins=200, 
        range=(0, 2), 
        histtype='step', 
        color='black', 
        label='Combined')
    plt.ylabel('Number of $\\tau_{had-vis}$')
    plt.xlabel('Predicted $p_{T}(\\tau_{had-vis})$ / True $p_{T}(\\tau_{had-vis})$')
    plt.legend()
    plt.savefig('plots/tes_response_lineshape.pdf')
    plt.yscale('linear')
    plt.close(fig)
    


def target_lineshape(testing_data, bins=100, range=(0, 10), basename='tes_target_lineshape', logy=True):
    """
    """
    log.info('Plotting the regressed target lineshape on the dataset')
    fig = plt.figure()
    if logy:
        plt.yscale('log')
    plt.hist(
        testing_data['TauJetsAuxDyn.truthPtVisDressed'] / testing_data['TauJetsAuxDyn.ptCombined'],
        bins=bins, 
        range=range, 
        histtype='stepfilled', 
        color='cyan', 
        label='Truth / Combined')
    plt.hist(
        testing_data['TauJetsAuxDyn.ptFinalCalib'] / testing_data['TauJetsAuxDyn.ptCombined'],
        bins=bins, 
        range=range, 
        histtype='step', 
        color='red', 
        label='Final / Combined')
    plt.hist(
        testing_data['regressed_target'],
        bins=bins, 
        range=range, 
        histtype='step', 
        color='purple', 
        label='This work')
    plt.ylabel('Number of $\\tau_{had-vis}$')
    plt.xlabel('Regressed target')
    plt.legend()
    plt.savefig('plots/{}.pdf'.format(basename))
    plt.yscale('linear')
    plt.close(fig)
    

def response_and_resol_vs_pt(testing_data):
    """
    """
    log.info('plotting the response and resolution versus pt')
    from .utils import response_curve

    response_reg = testing_data['regressed_target'] * testing_data['TauJetsAuxDyn.ptCombined'] / testing_data['TauJetsAuxDyn.truthPtVisDressed']
    response_ref = testing_data['TauJetsAuxDyn.ptFinalCalib'] / testing_data['TauJetsAuxDyn.truthPtVisDressed']
    response_comb = testing_data['TauJetsAuxDyn.ptCombined'] / testing_data['TauJetsAuxDyn.truthPtVisDressed']
    truth_pt = testing_data['TauJetsAuxDyn.truthPtVisDressed'] / 1000. 

    bins = [
        # (0, 10),
        (10, 20),
        (20, 30),
        (30, 40),
        (40, 50),
        (50, 60),
        (60, 70),
        (70, 80),
        (80, 90),
        (90, 100),
        (100, 150),
        (150, 200),
    ]

    bins_reg, bin_errors_reg, means_reg, errs_reg, resol_reg = response_curve(response_reg, truth_pt, bins)
    bins_ref, bin_errors_ref, means_ref, errs_ref, resol_ref = response_curve(response_ref, truth_pt, bins)
    bins_comb, bin_errors_comb, means_comb, errs_comb, resol_comb = response_curve(response_comb, truth_pt, bins)

    fig = plt.figure()
    plt.errorbar(bins_comb, means_comb, errs_comb, bin_errors_comb, fmt='o', color='black', label='Combined')
    plt.errorbar(bins_ref, means_ref, errs_ref, bin_errors_ref, fmt='o', color='red', label='Final')
    plt.errorbar(bins_reg, means_reg, errs_reg, bin_errors_reg, fmt='o', color='purple', label='This work')
    plt.grid(color='0.95')
    plt.ylabel('Predicted $p_{T}(\\tau_{had-vis})$ / True $p_{T}(\\tau_{had-vis})$')
    plt.xlabel('True $p_{T}(\\tau_{had-vis})$ [GeV]')
    plt.legend()
    plt.savefig('plots/tes_mdn_response_vs_truth_pt.pdf')
    plt.close(fig) 

    fig = plt.figure()
    plt.plot(bins_ref, 100 * resol_ref, color='red', label='Final')
    plt.plot(bins_ref, 100 * resol_comb, color='black', label='Combined')
    plt.plot(bins_ref, 100 * resol_reg, color='purple', label='This work')
    plt.ylabel('$p_{T}(\\tau_{had-vis})$ resolution [\%]')
    plt.xlabel('True $p_{T}(\\tau_{had-vis})$ [GeV]')
    plt.legend()
    plt.savefig('plots/tes_mdn_resolution_vs_truth_pt.pdf')
    plt.close(fig)
