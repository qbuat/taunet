import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['text.usetex'] = True

from . import log; log = log.getChild(__name__)

def nn_history(history, metric='loss'):

    fig = plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
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
    


def target_lineshape(testing_data):
    """
    """
    log.info('Plotting the regressed target lineshape on the dataset')
    fig = plt.figure()
    plt.yscale('log')
    plt.hist(
        testing_data['TauJetsAuxDyn.truthPtVisDressed'] / testing_data['TauJetsAuxDyn.ptCombined'],
        bins=200, 
        range=(0, 2), 
        histtype='stepfilled', 
        color='cyan', 
        label='Truth / Combined')
    plt.hist(
        testing_data['regressed_target'],
        bins=200, 
        range=(0, 2), 
        histtype='step', 
        color='purple', 
        label='This work')
    plt.ylabel('Number of $\\tau_{had-vis}$')
    plt.xlabel('Regressed target')
    plt.legend()
    plt.savefig('plots/tes_response_lineshape.pdf')
    plt.yscale('linear')
    plt.close(fig)
    


