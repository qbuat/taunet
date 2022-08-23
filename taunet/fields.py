"""Lists of variables used in fitting and plotting
"""

FEATURES = [
    'TauJetsAuxDyn.mu',
    'TauJetsAuxDyn.nVtxPU', 
    'TauJetsAuxDyn.rho', 
    'TauJetsAuxDyn.ClustersMeanCenterLambda',
    'TauJetsAuxDyn.ClustersMeanFirstEngDens',
    'TauJetsAuxDyn.ClustersMeanSecondLambda',
    'TauJetsAuxDyn.ClustersMeanPresamplerFrac',
    'TauJetsAuxDyn.ClustersMeanEMProbability',
    'TauJetsAuxDyn.ptIntermediateAxisEM/TauJetsAuxDyn.ptIntermediateAxis', 
    'TauJetsAuxDyn.ptCombined',
    'TauJetsAuxDyn.ptPanTauCellBased/TauJetsAuxDyn.ptCombined',
    'TauJetsAuxDyn.ptIntermediateAxis/TauJetsAuxDyn.ptCombined',
    'TauJetsAuxDyn.etaPanTauCellBased',
    'TauJetsAuxDyn.PFOEngRelDiff', 
    'TauJetsAuxDyn.ptTauEnergyScale',
    'TauJetsAuxDyn.NNDecayModeProb_1p0n',
    'TauJetsAuxDyn.NNDecayModeProb_1p1n',
    'TauJetsAuxDyn.NNDecayModeProb_1pXn',
    'TauJetsAuxDyn.NNDecayModeProb_3p0n',
    'TauJetsAuxDyn.NNDecayModeProb_3pXn',
]



TRUTH_FIELDS = [
    'TauJetsAuxDyn.truthPtVisDressed',
    'TauJetsAuxDyn.truthEtaVisDressed',
    'TauJetsAuxDyn.truthPhiVisDressed',
]

OTHER_TES = [
    'TauJetsAuxDyn.ptCombined',
    'TauJetsAuxDyn.ptTauEnergyScale',
    'TauJetsAuxDyn.ptPanTauCellBased',
    'TauJetsAuxDyn.ptFinalCalib',
    'TauJetsAuxDyn.mu',
    'TauJetsAuxDyn.NNDecayMode', 
    'TauJetsAuxDyn.nTracks', 
    'TauJetsAuxDyn.NNDecayModeProb_1p0n',
    'TauJetsAuxDyn.NNDecayModeProb_1p1n',
    'TauJetsAuxDyn.NNDecayModeProb_1pXn',
    'TauJetsAuxDyn.NNDecayModeProb_3p0n',
    'TauJetsAuxDyn.NNDecayModeProb_3pXn',
]

#%----------------------------------------------------------------------
# variables to normalize
VARNORM = [
    'TauJetsAuxDyn.mu', 
    'TauJetsAuxDyn.nVtxPU',
    'TauJetsAuxDyn.rho',
    'TauJetsAuxDyn.ClustersMeanCenterLambda',
    'TauJetsAuxDyn.ClustersMeanFirstEngDens',
    'TauJetsAuxDyn.ClustersMeanSecondLambda',
    'TauJetsAuxDyn.ptCombined',
    'TauJetsAuxDyn.etaPanTauCellBased',
    'TauJetsAuxDyn.ptTauEnergyScale'
]

#%----------------------------------------------------------------------
# Train and plot without combined variables

TARGET_FIELD = 'TauJetsAuxDyn.truthPtVisDressed/TauJetsAuxDyn.ptCombined'

FEATURES_NEW = [
    'TauJetsAuxDyn.mu', 
    'TauJetsAuxDyn.nVtxPU',
    'TauJetsAuxDyn.rho', 
    'TauJetsAuxDyn.ClustersMeanCenterLambda',
    'TauJetsAuxDyn.ClustersMeanFirstEngDens',
    'TauJetsAuxDyn.ClustersMeanSecondLambda',
    'TauJetsAuxDyn.ClustersMeanPresamplerFrac',
    'TauJetsAuxDyn.ClustersMeanEMProbability',
    'TauJetsAuxDyn.ptIntermediateAxisEM/TauJetsAuxDyn.ptIntermediateAxis',
    'TauJetsAuxDyn.ptPanTauCellBased/TauJetsAuxDyn.ptTauEnergyScale',
    'TauJetsAuxDyn.ptIntermediateAxis/TauJetsAuxDyn.ptTauEnergyScale',
    'TauJetsAuxDyn.etaPanTauCellBased',
    'TauJetsAuxDyn.PFOEngRelDiff',
    'TauJetsAuxDyn.ptTauEnergyScale',
    'TauJetsAuxDyn.NNDecayModeProb_1p0n',
    'TauJetsAuxDyn.NNDecayModeProb_1p1n',
    'TauJetsAuxDyn.NNDecayModeProb_1pXn',
    'TauJetsAuxDyn.NNDecayModeProb_3p0n',
    'TauJetsAuxDyn.NNDecayModeProb_3pXn',
    ]
TARGET_FIELD_NEW = 'TauJetsAuxDyn.truthPtVisDressed/TauJetsAuxDyn.ptTauEnergyScale'
