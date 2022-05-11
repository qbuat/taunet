"""
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
    'TauJetsAuxDyn.PanTau_BDTValue_1p0n_vs_1p1n',
    'TauJetsAuxDyn.PanTau_BDTValue_1p1n_vs_1pXn',
    'TauJetsAuxDyn.PanTau_BDTValue_1p1n_vs_1pXn',
    'TauJetsAuxDyn.nTracks',
    'TauJetsAuxDyn.PFOEngRelDiff',
    ]

TRUTH_FIELDS = [
    'TauJetsAuxDyn.truthPtVisDressed',
    'TauJetsAuxDyn.truthEtaVisDressed',
    # 'TauJetsAuxDyn.truthPhiVisDressed',
    ]

OTHER_TES = [
    'TauJetsAuxDyn.ptCombined',
    'TauJetsAuxDyn.ptFinalCalib',
]

TARGET_FIELD = 'TauJetsAuxDyn.truthPtVisDressed/TauJetsAuxDyn.ptCombined'
