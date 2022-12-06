import numpy as np
import awkward as ak
import math
from coffea.nanoevents.methods import candidate
from processNano.utils import p4_sum, delta_r, cs_variables
from processNano.corrections.muonMassResolution import smearMass
from processNano.corrections.muonMassScale import muonScaleUncert
from processNano.corrections.muonRecoUncert import muonRecoUncert


def initialize_leptons(processor, output):

    l1_variable_names = [
        "l1_pt",
        "l1_pt_gen",
        "l1_pt_over_mass",
        "l1_ptErr",
        "l1_eta",
        "l1_eta_gen",
        "l1_phi",
        "l1_phi_gen",
        "l1_iso",
        "l1_dxy",
        "l1_dz",
        "l1_genPartFlav",
        "l1_ip3d",
        "l1_sip3d",
    ]
    l2_variable_names = [
        "l2_pt",
        "l2_pt_gen",
        "l2_pt_over_mass",
        "l2_ptErr",
        "l2_eta",
        "l2_eta_gen",
        "l2_phi",
        "l2_phi_gen",
        "l2_iso",
        "l2_dxy",
        "l2_dz",
        "l2_genPartFlav",
        "l2_ip3d",
        "l2_sip3d",
    ]
    dilepton_variable_names = [
        "dilepton_mass",
        "dilepton_mass_gen",
        "dilepton_mass_res",
        "dilepton_mass_res_rel",
        "dilepton_ebe_mass_res",
        "dilepton_ebe_mass_res_rel",
        "dilepton_pt",
        "dilepton_pt_log",
        "dilepton_eta",
        "dilepton_phi",
        "dilepton_pt_gen",
        "dilepton_eta_gen",
        "dilepton_phi_gen",
        "dilepton_dEta",
        "dilepton_dPhi",
        "dilepton_dR",
        "dilepton_rap",
        "bbangle",
        "dilepton_cos_theta_cs",
        "dilepton_phi_cs",
        "wgt_nominal",
    ]
    v_names = l1_variable_names + l2_variable_names + dilepton_variable_names

    # Initialize columns for lepton variables

    for n in v_names:
        output[n] = 0.0

    return output


def fill_leptons(processor, output, dilepton,  l1, l2, is_mc, year, flavor):

    # Fill single lepton variables
    varList = []
    if flavor == "isDimuon": 
       varList = [
            "pt",
            "ptErr",
            "eta",
            "phi",
            "dxy",
            "dz",
            "ip3d",
            "sip3d",
            "charge",
            "tkRelIso",
        ]

    else:
        varList = [
            "pt",
            "eta",
            "phi",
            "dxy",
            "dz",
            "ip3d",
            "sip3d",
            "charge",
            "tkRelIso",  # dummy
        ]


    for v in varList: 
        if hasattr(l1,v):
            output.loc[output[flavor], f"l1_{v}"] = getattr(l1,v)
        else:
            output.loc[output[flavor], f"l1_{v}"] = -999.
        if hasattr(l2,v):
            output.loc[output[flavor], f"l2_{v}"] = getattr(l2,v)
        else:
            output.loc[output[flavor], f"l2_{v}"] = -999.

    if is_mc:
        for v in varList: 
            output.loc[output[flavor], f"l1_genPartFlav"] = l1.genPartFlav
            output.loc[output[flavor], f"l2_genPartFlav"] = l2.genPartFlav

    for v in [
        "pt",
        "eta",
        "phi",
        "mass",
        #"rap",
    ]:
        name = f"dilepton_{v}"
        output.loc[output[flavor], name] = getattr(dilepton,v)
        output.loc[output[flavor], name] = output.loc[output[flavor], name].fillna(-999.0)
    
    output.loc[output[flavor], "l1_pt_over_mass"] = output.loc[output[flavor]].l1_pt.values / output.loc[output[flavor]].dilepton_mass.values
    output.loc[output[flavor], "l2_pt_over_mass"] = output.loc[output[flavor]].l2_pt.values / output.loc[output[flavor]].dilepton_mass.values
    output.loc[output[flavor], "dilepton_pt_log"] = np.log(output.loc[output[flavor]].dilepton_pt[output.loc[output[flavor]].dilepton_pt > 0])
    output.loc[output[flavor] & output.dilepton_pt < 0, "dilepton_pt_log"] = -999.0

    ll_deta, ll_dphi, ll_dr = delta_r(l1.eta, l2.eta, l1.phi, l2.phi)
    output.loc[output[flavor], "dilepton_dEta"] = ll_deta
    output.loc[output[flavor], "dilepton_dPhi"] = ll_dphi
    output.loc[output[flavor], "dilepton_dR"] = ll_dr

    cs_cos, phi_cs =  cs_variables(ak.to_pandas(l1), ak.to_pandas(l2))
    output.loc[output[flavor], "dilepton_cos_theta_cs"] = np.array(cs_cos)
    output.loc[output[flavor], "dilepton_phi_cs"] = np.array(phi_cs)

    if is_mc:

        l1GenP4s = ak.zip({
            "pt": l1.matched_gen.pt,
            "eta": l1.matched_gen.eta,
            "phi": l1.matched_gen.phi,
            "mass": l1.matched_gen.mass,
            "charge": l1.charge,
        }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

        l2GenP4s = ak.zip({
            "pt": l2.matched_gen.pt,
            "eta": l2.matched_gen.eta,
            "phi": l2.matched_gen.phi,
            "mass": l2.matched_gen.mass,
            "charge": l2.charge,
        }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    


        l1GenP4s = ak.flatten(l1GenP4s)
        l2GenP4s = ak.flatten(l2GenP4s)

        genMask1 = ~ak.is_none(l1GenP4s.pt)    
        leptonGen1 = l1GenP4s[genMask1]
 
        genMask2 = ~ak.is_none(l2GenP4s.pt)    
        leptonGen2 = l2GenP4s[genMask2]
 
        tempFrame = output.loc[output[flavor]]

        for v in [
            "pt",
            "eta",
            "phi",
        ]:
            tempFrame.loc[np.array(genMask1), f"l1_{v}_gen"] = getattr(leptonGen1,v)
            tempFrame.loc[np.array(genMask2), f"l2_{v}_gen"] = getattr(leptonGen2,v)

        dileptonGen = l1GenP4s + l2GenP4s

        genMask = ~ak.is_none(dileptonGen.pt)    
        dileptonGen = dileptonGen[genMask]

        for v in [
            "pt",
            "eta",
            "phi",
            "mass",
            #"rap",
        ]:
            name = f"dilepton_{v}_gen"
            tempFrame.loc[np.array(genMask), name] = getattr(dileptonGen,v)

        output.loc[output[flavor]] = tempFrame

    return output


