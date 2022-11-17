import numpy as np
import math
import awkward as ak
from coffea.nanoevents.methods import candidate

from processNano.utils import p4_sum, delta_r, cs_variables
from processNano.corrections.electronMassScale import electronScaleUncert



def compute_eleScaleUncertainty(output, e1, e2):

    e1 = ak.flatten(e1)
    e2 = ak.flatten(e2)

    e1P4sBarrelUp = ak.zip({
        "pt": e1.pt*1.02,
        "eta": e1.eta,
        "phi": e1.phi,
        "mass": e1.mass,
        "charge": e1.charge,
    }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

    e1P4sEndcapUp = ak.zip({
        "pt": e1.pt*1.01,
        "eta": e1.eta,
        "phi": e1.phi,
        "mass": e1.mass,
        "charge": e1.charge,
    }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

    e1P4sBarrelDown = ak.zip({
        "pt": e1.pt*0.98,
        "eta": e1.eta,
        "phi": e1.phi,
        "mass": e1.mass,
        "charge": e1.charge,
    }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

    e1P4sEndcapDown = ak.zip({
        "pt": e1.pt*0.99,
        "eta": e1.eta,
        "phi": e1.phi,
        "mass": e1.mass,
        "charge": e1.charge,
    }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

    e2P4sBarrelUp = ak.zip({
        "pt": e2.pt*1.02,
        "eta": e2.eta,
        "phi": e2.phi,
        "mass": e2.mass,
        "charge": e2.charge,
    }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

    e2P4sEndcapUp = ak.zip({
        "pt": e2.pt*1.01,
        "eta": e2.eta,
        "phi": e2.phi,
        "mass": e2.mass,
        "charge": e2.charge,
    }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

    e2P4sBarrelDown = ak.zip({
        "pt": e2.pt*0.98,
        "eta": e2.eta,
        "phi": e2.phi,
        "mass": e2.mass,
        "charge": e2.charge,
    }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

    e2P4sEndcapDown = ak.zip({
        "pt": e2.pt*0.99,
        "eta": e2.eta,
        "phi": e2.phi,
        "mass": e2.mass,
        "charge": e2.charge,
    }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    


    tempFrame = output.loc[output['isDielectron']]


    bbMask = (abs(e1.eta) < 1.4442) & (abs(e2.eta) < 1.442)
    tempFrame.loc[np.array(bbMask), 'dilepton_mass_scaleUncUp'] = (e1P4sBarrelUp[bbMask] + e2P4sBarrelUp[bbMask]).mass
    tempFrame.loc[np.array(bbMask), 'dilepton_mass_scaleUncDown'] = (e1P4sBarrelDown[bbMask] + e2P4sBarrelDown[bbMask]).mass

    beMask = (abs(e1.eta) < 1.4442) & (abs(e2.eta) > 1.442)
    tempFrame.loc[np.array(beMask), 'dilepton_mass_scaleUncUp'] = (e1P4sBarrelUp[beMask] + e2P4sEndcapUp[beMask]).mass
    tempFrame.loc[np.array(beMask), 'dilepton_mass_scaleUncDown'] = (e1P4sBarrelDown[beMask] + e2P4sEndcapDown[beMask]).mass

    ebMask = (abs(e1.eta) > 1.4442) & (abs(e2.eta) < 1.442)
    tempFrame.loc[np.array(ebMask), 'dilepton_mass_scaleUncUp'] = (e1P4sEndcapUp[ebMask] + e2P4sBarrelUp[ebMask]).mass
    tempFrame.loc[np.array(ebMask), 'dilepton_mass_scaleUncDown'] = (e1P4sEndcapDown[ebMask] + e2P4sBarrelDown[ebMask]).mass

    eeMask = (abs(e1.eta) > 1.4442) & (abs(e2.eta) > 1.442)
    tempFrame.loc[np.array(eeMask), 'dilepton_mass_scaleUncUp'] = (e1P4sEndcapUp[eeMask] + e2P4sEndcapUp[eeMask]).mass
    tempFrame.loc[np.array(eeMask), 'dilepton_mass_scaleUncDown'] = (e1P4sEndcapDown[eeMask] + e2P4sEndcapDown[eeMask]).mass


    output.loc[output['isDielectron']] = tempFrame

    return output


def fill_electrons(output, e1, e2, is_mc=True):
    e1_variable_names = [
        "e1_pt",
        "e1_pt_gen",
        "e1_pt_over_mass",
        "e1_ptErr",
        "e1_eta",
        "e1_eta_gen",
        "e1_phi",
        "e1_phi_gen",
        "e1_iso",
        "e1_dxy",
        "e1_dz",
        "e1_genPartFlav",
        "e1_ip3d",
        "e1_sip3d",
    ]
    e2_variable_names = [
        "e2_pt",
        "e2_pt_gen",
        "e2_pt_over_mass",
        "e2_ptErr",
        "e2_eta",
        "e2_eta_gen",
        "e2_phi",
        "e2_phi_gen",
        "e2_iso",
        "e2_dxy",
        "e2_dz",
        "e2_genPartFlav",
        "e2_ip3d",
        "e2_sip3d",
    ]

    dilepton_variable_names = [
        "dilepton_mass",
#        "dilepton_mass_gen",
        "dilepton_mass_res",
        "dilepton_mass_res_rel",
        "dilepton_ebe_mass_res",
        "dilepton_ebe_mass_res_rel",
        "dilepton_pt",
        "dilepton_pt_log",
        "dilepton_eta",
        "dilepton_phi",
        "dilepton_dEta",
        "dilepton_dPhi",
        "dilepton_dR",
        "dilepton_rap",
        "bbangle", 
        "dilepton_cos_theta_cs",
        "dilepton_phi_cs",
        "wgt_nominal",
        "pu_wgt",
    ]
    v_names = e1_variable_names + e2_variable_names + dilepton_variable_names

    # Initialize columns for electron variables
    for n in v_names:
        output[n] = 0.0

    # Fill single electron variables
    ee = p4_sum(e1, e2, is_mc)
    for v in [
        "pt",
        "eta",
        "phi",
        "pt_raw",
        "eta_raw",
        "phi_raw",
        "pt_gen",
        "eta_gen",
        "phi_gen",
    ]:
        try:
            output[f"e1_{v}"] = e1[v]
            output[f"e2_{v}"] = e2[v]
        except Exception:
            output[f"e1_{v}"] = -999.0
            output[f"e2_{v}"] = -999.0

    # Fill dilepton variables
    for v in [
        "pt",
        "eta",
        "phi",
        "mass",
        "pt_gen",
        "eta_gen",
        "phi_gen",
        "mass_gen",
        "rap",
    ]:
        name = f"dilepton_{v}"
        try:
            output[name] = ee[v]
            output[name] = output[name].fillna(-999.0)
        except Exception:
            output[name] = -999.0

    output["dilepton_pt_log"] = np.log(output.dilepton_pt[output.dilepton_pt > 0])
    output.loc[output.dilepton_pt < 0, "dilepton_pt_log"] = -999.0

    ee_deta, ee_dphi, ee_dr = delta_r(e1.eta, e2.eta, e1.phi, e2.phi)
    output["dilepton_pt"] = ee.pt
    output["dilepton_eta"] = ee.eta
    output["dilepton_phi"] = ee.phi
    output["dilepton_dEta"] = ee_deta
    output["dilepton_dPhi"] = ee_dphi
    output["dilepton_dR"] = ee_dr
    output["dilepton_cos_theta_cs"], output["dilepton_phi_cs"] = cs_variables(
        e1, e2
    )


    ee = p4_sum(e1, e2, is_mc,eScale="Up")
    for v in [
        "mass_scaleUncUp",
    ]:
        name = f"dilepton_{v}"
        try:
            output[name] = ee[v]
            output[name] = output[name].fillna(-999.0)
        except Exception:
            output[name] = -999.0

    ee = p4_sum(e1, e2, is_mc,eScale="Down")
    for v in [
        "mass_scaleUncDown",
    ]:
        name = f"dilepton_{v}"
        try:
            output[name] = ee[v]
            output[name] = output[name].fillna(-999.0)
        except Exception:
            output[name] = -999.0
