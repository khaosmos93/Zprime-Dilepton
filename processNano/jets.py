import numpy as np
import pandas as pd
import awkward as ak
from processNano.utils import p4, p4_sum, delta_r, rapidity
import correctionlib
import pickle
from coffea.lookup_tools.dense_lookup import dense_lookup
from coffea.nanoevents.methods import candidate
from coffea.btag_tools import BTagScaleFactor
from config.parameters import parameters

# HERE
import pprint
pp = pprint.PrettyPrinter(indent=4)

def btagSF_new(jets, year, correction="shape", is_UL=True):

    if is_UL:
        if correction == "wp":
            systs = [
                "central",
                "up",
                "down",
                "up_correlated",
                "up_uncorrelated",
                "down_correlated",
                "down_uncorrelated",
                "up_isr",
                "down_isr",
                "up_fsr",
                "down_fsr",
                "up_hdamp",
                "down_hdamp",
                "up_jes",
                "down_jes",
                "up_jer",
                "down_jer",
                "up_pileup",
                "down_pileup",
                "up_qcdscale",
                "down_qcdscale",
                "up_statistic",
                "down_statistic",
                "up_topmass",
                "down_topmass",
                "up_type3",
                "down_type3",
            ]
        else:
            jets["btag_sf_shape"] = 1.0
            systs = ["central"]
        cset = correctionlib.CorrectionSet.from_file(parameters["btag_sf_UL"][year])
    else:
        if correction == "wp":
            systs = [
                "central",
                "up",
                "down",
                "up_correlated",
                "up_uncorrelated",
                "down_correlated",
                "down_uncorrelated",
            ]
        else:
            systs = ["central"]

        cset = BTagScaleFactor(parameters["btag_sf_pre_UL"][year], "medium")

    mask = (jets.pt > 30.0) & (abs(jets.eta) < 2.4) & (jets.jetId >= 2)

    btag_wp_sf = {}
    btag_wp_wgt = {}

    for syst in systs:
        if correction == "shape":
            jets[f"btag_sf_shape_{syst}"] = 1.0
            j, nj = ak.flatten(jets[mask]), ak.num(jets[mask])

            if is_UL:
                sf = cset["deepJet_shape"].evaluate(syst, np.array(j.hadronFlavour), np.abs(np.array(j.eta)), np.array(j.pt), np.array(j.btagDeepFlavB))
                sf = ak.unflatten(sf,nj)

            if syst == "central":
                jets[mask]["btag_sf_shape"] = sf

            else:
                jets[mask][f"btag_sf_shape_{syst}"] = sf
        elif correction == "wp":
            btag_wp_sf[syst] = {}
            btag_wp_wgt[syst] = {}
            if syst == "central":
                jets["btag_sf_wp"] = 1.0
            else:
                jets[f"btag_sf_wp_{syst}"] = 1.0
            path_eff = parameters["btag_sf_eff"][year]
            wp = parameters["UL_btag_medium"][year]
            with open(path_eff, "rb") as handle:
                eff = pickle.load(handle)
           
            efflookup = dense_lookup(eff.values(), [ax.edges for ax in eff.axes])
            btag_eff_flavor_bin = {
                0: 0,
                4: 1,
                5: 2
            }
            keys = [0,4,5]
            for ikey, key in enumerate(keys):
                # if key == 0: 
                #     mask_flavor = jets["hadronFlavour"] < 4
                # else:
                #     mask_flavor = jets["hadronFlavour"] > 4
                mask_flavor = (jets["hadronFlavour"] == key)

                j = ak.flatten(jets[mask & mask_flavor])
                nj = ak.num(jets[mask & mask_flavor])
                flavor = j.hadronFlavour.to_numpy()
                eta = np.abs(j.eta.to_numpy())
                pt = j.pt.to_numpy()
                mva = j.btagDeepFlavB.to_numpy()

                if is_UL:
                    if key < 4:
                        corr = "deepJet_incl"
                    else:
                        corr = "deepJet_comb"
                    try:
                        fac = cset[corr].evaluate(syst, "M", flavor, eta, pt)
                    except Exception:
                        fac = cset[corr].evaluate("central", "M", flavor, eta, pt)
                else:
                    try:
                        fac = cset.eval(syst, flavor, eta, pt, wp)
                    except Exception:
                        fac = cset.eval("central", flavor, eta, pt, wp)

                prob = efflookup(pt, eta, btag_eff_flavor_bin[key])
                prob_nosf = np.copy(prob)
                prob_sf = np.copy(prob) * fac
                prob_sf[mva < wp] = 1.0 - prob_sf[mva < wp]
                prob_nosf[mva < wp] = 1.0 - prob_nosf[mva < wp]
                sf = prob_sf / prob_nosf
                sf = ak.unflatten(sf,nj)

                btag_wp_sf[syst][key] = np.copy(sf)
                btag_wp_wgt[syst][key] = ak.prod(sf, axis=1)

                if ikey == 0:
                    btag_wp_wgt[syst]["wgt"] = ak.prod(sf, axis=1)
                else:
                    btag_wp_wgt[syst]["wgt"] = btag_wp_wgt[syst]["wgt"]*ak.prod(sf, axis=1)

                
#            if syst == "central":
#                jets["btag_sf_wp"].fill_none(1.0)
#            else:
#                jets[f"btag_sf_wp_{syst}"].fill_none(1.0)

    return jets, btag_wp_wgt


def btagSF(df, year, correction="shape", is_UL=True):

    if is_UL:
        if correction == "wp":
            systs = [
                "central",
                "up",
                "down",
                "up_correlated",
                "up_uncorrelated",
                "down_correlated",
                "down_uncorrelated",
                "up_isr",
                "down_isr",
                "up_fsr",
                "down_fsr",
                "up_hdamp",
                "down_hdamp",
                "up_jes",
                "down_jes",
                "up_jer",
                "down_jer",
                "up_pileup",
                "down_pileup",
                "up_qcdscale",
                "down_qcdscale",
                "up_statistic",
                "down_statistic",
                "up_topmass",
                "down_topmass",
                "up_type3",
                "down_type3",
            ]
        else:
            systs = ["central"]
        cset = correctionlib.CorrectionSet.from_file(parameters["btag_sf_UL"][year])
    else:
        if correction == "wp":
            systs = [
                "central",
                "up",
                "down",
                "up_correlated",
                "up_uncorrelated",
                "down_correlated",
                "down_uncorrelated",
            ]
        else:
            systs = ["central"]

        cset = BTagScaleFactor(parameters["btag_sf_pre_UL"][year], "medium")

    df["pre_selection"] = False
    df.loc[
        (df.pt > 30.0) & (abs(df.eta) < 2.4) & (df.jetId >= 2), "pre_selection"
    ] = True
    mask = df["pre_selection"]
    for syst in systs:
        if correction == "shape":

            df["btag_sf_shape"] = 1.0
            flavor = df[mask].hadronFlavour.to_numpy()
            eta = np.abs(df[mask].eta.to_numpy())
            pt = df[mask].pt.to_numpy()
            mva = df[mask].btagDeepFlavB.to_numpy()

            if is_UL:
                sf = cset["deepJet_shape"].evaluate(syst, flavor, eta, pt, mva)
            if syst == "central":
                df.loc[mask, "btag_sf_shape"] = sf
            else:
                df.loc[mask, f"btag_sf_shape_{syst}"] = sf

        elif correction == "wp":

            is_bc = df["hadronFlavour"] >= 4
            is_light = df["hadronFlavour"] < 4
            path_eff = parameters["btag_sf_eff"][year]
            wp = parameters["UL_btag_medium"][year]
            with open(path_eff, "rb") as handle:
                eff = pickle.load(handle)

            efflookup = dense_lookup(eff.values(), [ax.edges for ax in eff.axes])
            mask_dict = {0: is_light, 4: is_bc, 5: is_bc}
            for key in mask_dict.keys():

                mask_flavor = mask_dict[key]
                flavor = df[mask & mask_flavor].hadronFlavour.to_numpy()
                eta = np.abs(df[mask & mask_flavor].eta.to_numpy())
                pt = df[mask & mask_flavor].pt.to_numpy()
                mva = df[mask & mask_flavor].btagDeepFlavB.to_numpy()

                if is_UL:
                    if key < 4:
                        corr = "deepJet_incl"
                    else:
                        corr = "deepJet_comb"
                    try:
                        fac = cset[corr].evaluate(syst, "M", flavor, eta, pt)
                    except Exception:
                        fac = cset[corr].evaluate("central", "M", flavor, eta, pt)
                else:
                    try:

                        fac = cset.eval(syst, flavor, eta, pt, wp)
                        print("uncertainty is " + syst)
                    except Exception:
                        fac = cset.eval("central", flavor, eta, pt, wp)

                prob = efflookup(pt, eta, key)
                prob_nosf = np.copy(prob)
                prob_sf = np.copy(prob) * fac
                prob_sf[mva < wp] = 1.0 - prob_sf[mva < wp]
                prob_nosf[mva < wp] = 1.0 - prob_nosf[mva < wp]
                sf = prob_sf / prob_nosf
                if syst == "central":
                    df.loc[mask & mask_flavor, "btag_sf_wp"] = sf
                else:
                    df.loc[mask & mask_flavor, f"btag_sf_wp_{syst}"] = sf
            if syst == "central":
                df["btag_sf_wp"].fillna(1.0)
            else:
                df[f"btag_sf_wp_{syst}"].fillna(1.0)


def prepare_jets(df, is_mc):
    # Initialize missing fields (needed for JEC)
    df["Jet", "pt_raw"] = (1 - df.Jet.rawFactor) * df.Jet.pt
    df["Jet", "mass_raw"] = (1 - df.Jet.rawFactor) * df.Jet.mass
    # HERE tmp, incompetible with Run 3 nano (v10)
    # df["Jet", "rho"] = ak.broadcast_arrays(df.fixedGridRhoFastjetAll, df.Jet.pt)[0]

    if is_mc:
        df["Jet", "pt_gen"] = ak.values_astype(
            ak.fill_none(df.Jet.matched_gen.pt, 0), np.float32
        )


def fill_jets(output, variables, jets, njet, muons, electrons, is_mc=True):
    variable_names = [
        "jet1_pt",
        "jet1_eta",
        "jet1_rap",
        "jet1_phi",
        "jet1_qgl",
        "jet1_jetId",
        "jet1_puId",
        "jet1_btagDeepB",
        "jet1_btagDeepFlavB",
        "jet2_pt",
        "jet2_eta",
        "jet2_rap",
        "jet2_phi",
        "jet2_qgl",
        "jet2_jetId",
        "jet2_puId",
        "jet2_btagDeepB",
        "jet2_btagDeepFlavB",
        "jet1_sf",
        "jet2_sf",
        "jj_mass",
        "jj_mass_log",
        "jj_pt",
        "jj_eta",
        "jj_phi",
        "jj_dEta",
        "jj_dPhi",
        "llj1_dEta",
        "llj1_dPhi",
        "llj1_dR",
        "llj2_dEta",
        "llj2_dPhi",
        "llj2_dR",
        "llj_min_dEta",
        "llj_min_dPhi",
        "lljj_pt",
        "lljj_eta",
        "lljj_phi",
        "lljj_mass",
        "rpt",
        "zeppenfeld",
        "ll_zstar_log",
        "nsoftjets2",
        "nsoftjets5",
        "htsoft2",
        "htsoft5",
        "selection",
    ]

    for v in variable_names:
        variables[v] = -999.0

    jet1 = jets[0]
    jet2 = jets[1]

    jet1Mask = ~ak.is_none(jet1)
    jet2Mask = ~ak.is_none(jet2)

    # Fill single jet variables
    for v in [
        "pt",
        "eta",
        "phi",
        "mass",
        # "pt_gen",
        # "eta_gen",
        # "phi_gen",
        # "qgl",
        "btagDeepB",
        "btagDeepFlavB",
    ]:
        variables.loc[np.array(jet1Mask), f"jet1_{v}"] = getattr(jet1[jet1Mask], v)
        variables.loc[np.array(jet2Mask), f"jet2_{v}"] = getattr(jet2[jet2Mask], v)

    variables.loc[np.array(jet1Mask), "jet1_rap"] = rapidity(jet1[jet1Mask])
    variables.loc[np.array(jet2Mask), "jet2_rap"] = rapidity(jet2[jet2Mask])

    jet1Filtered = jet1[jet2Mask]
    jet2Filtered = jet2[jet2Mask]

    jet1Filtered["charge"] = 1 
    jet2Filtered["charge"] = -1 

    jet1P4s = ak.zip({
        "pt": jet1Filtered.pt,
        "eta": jet1Filtered.eta,
        "phi": jet1Filtered.phi,
        "mass": jet1Filtered.mass,
        "charge": jet1Filtered.charge,
    }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

    jet2P4s = ak.zip({
        "pt": jet2Filtered.pt,
        "eta": jet2Filtered.eta,
        "phi": jet2Filtered.phi,
        "mass": jet2Filtered.mass,
        "charge": jet2Filtered.charge,
    }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

    jj = jet1P4s + jet2P4s

    if is_mc:

        jet1Filtered.matched_gen["charge"] = 1
        jet2Filtered.matched_gen["charge"] = -1

        jet1GenP4s = ak.zip({
            "pt": jet1Filtered.matched_gen.pt,
            "eta": jet1Filtered.matched_gen.eta,
            "phi": jet1Filtered.matched_gen.phi,
            "mass": jet1Filtered.matched_gen.mass,
            "charge": jet1Filtered.charge,
        }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

        jet2GenP4s = ak.zip({
            "pt": jet2Filtered.matched_gen.pt,
            "eta": jet2Filtered.matched_gen.eta,
            "phi": jet2Filtered.matched_gen.phi,
            "mass": jet2Filtered.matched_gen.mass,
            "charge": jet2Filtered.charge,
        }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

        jjGen = jet1GenP4s + jet2GenP4s

    #    for v in [
    #        "pt",
    #        "eta",
    #        "phi",
    #        "mass",
    #    ]:
    #        variables.loc[np.array(jet2Mask), f"jj_{v}_gen"] = getattr(jjGen, v)


    for v in [
        "pt",
        "eta",
        "phi",
        "mass",
    ]:
        variables.loc[np.array(jet2Mask), f"jj_{v}"] = getattr(jj, v)

    variables.jj_mass_log = np.log(variables.jj_mass)

    variables.jj_dEta, variables.jj_dPhi, _ = delta_r(
        variables.jet1_eta,
        variables.jet2_eta,
        variables.jet1_phi,
        variables.jet2_phi,
    )

    muJet1 = jet1[output["isDimuon"]]
    muJet2 = jet2[output["isDimuon"]]

    haveMu = True
    if len(muJet1) == 1:
        if ak.is_none(muJet1):
            haveMu = False

    if haveMu and len(muJet1) > 0:

        muJet2Mask = ~ak.is_none(muJet2)

        muJet1Filtered = muJet1[muJet2Mask]
        muJet2Filtered = muJet2[muJet2Mask]

        muJet1Filtered["charge"] = 1 
        muJet2Filtered["charge"] = -1 

        muJet1P4s = ak.zip({
            "pt": muJet1Filtered.pt,
            "eta": muJet1Filtered.eta,
            "phi": muJet1Filtered.phi,
            "mass": muJet1Filtered.mass,
            "charge": muJet1Filtered.charge,
         }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

        muJet2P4s = ak.zip({
            "pt": muJet2Filtered.pt,
            "eta": muJet2Filtered.eta,
            "phi": muJet2Filtered.phi,
            "mass": muJet2Filtered.mass,
            "charge": muJet2Filtered.charge,
        }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

        mmjj = muons[0][muJet2Mask] + muons[1][muJet2Mask] + muJet1P4s + muJet2P4s

        if len(mmjj) > 0:
            for v in [
                "pt",
                "eta",
                "phi",
                "mass",
            ]:
                variables.loc[output["isDimuon"] & np.array(jet2Mask), f"lljj_{v}"] = getattr(mmjj, v)
   
    elJet1 = jet1[output["isDielectron"]]
    elJet2 = jet2[output["isDielectron"]]

    haveEl = True
    if len(elJet1) == 1:
        if ak.is_none(elJet1):
            haveEl = False

    if haveEl and len(elJet1) > 0:

        elJet2Mask = ~ak.is_none(elJet2)

        elJet1Filtered = elJet1[elJet2Mask]
        elJet2Filtered = elJet2[elJet2Mask]

        elJet1Filtered["charge"] = 1 
        elJet2Filtered["charge"] = -1 

        elJet1P4s = ak.zip({
            "pt": elJet1Filtered.pt,
            "eta": elJet1Filtered.eta,
            "phi": elJet1Filtered.phi,
            "mass": elJet1Filtered.mass,
            "charge": elJet1Filtered.charge,
        }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

        elJet2P4s = ak.zip({
            "pt": elJet2Filtered.pt,
            "eta": elJet2Filtered.eta,
            "phi": elJet2Filtered.phi,
            "mass": elJet2Filtered.mass,
            "charge": elJet2Filtered.charge,
        }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

        eejj = electrons[0][elJet2Mask] + electrons[1][elJet2Mask] + elJet1P4s + elJet2P4s

        if len(eejj) > 0:
            for v in [
                "pt",
                "eta",
                "phi",
                "mass",
            ]:
                 variables.loc[output["isDielectron"] & np.array(jet2Mask), f"lljj_{v}"] = getattr(eejj, v)

    dilepton_pt, dilepton_eta, dilepton_phi, dilepton_rap = (
       output.dilepton_pt,
       output.dilepton_eta,
       output.dilepton_phi,
       output.dilepton_rap,
    )

    variables.zeppenfeld = dilepton_eta - 0.5 * (
       variables.jet1_eta + variables.jet2_eta
    )

    variables.rpt = variables.lljj_pt / (
        dilepton_pt + variables.jet1_pt + variables.jet2_pt
    )

    ll_ystar = dilepton_rap - (variables.jet1_rap + variables.jet2_rap) / 2

    ll_zstar = abs(ll_ystar / (variables.jet1_rap - variables.jet2_rap))

    variables.ll_zstar_log = np.log(ll_zstar)

    variables.llj1_dEta, variables.llj1_dPhi, variables.llj1_dR = delta_r(
        dilepton_eta, variables.jet1_eta, dilepton_phi, variables.jet1_phi
    )

    variables.llj2_dEta, variables.llj2_dPhi, variables.llj2_dR = delta_r(
        dilepton_eta, variables.jet2_eta, dilepton_phi, variables.jet2_phi
    )

    variables.llj_min_dEta = np.where(
        variables.llj1_dEta,
        variables.llj2_dEta,
        (variables.llj1_dEta < variables.llj2_dEta),
    )

    variables.llj_min_dPhi = np.where(
        variables.llj1_dPhi,
        variables.llj2_dPhi,
        (variables.llj1_dPhi < variables.llj2_dPhi),
    )


def initialize_bjet_var(variables):
    variable_names = [
        "bjet1_pt",
        "bjet1_eta",
        "bjet1_rap",
        "bjet1_phi",
        "bjet1_qgl",
        "bjet1_jetId",
        "bjet1_puId",
        "bjet1_btagDeepB",
        "bjet1_btagDeepFlavB",
        "bjet2_pt",
        "bjet2_eta",
        "bjet2_rap",
        "bjet2_phi",
        "bjet2_qgl",
        "bjet2_jetId",
        "bjet2_puId",
        "bjet2_btagDeepB",
        "bjet2_btagDeepFlavB",
        "bjet1_sf",
        "bjet2_sf",
        "bjj_mass",
        "bjj_mass_log",
        "bjj_pt",
        "bjj_eta",
        "bjj_phi",
        "bjj_dEta",
        "bjj_dPhi",
        "bllj1_dEta",
        "bllj1_dPhi",
        "bllj1_dR",
        "bllj2_dEta",
        "bllj2_dPhi",
        "bllj2_dR",
        "bllj_min_dEta",
        "bllj_min_dPhi",
        "blljj_pt",
        "blljj_eta",
        "blljj_phi",
        "blljj_mass",
        "bllj1_pt",
        "bllj1_eta",
        "bllj1_phi",
        "bllj1_mass",
        "bllj2_pt",
        "bllj2_eta",
        "bllj2_phi",
        "bllj2_mass",
        "b1l1_mass",
        "b1l2_mass",
        "b2l1_mass",
        "b2l2_mass",
        "min_bl_mass",
        "min_b1l_mass",
        "min_b2l_mass",
        "bselection",
    ]

    for v in variable_names:
        variables[v] = -999.0

    return variables


def fill_bjets(output, variables, jets, leptons, flavor, is_mc=True):


    jet1 = jets[0]
    jet2 = jets[1]
    lepton1 = leptons[0]
    lepton2 = leptons[1]
    jet1Mask = ~ak.is_none(jet1)
    jet2Mask = ~ak.is_none(jet2)

   # Fill single jet variables
    for v in [
        "pt",
        "eta",
        "phi",
        # "pt_gen",
        # "eta_gen",
        # "phi_gen",
        # "qgl",
        "btagDeepB",
        "btagDeepFlavB",
    ]:
        variables.loc[np.array(jet1Mask), f"bjet1_{v}"] = getattr(jet1[jet1Mask], v)
        variables.loc[np.array(jet2Mask), f"bjet2_{v}"] = getattr(jet2[jet2Mask], v)

    if is_mc:
        for v in [
            "btag_sf_shape",
            "btag_sf_wp",
        ]:
            variables.loc[np.array(jet1Mask), f"bjet1_{v}"] = getattr(jet1[jet1Mask], v)
            variables.loc[np.array(jet2Mask), f"bjet2_{v}"] = getattr(jet2[jet2Mask], v)

    else:
        for v in [
            "btag_sf_shape",
            "btag_sf_wp",
        ]:
            variables.loc[np.array(jet1Mask), f"bjet1_{v}"] = 1.0
            variables.loc[np.array(jet2Mask), f"bjet2_{v}"] = 1.0



    variables.loc[np.array(jet1Mask), "bjet1_rap"] = rapidity(jet1[jet1Mask])
    variables.loc[np.array(jet2Mask), "bjet2_rap"] = rapidity(jet2[jet2Mask])

    jet1Filtered = jet1[jet2Mask]
    jet2Filtered = jet2[jet2Mask]

    jet1Filtered["charge"] = 1 
    jet2Filtered["charge"] = -1 

    jet1P4s = ak.zip({
        "pt": jet1Filtered.pt,
        "eta": jet1Filtered.eta,
        "phi": jet1Filtered.phi,
        "mass": jet1Filtered.mass,
        "charge": jet1Filtered.charge,
    }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

    jet2P4s = ak.zip({
        "pt": jet2Filtered.pt,
        "eta": jet2Filtered.eta,
        "phi": jet2Filtered.phi,
        "mass": jet2Filtered.mass,
        "charge": jet2Filtered.charge,
    }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

    jj = jet1P4s + jet2P4s


    if is_mc:

        jet1Filtered.matched_gen["charge"] = 1
        jet2Filtered.matched_gen["charge"] = -1

        jet1GenP4s = ak.zip({
            "pt": jet1Filtered.matched_gen.pt,
            "eta": jet1Filtered.matched_gen.eta,
            "phi": jet1Filtered.matched_gen.phi,
            "mass": jet1Filtered.matched_gen.mass,
            "charge": jet1Filtered.charge,
        }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

        jet2GenP4s = ak.zip({
            "pt": jet2Filtered.matched_gen.pt,
            "eta": jet2Filtered.matched_gen.eta,
            "phi": jet2Filtered.matched_gen.phi,
            "mass": jet2Filtered.matched_gen.mass,
            "charge": jet2Filtered.charge,
        }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

        jjGen = jet1GenP4s + jet2GenP4s

        #for v in [
        #    "pt",
        #    "eta",
        #    "phi",
        #    "mass",
        #]:
        #    variables.loc[np.array(jet2Mask), f"bjj_{v}_gen"] = getattr(jjGen, v)


    for v in [
        "pt",
        "eta",
        "phi",
        "mass",
    ]:
        variables.loc[np.array(jet2Mask), f"bjj_{v}"] = getattr(jj, v)

    variables.bjj_mass_log = np.log(variables.bjj_mass)

    variables.bjj_dEta, variables.bjj_dPhi, _ = delta_r(
        variables.bjet1_eta,
        variables.bjet2_eta,
        variables.bjet1_phi,
        variables.bjet2_phi,
    )

    lJet1 = jet1[output[flavor]]
    lJet2 = jet2[output[flavor]]

    haveL = True
    if len(lJet1) == 1:
        if ak.is_none(lJet1):
            haveL = False

    if haveL and len(lJet1) > 0:

        lJet1Mask = ~ak.is_none(lJet1)
        lJet2Mask = ~ak.is_none(lJet2)

        lJet1Filtered = lJet1[lJet1Mask]
        lJet1Filtered["charge"] = 1 

        lJet1P4s = ak.zip({
            "pt": lJet1Filtered.pt,
            "eta": lJet1Filtered.eta,
            "phi": lJet1Filtered.phi,
            "mass": lJet1Filtered.mass,
            "charge": lJet1Filtered.charge,
        }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

        llj = leptons[0][lJet1Mask] + leptons[1][lJet1Mask] + lJet1P4s
        if len(llj) > 0:
            for v in [
                "pt",
                "eta",
                "phi",
                "mass",
            ]:
                variables.loc[variables[flavor] & jet1Mask, f"bllj1_{v}"] = getattr(llj, v)

        ml1 = leptons[0][lJet1Mask] + lJet1P4s
        ml2 = leptons[1][lJet1Mask] + lJet1P4s

        if len(ml1) > 0: 

            variables.loc[variables[flavor] & jet1Mask, "b1l1_mass"] = ml1.mass
            variables.loc[variables[flavor] & jet1Mask, "b1l2_mass"] = ml2.mass

            variables.loc[variables[flavor], "min_b1l_mass"] = variables.loc[variables[flavor], ["b1l1_mass", "b1l2_mass"]].min(axis=1)
            variables.loc[variables[flavor], "min_bl_mass"] = variables.loc[variables[flavor], ["b1l1_mass", "b1l2_mass"]].min(axis=1)

        lJet1Filtered = lJet1[lJet2Mask]
        lJet2Filtered = lJet2[lJet2Mask]

        lJet1Filtered["charge"] = 1 
        lJet2Filtered["charge"] = -1 

        lJet1P4s = ak.zip({
            "pt": lJet1Filtered.pt,
            "eta": lJet1Filtered.eta,
            "phi": lJet1Filtered.phi,
            "mass": lJet1Filtered.mass,
            "charge": lJet1Filtered.charge,
        }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

        lJet2P4s = ak.zip({
            "pt": lJet2Filtered.pt,
            "eta": lJet2Filtered.eta,
            "phi": lJet2Filtered.phi,
            "mass": lJet2Filtered.mass,
            "charge": lJet2Filtered.charge,
        }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)    

        llj2 = leptons[0][lJet2Mask] + leptons[1][lJet2Mask] + lJet2P4s
        lljj = leptons[0][lJet2Mask] + leptons[1][lJet2Mask] + lJet1P4s + lJet2P4s

        if len(llj2) > 0:
            for v in [
                "pt",
                "eta",
                "phi",
                "mass",
            ]:
                variables.loc[variables[flavor] & jet2Mask, f"bllj2_{v}"] = getattr(llj2, v)


        ml1 = leptons[0][lJet2Mask] + lJet2P4s
        ml2 = leptons[1][lJet2Mask] + lJet2P4s

        if len(ml1) > 0:

            variables.loc[variables[flavor] & jet2Mask, "b2l1_mass"] = ml1.mass
            variables.loc[variables[flavor] & jet2Mask, "b2l2_mass"] = ml2.mass

            variables.loc[variables[flavor], "min_b2l_mass"] = variables.loc[variables[flavor], ["b2l1_mass", "b2l2_mass"]].min(axis=1)
            variables.loc[variables[flavor], "min_bl_mass"] = variables.loc[variables[flavor],
                ["b1l1_mass", "b1l2_mass", "b2l1_mass", "b2l2_mass"]
            ].min(axis=1)

        if len(lljj) > 0:
            for v in [
                "pt",
                "eta",
                "phi",
                "mass",
                ]:
                    variables.loc[variables[flavor] & jet2Mask, f"blljj_{v}"] = getattr(lljj, v)


def jet_id(jets, parameters, year):
    pass_jet_id = np.ones_like(jets.jetId, dtype=bool)
    if "loose" in parameters["jet_id"]:
        pass_jet_id = jets.jetId >= 1
    elif "tight" in parameters["jet_id"]:
        if "2016" in year:
            pass_jet_id = jets.jetId >= 3
        else:
            pass_jet_id = jets.jetId >= 2
    return pass_jet_id


def jet_puid(jets, parameters, year):
    jet_puid_opt = parameters["jet_puid"]
    puId = jets.puId17 if year == "2017" else jets.puId
    jet_puid_wps = {
        "loose": (puId >= 4) | (jets.pt > 50),
        "medium": (puId >= 6) | (jets.pt > 50),
        "tight": (puId >= 7) | (jets.pt > 50),
    }
    pass_jet_puid = np.ones_like(jets.pt.values)
    if jet_puid_opt in ["loose", "medium", "tight"]:
        pass_jet_puid = jet_puid_wps[jet_puid_opt]
    elif "2017corrected" in jet_puid_opt:
        eta_window = (abs(jets.eta) > 2.6) & (abs(jets.eta) < 3.0)
        pass_jet_puid = (eta_window & (puId >= 7)) | (
            (~eta_window) & jet_puid_wps["loose"]
        )
    return pass_jet_puid


def gen_jet_pair_mass(df):
    gjmass = None
    gjets = df.GenJet
    gleptons = df.GenPart[
        (abs(df.GenPart.pdgId) == 13)
        | (abs(df.GenPart.pdgId) == 11)
        | (abs(df.GenPart.pdgId) == 15)
    ]
    gl_pair = ak.cartesian({"jet": gjets, "lepton": gleptons}, axis=1, nested=True)
    _, _, dr_gl = delta_r(
        gl_pair["jet"].eta,
        gl_pair["lepton"].eta,
        gl_pair["jet"].phi,
        gl_pair["lepton"].phi,
    )
    isolated = ak.all((dr_gl > 0.3), axis=-1)
    if ak.count(gjets[isolated], axis=None) > 0:
        # TODO: convert only relevant fields!
        gjet1 = ak.to_pandas(gjets[isolated]).loc[
            pd.IndexSlice[:, 0], ["pt", "eta", "phi", "mass"]
        ]
        gjet2 = ak.to_pandas(gjets[isolated]).loc[
            pd.IndexSlice[:, 1], ["pt", "eta", "phi", "mass"]
        ]
        gjet1.index = gjet1.index.droplevel("subentry")
        gjet2.index = gjet2.index.droplevel("subentry")

        gjsum = p4_sum(gjet1, gjet2)
        gjmass = gjsum.mass
    return gjmass


def fill_softjets(df, output, variables, cutoff):
    saj_df = ak.to_pandas(df.SoftActivityJet)
    saj_df["mass"] = 0.0
    nj_name = f"SoftActivityJetNjets{cutoff}"
    ht_name = f"SoftActivityJetHT{cutoff}"
    res = ak.to_pandas(df[[nj_name, ht_name]])

    res["to_correct"] = output.two_muons | (variables.njets > 0)
    _, _, dR_m1 = delta_r(saj_df.eta, output.mu1_eta, saj_df.phi, output.mu1_phi)
    _, _, dR_m2 = delta_r(saj_df.eta, output.mu2_eta, saj_df.phi, output.mu2_phi)
    _, _, dR_j1 = delta_r(
        saj_df.eta, variables.jet1_eta, saj_df.phi, variables.jet1_phi
    )
    _, _, dR_j2 = delta_r(
        saj_df.eta, variables.jet2_eta, saj_df.phi, variables.jet2_phi
    )
    saj_df["dR_m1"] = dR_m1 < 0.4
    saj_df["dR_m2"] = dR_m2 < 0.4
    saj_df["dR_j1"] = dR_j1 < 0.4
    saj_df["dR_j2"] = dR_j2 < 0.4
    dr_cols = ["dR_m1", "dR_m2", "dR_j1", "dR_j2"]
    saj_df[dr_cols] = saj_df[dr_cols].fillna(False)
    saj_df["to_remove"] = saj_df[dr_cols].sum(axis=1).astype(bool)

    saj_df_filtered = saj_df[(~saj_df.to_remove) & (saj_df.pt > cutoff)]
    footprint = saj_df[(saj_df.to_remove) & (saj_df.pt > cutoff)]
    res["njets_corrected"] = (
        saj_df_filtered.reset_index().groupby("entry")["subentry"].nunique()
    )
    res["njets_corrected"] = res["njets_corrected"].fillna(0).astype(int)
    res["footprint"] = footprint.pt.groupby(level=[0]).sum()
    res["footprint"] = res["footprint"].fillna(0.0)
    res["ht_corrected"] = res[ht_name] - res.footprint
    res.loc[res.ht_corrected < 0, "ht_corrected"] = 0.0

    res.loc[res.to_correct, nj_name] = res.loc[res.to_correct, "njets_corrected"]

    res.loc[res.to_correct, ht_name] = res.loc[res.to_correct, "ht_corrected"]

    variables[f"nsoftjets{cutoff}"] = res[f"SoftActivityJetNjets{cutoff}"]
    variables[f"htsoft{cutoff}"] = res[f"SoftActivityJetHT{cutoff}"]
