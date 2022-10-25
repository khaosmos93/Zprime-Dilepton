import numpy as np
import math
from processNano.utils import p4_sum, delta_r, cs_variables
from processNano.corrections.muonMassResolution import smearMass
from processNano.corrections.muonMassScale import muonScaleUncert
from processNano.corrections.muonRecoUncert import muonRecoUncert


def find_dimuon(objs, is_mc=False):
    is_mc = False

    objs1 = objs[objs.charge > 0]
    objs2 = objs[objs.charge < 0]
    objs1["mu_idx"] = objs1.index.to_numpy()
    objs2["mu_idx"] = objs2.index.to_numpy()
    dmass = 20.0

    for i in range(objs1.shape[0]):
        for j in range(objs2.shape[0]):
            px1_ = objs1.iloc[i].pt * np.cos(objs1.iloc[i].phi)
            py1_ = objs1.iloc[i].pt * np.sin(objs1.iloc[i].phi)
            pz1_ = objs1.iloc[i].pt * np.sinh(objs1.iloc[i].eta)
            e1_ = np.sqrt(px1_ ** 2 + py1_ ** 2 + pz1_ ** 2 + objs1.iloc[i].mass ** 2)
            px2_ = objs2.iloc[j].pt * np.cos(objs2.iloc[j].phi)
            py2_ = objs2.iloc[j].pt * np.sin(objs2.iloc[j].phi)
            pz2_ = objs2.iloc[j].pt * np.sinh(objs2.iloc[j].eta)
            e2_ = np.sqrt(px2_ ** 2 + py2_ ** 2 + pz2_ ** 2 + objs2.iloc[j].mass ** 2)
            m2 = (
                (e1_ + e2_) ** 2
                - (px1_ + px2_) ** 2
                - (py1_ + py2_) ** 2
                - (pz1_ + pz2_) ** 2
            )
            mass = math.sqrt(max(0, m2))

            if abs(mass - 91.1876) < dmass:
                dmass = abs(mass - 91.1876)
                obj1_selected = objs1.iloc[i]
                obj2_selected = objs2.iloc[j]
                idx1 = objs1.iloc[i].mu_idx
                idx2 = objs2.iloc[j].mu_idx

                dilepton_mass = mass
                if is_mc:
                    gpx1_ = objs1.iloc[i].pt_gen * np.cos(objs1.iloc[i].phi_gen)
                    gpy1_ = objs1.iloc[i].pt_gen * np.sin(objs1.iloc[i].phi_gen)
                    gpz1_ = objs1.iloc[i].pt_gen * np.sinh(objs1.iloc[i].eta_gen)
                    ge1_ = np.sqrt(
                        gpx1_ ** 2 + gpy1_ ** 2 + gpz1_ ** 2 + objs1.iloc[i].mass ** 2
                    )
                    gpx2_ = objs2.iloc[j].pt_gen * np.cos(objs2.iloc[j].phi_gen)
                    gpy2_ = objs2.iloc[j].pt_gen * np.sin(objs2.iloc[j].phi_gen)
                    gpz2_ = objs2.iloc[j].pt_gen * np.sinh(objs2.iloc[j].eta_gen)
                    ge2_ = np.sqrt(
                        gpx2_ ** 2 + gpy2_ ** 2 + gpz2_ ** 2 + objs2.iloc[j].mass ** 2
                    )
                    gm2 = (
                        (ge1_ + ge2_) ** 2
                        - (gpx1_ + gpx2_) ** 2
                        - (gpy1_ + gpy2_) ** 2
                        - (gpz1_ + gpz2_) ** 2
                    )
                    dilepton_mass_gen = math.sqrt(max(0, gm2))

    if dmass == 20:
        obj1 = objs1.loc[objs1.pt.idxmax()]
        obj2 = objs2.loc[objs2.pt.idxmax()]
        px1_ = obj1.pt * np.cos(obj1.phi)
        py1_ = obj1.pt * np.sin(obj1.phi)
        pz1_ = obj1.pt * np.sinh(obj1.eta)
        e1_ = np.sqrt(px1_ ** 2 + py1_ ** 2 + pz1_ ** 2 + obj1.mass ** 2)
        px2_ = obj2.pt * np.cos(obj2.phi)
        py2_ = obj2.pt * np.sin(obj2.phi)
        pz2_ = obj2.pt * np.sinh(obj2.eta)
        e2_ = np.sqrt(px2_ ** 2 + py2_ ** 2 + pz2_ ** 2 + obj2.mass ** 2)
        m2 = (
            (e1_ + e2_) ** 2
            - (px1_ + px2_) ** 2
            - (py1_ + py2_) ** 2
            - (pz1_ + pz2_) ** 2
        )
        mass = math.sqrt(max(0, m2))
        dilepton_mass = mass

        if is_mc:
            gpx1_ = obj1.pt_gen * np.cos(obj1.phi_gen)
            gpy1_ = obj1.pt_gen * np.sin(obj1.phi_gen)
            gpz1_ = obj1.pt_gen * np.sinh(obj1.eta_gen)
            ge1_ = np.sqrt(gpx1_ ** 2 + gpy1_ ** 2 + gpz1_ ** 2 + obj1.mass ** 2)
            gpx2_ = obj2.pt_gen * np.cos(obj2.phi_gen)
            gpy2_ = obj2.pt_gen * np.sin(obj2.phi_gen)
            gpz2_ = obj2.pt_gen * np.sinh(obj2.eta_gen)
            ge2_ = np.sqrt(gpx2_ ** 2 + gpy2_ ** 2 + gpz2_ ** 2 + obj2.mass ** 2)
            gm2 = (
                (ge1_ + ge2_) ** 2
                - (gpx1_ + gpx2_) ** 2
                - (gpy1_ + gpy2_) ** 2
                - (gpz1_ + gpz2_) ** 2
            )
            dilepton_mass_gen = math.sqrt(max(0, gm2))


        obj1_selected = obj1
        obj2_selected = obj2
        idx1 = objs1.pt.idxmax()
        idx2 = objs2.pt.idxmax()

        log1 = obj1_selected.to_numpy()
        log2 = obj2_selected.to_numpy()
        if log1[0] == -1 or log2[0] == -1:
            dilepton_mass_gen = -999.0

    if obj1_selected.pt > obj2_selected.pt:
        if is_mc:
            return [idx1, idx2, dilepton_mass, dilepton_mass_gen]
        else:
            return [idx1, idx2, dilepton_mass]
    else:
        if is_mc:
            return [idx2, idx1, dilepton_mass, dilepton_mass_gen]
        else:
            return [idx2, idx1, dilepton_mass]


def fill_muons(processor, output, mu1, mu2, is_mc, year, weights):
    mu1_variable_names = [
        "mu1_pt",
        "mu1_pt_gen",
        "mu1_pt_over_mass",
        "mu1_ptErr",
        "mu1_eta",
        "mu1_eta_gen",
        "mu1_phi",
        "mu1_phi_gen",
        "mu1_iso",
        "mu1_dxy",
        "mu1_dz",
        "mu1_genPartFlav",
        "mu1_ip3d",
        "mu1_sip3d",
    ]
    mu2_variable_names = [
        "mu2_pt",
        "mu2_pt_gen",
        "mu2_pt_over_mass",
        "mu2_ptErr",
        "mu2_eta",
        "mu2_eta_gen",
        "mu2_phi",
        "mu2_phi_gen",
        "mu2_iso",
        "mu2_dxy",
        "mu2_dz",
        "mu2_genPartFlav",
        "mu2_ip3d",
        "mu2_sip3d",
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
        # "dilepton_pt_gen",
        # "dilepton_eta_gen",
        # "dilepton_phi_gen",
        "dilepton_dEta",
        "dilepton_dPhi",
        "dilepton_dR",
        "dilepton_rap",
        "bbangle",
        "dilepton_cos_theta_cs",
        "dilepton_phi_cs",
        "wgt_nominal",
    ]
    v_names = mu1_variable_names + mu2_variable_names + dilepton_variable_names

    # Initialize columns for muon variables

    for n in v_names:
        output[n] = 0.0

    # Fill single muon variables
    mm = p4_sum(mu1, mu2, is_mc)
    for v in [
        "pt",
        "pt_gen",
        "ptErr",
        "eta",
        "eta_gen",
        "phi",
        "phi_gen",
        "dxy",
        "dz",
        "genPartFlav",
        "ip3d",
        "sip3d",
        "tkRelIso",
        "charge",
    ]:

        try:
            output[f"mu1_{v}"] = mu1[v]
            output[f"mu2_{v}"] = mu2[v]
        except Exception:
            output[f"mu1_{v}"] = -999.0
            output[f"mu2_{v}"] = -999.0

    for v in [
        "pt",
        "eta",
        "phi",
        "mass",
        "rap",
    ]:
        name = f"dilepton_{v}"
        try:
            output[name] = mm[v]
            output[name] = output[name].fillna(-999.0)
        except Exception:
            output[name] = -999.0

    # create numpy arrays for reco and gen mass needed for mass variations
    recoMassBB = output.loc[
        ((abs(output.mu1_eta < 1.2)) & (abs(output.mu2_eta < 1.2))), "dilepton_mass"
    ].to_numpy()
    recoMassBE = output.loc[
        ((abs(output.mu1_eta > 1.2)) | (abs(output.mu2_eta > 1.2))), "dilepton_mass"
    ].to_numpy()
    genMassBB = output.loc[
        ((abs(output.mu1_eta < 1.2)) & (abs(output.mu2_eta < 1.2))), "dilepton_mass_gen"
    ].to_numpy()
    genMassBE = output.loc[
        ((abs(output.mu1_eta > 1.2)) | (abs(output.mu2_eta > 1.2))), "dilepton_mass_gen"
    ].to_numpy()

    # apply additional mass smearing for MC events in the BE category
    if is_mc:
        output.loc[
            ((abs(output.mu1_eta > 1.2)) | (abs(output.mu2_eta > 1.2))), "dilepton_mass"
        ] = (
            output.loc[
                ((abs(output.mu1_eta > 1.2)) | (abs(output.mu2_eta > 1.2))),
                "dilepton_mass",
            ]
            * smearMass(genMassBE, year, bb=False, forUnc=False)
        ).values

    # calculate mass values smeared by mass resolution uncertainty
    output["dilepton_mass_resUnc"] = output.dilepton_mass.values
    if is_mc:

        output.loc[
            ((abs(output.mu1_eta < 1.2)) & (abs(output.mu2_eta < 1.2))),
            "dilepton_mass_resUnc",
        ] = (
            output.loc[
                ((abs(output.mu1_eta < 1.2)) & (abs(output.mu2_eta < 1.2))),
                "dilepton_mass_resUnc",
            ]
            * smearMass(genMassBB, year, bb=True)
        ).values
        output.loc[
            ((abs(output.mu1_eta > 1.2)) | (abs(output.mu2_eta > 1.2))),
            "dilepton_mass_resUnc",
        ] = (
            output.loc[
                ((abs(output.mu1_eta > 1.2)) | (abs(output.mu2_eta > 1.2))),
                "dilepton_mass_resUnc",
            ]
            * smearMass(genMassBE, year, bb=False)
        ).values

    # calculate mass values shifted by mass scale uncertainty
    output["dilepton_mass_scaleUncUp"] = output.dilepton_mass.values
    output["dilepton_mass_scaleUncDown"] = output.dilepton_mass.values
    if is_mc:
        output.loc[
            ((abs(output.mu1_eta < 1.2)) & (abs(output.mu2_eta < 1.2))),
            "dilepton_mass_scaleUncUp",
        ] = (
            output.loc[
                ((abs(output.mu1_eta < 1.2)) & (abs(output.mu2_eta < 1.2))),
                "dilepton_mass_scaleUncUp",
            ]
            * muonScaleUncert(recoMassBB, True, year)
        ).values
        output.loc[
            ((abs(output.mu1_eta > 1.2)) | (abs(output.mu2_eta > 1.2))),
            "dilepton_mass_scaleUncUp",
        ] = (
            output.loc[
                ((abs(output.mu1_eta > 1.2)) | (abs(output.mu2_eta > 1.2))),
                "dilepton_mass_scaleUncUp",
            ]
            * muonScaleUncert(recoMassBE, False, year)
        ).values
        output.loc[
            ((abs(output.mu1_eta < 1.2)) & (abs(output.mu2_eta < 1.2))),
            "dilepton_mass_scaleUncDown",
        ] = (
            output.loc[
                ((abs(output.mu1_eta < 1.2)) & (abs(output.mu2_eta < 1.2))),
                "dilepton_mass_scaleUncDown",
            ]
            * muonScaleUncert(recoMassBB, True, year, up=False)
        ).values
        output.loc[
            ((abs(output.mu1_eta > 1.2)) | (abs(output.mu2_eta > 1.2))),
            "dilepton_mass_scaleUncDown",
        ] = (
            output.loc[
                ((abs(output.mu1_eta > 1.2)) | (abs(output.mu2_eta > 1.2))),
                "dilepton_mass_scaleUncDown",
            ]
            * muonScaleUncert(recoMassBE, False, year, up=False)
        ).values

    # calculate event weights for muon reconstruction efficiency uncertainty
    eta1 = output["mu1_eta"].to_numpy()
    eta2 = output["mu2_eta"].to_numpy()
    pT1 = output["mu1_pt"].to_numpy()
    pT2 = output["mu2_pt"].to_numpy()
    mass = output["dilepton_mass"].to_numpy()
    isDimuon = output["two_muons"].to_numpy()

    recowgts = {}
    recowgts["nom"] = muonRecoUncert(
        mass, pT1, pT2, eta1, eta2, isDimuon, year, how="nom"
    )
    recowgts["up"] = muonRecoUncert(
        mass, pT1, pT2, eta1, eta2, isDimuon, year, how="up"
    )
    recowgts["down"] = muonRecoUncert(
        mass, pT1, pT2, eta1, eta2, isDimuon, year, how="down"
    )
    weights.add_weight("recowgt", recowgts, how="all")

    output["mu1_pt_over_mass"] = output.mu1_pt.values / output.dilepton_mass.values
    output["mu2_pt_over_mass"] = output.mu2_pt.values / output.dilepton_mass.values
    output["dilepton_pt_log"] = np.log(output.dilepton_pt[output.dilepton_pt > 0])
    output.loc[output.dilepton_pt < 0, "dilepton_pt_log"] = -999.0

    mm_deta, mm_dphi, mm_dr = delta_r(mu1.eta, mu2.eta, mu1.phi, mu2.phi)
    output["dilepton_pt"] = mm.pt
    output["dilepton_eta"] = mm.eta
    output["dilepton_phi"] = mm.phi
    output["dilepton_dEta"] = mm_deta
    output["dilepton_dPhi"] = mm_dphi
    output["dilepton_dR"] = mm_dr

    # output["dilepton_ebe_mass_res"] = mass_resolution(
    #    is_mc, processor.evaluator, output, processor.year
    # )
    # output["dilepton_ebe_mass_res_rel"] = output.dilepton_ebe_mass_res / output.dilepton_mass
    output["dilepton_cos_theta_cs"], output["dilepton_phi_cs"] = cs_variables(mu1, mu2)


def mass_resolution(is_mc, evaluator, df, year):
    # Returns absolute mass resolution!
    dpt1 = (df.mu1_ptErr * df.dilepton_mass) / (2 * df.mu1_pt)
    dpt2 = (df.mu2_ptErr * df.dilepton_mass) / (2 * df.mu2_pt)

    if is_mc:
        label = f"res_calib_MC_{year}"
    else:
        label = f"res_calib_Data_{year}"
    calibration = np.array(
        evaluator[label](
            df.mu1_pt.values, abs(df.mu1_eta.values), abs(df.mu2_eta.values)
        )
    )

    return np.sqrt(dpt1 * dpt1 + dpt2 * dpt2) * calibration
