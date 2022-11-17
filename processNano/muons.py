import numpy as np
import awkward as ak
import math
from processNano.utils import p4_sum, delta_r, cs_variables
from processNano.corrections.muonMassResolution import smearMass
from processNano.corrections.muonMassScale import muonScaleUncert
from processNano.corrections.muonRecoUncert import muonRecoUncert

def fill_muonUncertainties(processor, output, mu1, mu2, is_mc, year, weights):

    # create numpy arrays for reco and gen mass needed for mass variations
    recoMassBB = output.loc[
        ((abs(output.l1_eta < 1.2)) & (abs(output.l2_eta < 1.2))), "dilepton_mass"
    ].to_numpy()
    recoMassBE = output.loc[
        ((abs(output.l1_eta > 1.2)) | (abs(output.l2_eta > 1.2))), "dilepton_mass"
    ].to_numpy()
    genMassBB = output.loc[
        ((abs(output.l1_eta < 1.2)) & (abs(output.l2_eta < 1.2))), "dilepton_mass_gen"
    ].to_numpy()
    genMassBE = output.loc[
        ((abs(output.l1_eta > 1.2)) | (abs(output.l2_eta > 1.2))), "dilepton_mass_gen"
    ].to_numpy()

    # apply additional mass smearing for MC events in the BE category
    if is_mc:
        output.loc[
            ((abs(output.l1_eta > 1.2)) | (abs(output.l2_eta > 1.2))), "dilepton_mass"
        ] = (
            output.loc[
                ((abs(output.l1_eta > 1.2)) | (abs(output.l2_eta > 1.2))),
                "dilepton_mass",
            ]
            * smearMass(genMassBE, year, bb=False, forUnc=False)
        ).values

    # calculate mass values smeared by mass resolution uncertainty
    output["dilepton_mass_resUnc"] = output.dilepton_mass.values
    if is_mc:

        output.loc[
            ((abs(output.l1_eta < 1.2)) & (abs(output.l2_eta < 1.2))),
            "dilepton_mass_resUnc",
        ] = (
            output.loc[
                ((abs(output.l1_eta < 1.2)) & (abs(output.l2_eta < 1.2))),
                "dilepton_mass_resUnc",
            ]
            * smearMass(genMassBB, year, bb=True)
        ).values
        output.loc[
            ((abs(output.l1_eta > 1.2)) | (abs(output.l2_eta > 1.2))),
            "dilepton_mass_resUnc",
        ] = (
            output.loc[
               ((abs(output.l1_eta > 1.2)) | (abs(output.l2_eta > 1.2))),
                "dilepton_mass_resUnc",
            ]
            * smearMass(genMassBE, year, bb=False)
        ).values

    # calculate mass values shifted by mass scale uncertainty
    output["dilepton_mass_scaleUncUp"] = output.dilepton_mass.values
    output["dilepton_mass_scaleUncDown"] = output.dilepton_mass.values
    if is_mc:
        output.loc[
            ((abs(output.l1_eta < 1.2)) & (abs(output.l2_eta < 1.2))),
            "dilepton_mass_scaleUncUp",
        ] = (
            output.loc[
                ((abs(output.l1_eta < 1.2)) & (abs(output.l2_eta < 1.2))),
                "dilepton_mass_scaleUncUp",
            ]
            * muonScaleUncert(recoMassBB, True, year)
        ).values
        output.loc[
            ((abs(output.l1_eta > 1.2)) | (abs(output.l2_eta > 1.2))),
            "dilepton_mass_scaleUncUp",
        ] = (
            output.loc[
                ((abs(output.l1_eta > 1.2)) | (abs(output.l2_eta > 1.2))),
                "dilepton_mass_scaleUncUp",
            ]
            * muonScaleUncert(recoMassBE, False, year)
        ).values
        output.loc[
            ((abs(output.l1_eta < 1.2)) & (abs(output.l2_eta < 1.2))),
            "dilepton_mass_scaleUncDown",
        ] = (
            output.loc[
                ((abs(output.l1_eta < 1.2)) & (abs(output.l2_eta < 1.2))),
                "dilepton_mass_scaleUncDown",
            ]
            * muonScaleUncert(recoMassBB, True, year, up=False)
        ).values
        output.loc[
            ((abs(output.l1_eta > 1.2)) | (abs(output.l2_eta > 1.2))),
            "dilepton_mass_scaleUncDown",
        ] = (
            output.loc[
                ((abs(output.l1_eta > 1.2)) | (abs(output.l2_eta > 1.2))),
                "dilepton_mass_scaleUncDown",
            ]
            * muonScaleUncert(recoMassBE, False, year, up=False)
        ).values

    # calculate event weights for muon reconstruction efficiency uncertainty
    eta1 = output["l1_eta"].to_numpy()
    eta2 = output["l2_eta"].to_numpy()
    pT1 = output["l1_pt"].to_numpy()
    pT2 = output["l2_pt"].to_numpy()
    mass = output["dilepton_mass"].to_numpy()
    isDimuon = output["isDimuon"].to_numpy()

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

def mass_resolution(is_mc, evaluator, df, year):
    # Returns absolute mass resolution!
    dpt1 = (df.l1_ptErr * df.dilepton_mass) / (2 * df.l1_pt)
    dpt2 = (df.l2_ptErr * df.dilepton_mass) / (2 * df.l2_pt)

    if is_mc:
        label = f"res_calib_MC_{year}"
    else:
        label = f"res_calib_Data_{year}"
    calibration = np.array(
        evaluator[label](
            df.l1_pt.values, abs(df.l1_eta.values), abs(df.l2_eta.values)
        )
    )

    return np.sqrt(dpt1 * dpt1 + dpt2 * dpt2) * calibration
