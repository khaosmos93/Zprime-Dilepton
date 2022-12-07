import sys
sys.path.append("copperhead/")

from warnings import simplefilter, filterwarnings
filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=RuntimeWarning)

import awkward
import awkward as ak
import numpy as np
import numba

import pandas as pd
import coffea.processor as processor
from coffea.lumi_tools import LumiMask
from coffea.nanoevents.methods import candidate

from processNano.timer import Timer
from processNano.weights import Weights

# correction helpers included from copperhead
from copperhead.stage1.corrections.pu_reweight import pu_lookups, pu_evaluator
from copperhead.stage1.corrections.l1prefiring_weights import l1pf_weights

# from copperhead.stage1.corrections.lhe_weights import lhe_weights
# from copperhead.stage1.corrections.pdf_variations import add_pdf_variations

# high mass dilepton specific corrections
from processNano.corrections.kFac import kFac
from processNano.corrections.nnpdfWeight import NNPDFWeight
from copperhead.stage1.corrections.jec import jec_factories, apply_jec
from copperhead.config.jec_parameters import jec_parameters

from processNano.jets import prepare_jets, fill_jets, fill_bjets, initialize_bjet_var, btagSF_new  # btagSF

import copy

from processNano.leptons import fill_leptons, initialize_leptons
from processNano.muons import fill_muonUncertainties
from processNano.electrons import compute_eleScaleUncertainty
from processNano.utils import bbangle

from config.parameters import parameters, muon_branches, ele_branches, jet_branches

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

@numba.njit
def calc_mass(obj1, obj2):
    px = 0
    py = 0 
    pz = 0 
    e = 0
    for obj in [obj1, obj2]:
        px_ = obj.pt * np.cos(obj.phi)
        py_ = obj.pt * np.sin(obj.phi)
        pz_ = obj.pt * np.sinh(obj.eta)
        e_ = np.sqrt(px_**2 + py_**2 + pz_**2 + obj.mass**2)
        px += px_
        py += py_
        pz += pz_
        e += e_

    mass = np.sqrt(
        e**2 - px**2 - py**2 - pz**2
    )

    return mass

@numba.njit
def find_dilepton(events_leptons, builder):

    for leptons in events_leptons:
        idx1 = -1
        idx2 = -1
        dMass = 20.0
        maxM = 0 

        builder.begin_list()
        nlep = len(leptons)
        for i in range (0,nlep):
            for y in range (i+1,nlep):
                if leptons[i].charge + leptons[y].charge != 0: continue
                #if bbangle(leptons[i],leptons[y]) < parameters["3dangle"]: continue
                mass = calc_mass(leptons[i],leptons[y])
                if abs(mass - 91.1876) < dMass:
                    dMass = abs(mass - 91.1876)
                    idx1 = i
                    idx2 = y
                elif mass > maxM:
                    maxM = mass
                    idx1 = i
                    idx2 = y
        builder.begin_tuple(2)
        builder.index(0).integer(idx1)
        builder.index(1).integer(idx2)
        builder.end_tuple()
        builder.end_list()

    return builder


class DileptonProcessor(processor.ProcessorABC):
    def __init__(self, **kwargs):
        self.samp_info = kwargs.pop("samp_info", None)
        do_timer = kwargs.pop("do_timer", True)
        self.apply_to_output = kwargs.pop("apply_to_output", None)
        self.channel = kwargs.pop("channel","mu")
        self.pt_variations = kwargs.pop("pt_variations", ["nominal"])
 
        self.year = self.samp_info.year
        self.parameters = {k: v[self.year] for k, v in parameters.items()}

        self.do_btag = True

        if self.samp_info is None:
            print("Samples info missing!")
            return

        self._accumulator = processor.defaultdict_accumulator(int)

        self.applykFac = False
        self.applyNNPDFWeight = False
        self.do_pu = True
        self.auto_pu = False
        self.do_l1pw = True  # L1 prefiring weights
        self.do_jecunc = True
        self.do_jerunc = False

        self.timer = Timer("global") if do_timer else None

        self._columns = self.parameters["proc_columns"]

        self.regions = ["bb", "be"]

        self.lumi_weights = self.samp_info.lumi_weights

        self.prepare_lookups()

    def process(self, df):
        # Initialize timer
        if self.timer:
            self.timer.update()

        # Dataset name (see definitions in config/datasets.py)
        dataset = df.metadata["dataset"]

        is_mc = True
        if "data" in dataset:
            is_mc = False
        # ------------------------------------------------------------#
        # Apply HLT, lumimask, genweights, PU weights
        # and L1 prefiring weights
        # ------------------------------------------------------------#

        numevents = len(df)
        # All variables that we want to save
        # will be collected into the 'output' dataframe
        output = pd.DataFrame(
            {"run": df.run, "event": df.event, "luminosityBlock": df.luminosityBlock}
        )
        output.index.name = "entry"
        output["npv"] = df.PV.npvs
        output["met"] = df.MET.pt
        output["dataset"] = dataset
        output["year"] = int(self.year)

        # Separate dataframe to keep track on weights
        # and their systematic variations
        weights = Weights(output)

        output["event_selection"] = False
        output["isDimuon"] = False
        output["isDielectron"] = False
        output["r"] = None
        output["bbangle"] = -999.0 

        output = initialize_leptons(self, output)

        # HERE tmp
        if "data_2022" in dataset:
            if "Mu50" not in ak.fields(df.HLT):
                hlt_mu = ak.to_pandas(df.HLT)
                trigFilterMu = hlt_mu.sum(axis=1) < -1  # set to false
            else:
                hlt_mu = ak.to_pandas(df.HLT[["Mu50", "HighPtTkMu100", "CascadeMu100"]])
                trigFilterMu = hlt_mu[["Mu50", "HighPtTkMu100", "CascadeMu100"]].sum(axis=1) > 0
            if "DoubleEle25_CaloIdL_MW" not in ak.fields(df.HLT):
                hlt_el = ak.to_pandas(df.HLT)
                trigFilterEl = hlt_el.sum(axis=1) < -1  # set to false
            else:
                hlt_el= ak.to_pandas(df.HLT[["DoubleEle25_CaloIdL_MW"]])
                trigFilterEl = hlt_el[["DoubleEle25_CaloIdL_MW"]].sum(axis=1) > 0
        else:
            hlt_mu = ak.to_pandas(df.HLT[self.parameters["mu_hlt"]])
            trigFilterMu = hlt_mu[self.parameters["mu_hlt"]].sum(axis=1) > 0
            hlt_el= ak.to_pandas(df.HLT[self.parameters["el_hlt"]])
            trigFilterEl = hlt_el[self.parameters["el_hlt"]].sum(axis=1) > 0
        output["isDielectron"] = trigFilterEl
        output["isDimuon"] = trigFilterMu

        ### Set additional lepton variables:
        df["Muon", "pt_raw"] = df.Muon.pt
        df["Muon", "eta_raw"] = df.Muon.eta
        df["Muon", "phi_raw"] = df.Muon.phi
        if is_mc:
            df["Muon", "pt_gen"] = df.Muon.matched_gen.pt
            df["Muon", "eta_gen"] = df.Muon.matched_gen.eta
            df["Muon", "phi_gen"] = df.Muon.matched_gen.phi
            df["Muon", "idx"] = df.Muon.genPartIdx

        muons = df.Muon
        electrons = df.Electron

        muons = muons[
            (muons.pt > self.parameters["muon_pt_cut"])
            & (abs(muons.eta) < self.parameters["muon_eta_cut"])
            & (muons.tkRelIso < self.parameters["muon_iso_cut"])
            & (muons[self.parameters["muon_id"]] > 0)
            & (muons.dxy < self.parameters["muon_dxy"])
            & (
               (muons.ptErr / muons.pt)
                < self.parameters["muon_ptErr/pt"]
               )
        ]
        output["isDimuon"] = output["isDimuon"] & (ak.num(muons) >= 2)

        electrons = electrons[
            (electrons.pt > self.parameters["electron_pt_cut"])
            & (abs(electrons.eta) < self.parameters["electron_eta_cut"])
            & (electrons[self.parameters["electron_id"]] > 0)
        ] 
        output["isDielectron"] = output["isDielectron"] & (ak.num(electrons) >= 2)

 
        '''
           Cleaning of the overlap between the dimuon and dielectron channel
           The overlapping events get identified, and the same logic as for finding the best pair within a channel is used for the disambiguation
           Finally, the isDielectron or isDimuon flags in the output dataframe are adjusted to keep only one of the pairs
        '''

        cutMuon = ak.num(df.Muon[
            (df.Muon.pt > self.parameters["muon_pt_cut"])
            & (abs(df.Muon.eta) < self.parameters["muon_eta_cut"])
            & (df.Muon.tkRelIso < self.parameters["muon_iso_cut"])
            & (df.Muon[self.parameters["muon_id"]] > 0)
            & (df.Muon.dxy < self.parameters["muon_dxy"])
            & (
               (df.Muon.ptErr / df.Muon.pt)
                < self.parameters["muon_ptErr/pt"]
               )
        ]) >= 2

        cutElectron = ak.num(df.Electron[
            (df.Electron.pt > self.parameters["electron_pt_cut"])
            & (abs(df.Electron.eta) < self.parameters["electron_eta_cut"])
            & (df.Electron[self.parameters["electron_id"]] > 0)
        ]) >= 2

        overlapEvents = df[cutMuon & cutElectron & trigFilterMu & trigFilterEl]

        if len(overlapEvents) > 0:
            muonsOverlap = overlapEvents.Muon
            electronsOverlap = overlapEvents.Electron

            muonsOverlap = ak.zip({
                "pt": muonsOverlap.pt,
                "eta": muonsOverlap.eta,
                "phi": muonsOverlap.phi,
                "mass": muonsOverlap.mass,
                "charge": muonsOverlap.charge,
            }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)

            electronsOverlap = ak.zip({
                "pt": electronsOverlap.pt,
                "eta": electronsOverlap.eta,
                "phi": electronsOverlap.phi,
                "mass": electronsOverlap.mass,
                "charge": electronsOverlap.charge,
            }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)
            
            #Somehow have to make sure that the VirtualArrays have been materialized so that the find_dilepton function doesn't crash
            muonsOverlap = ak.materialized(muonsOverlap)
            electronsOverlap = ak.materialized(electronsOverlap)

            selectedIndicesMuOverlap = find_dilepton(muonsOverlap, ak.ArrayBuilder()).snapshot()
            selectedIndicesElOverlap = find_dilepton(electronsOverlap, ak.ArrayBuilder()).snapshot()
            dimuonOverlap = [muonsOverlap[selectedIndicesMuOverlap[idx]] for idx in "01"]
            dielectronOverlap = [electronsOverlap[selectedIndicesElOverlap[idx]] for idx in "01"]

            dimuonP4Overlap = dimuonOverlap[0] + dimuonOverlap[1]
            dielectronP4Overlap = dielectronOverlap[0] + dielectronOverlap[1]
      
            tempMask = output["isDimuon"] & output["isDielectron"]

            isDimuonWithinZwindow = (abs(dimuonP4Overlap.mass - 91.1876) < 20)
            isDielectronWithinZwindow = (abs(dielectronP4Overlap.mass - 91.1876) < 20)

            isDimuonOverlap= (
                (  # both in Z window -> choose closer one to the Z mass
                    ((isDimuonWithinZwindow) & (isDielectronWithinZwindow))
                    & (abs(dimuonP4Overlap.mass - 91.1876) < abs(dielectronP4Overlap.mass - 91.1876))
                ) | (  # both not in Z window -> choose higher mass
                    ((~isDimuonWithinZwindow) & (~isDielectronWithinZwindow))
                    & (dimuonP4Overlap.mass > dielectronP4Overlap.mass)
                ) | (  # only dimuon in Z window -> choose dimuon
                    ((isDimuonWithinZwindow) & (~isDielectronWithinZwindow))
                    # & True
                )
                #  | (  # only dielectron in Z window -> choose dielectron
                #     ((~isDimuonWithinZwindow) & (isDielectronWithinZwindow)) &
                #     False
                # )
            )

            output.loc[tempMask, "isDimuon"] = isDimuonOverlap
            output.loc[tempMask, "isDielectron"] = ~isDimuonOverlap

        #data events have to come from the correct PD
        if not is_mc and "El" in dataset:
            output["isDimuon"] = False 

        if not is_mc and "Mu" in dataset:
            output["isDielectron"] = False 

        if self.timer:
            self.timer.add_checkpoint("Electron/Muon disambiguation")

        muons = muons[output['isDimuon']]
        muons = muons[(ak.num(muons) >= 2)]

        muonP4s = ak.zip({
            "pt": muons.pt,
            "eta": muons.eta,
            "phi": muons.phi,
            "mass": muons.mass,
            "charge": muons.charge,
        }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)
        #Somehow have to make sure that the VirtualArrays have been materialized so that the find_dilepton function doesn't crash
        muonP4s = ak.materialized(muonP4s)

        selectedIndicesMu = find_dilepton(muonP4s, ak.ArrayBuilder()).snapshot()
        
        electrons = electrons[output['isDielectron']]
        electrons = electrons[(ak.num(electrons) >= 2)]

        electronP4s = ak.zip({
            "pt": electrons.pt,
            "eta": electrons.eta,
            "phi": electrons.phi,
            "mass": electrons.mass,
            "charge": electrons.charge,
        }, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)
        electronP4s = ak.materialized(electronP4s)

        selectedIndicesEl = find_dilepton(electronP4s, ak.ArrayBuilder()).snapshot()

        if self.timer:
            self.timer.add_checkpoint("Lepton pair finding")


        if len(muonP4s) > 0:
            muonP4s = [muonP4s[selectedIndicesMu[idx]] for idx in "01"]
            dimuonP4 = muonP4s[0] + muonP4s[1]
            selectedMuons = [muons[selectedIndicesMu[idx]] for idx in "01"]
            mu1 = selectedMuons[0]
            mu2 = selectedMuons[1]
            output = fill_leptons(self, output, dimuonP4, mu1, mu2, is_mc, self.year, "isDimuon")
            output.loc[output["isDimuon"], "bbangle"] = muonP4s[0].pvec.dot(muonP4s[1].pvec) / (muonP4s[0].pvec.p * muonP4s[1].pvec.p)
        else:
            mu1 = ak.Array([])
            mu2 = ak.Array([])
            dimuonP4 = ak.Array([])

        if len(electronP4s) > 0:
            electronP4s = [electronP4s[selectedIndicesEl[idx]] for idx in "01"]
            dielectronP4 = electronP4s[0] + electronP4s[1]        
            selectedElectrons = [electrons[selectedIndicesEl[idx]] for idx in "01"]
            el1 = selectedElectrons[0]
            el2 = selectedElectrons[1]
            output = fill_leptons(self, output, dielectronP4, el1, el2, is_mc, self.year, "isDielectron")
        else:
            el1 = ak.Array([])
            el2 = ak.Array([])
            dielectronP4 = ak.Array([])


        if self.timer:
            self.timer.add_checkpoint("Fill lepton variables")

        if is_mc:
            # For MC: Apply gen.weights, pileup weights, lumi weights,
            # L1 prefiring weights
            mask = np.ones(numevents, dtype=bool)
            genweight = df.genWeight
            weights.add_weight("genwgt", genweight)
            weights.add_weight("lumi", self.lumi_weights[dataset])
            output["wgt_raw_gen"] = genweight
            output["wgt_raw_lumi"] = self.lumi_weights[dataset]
            if self.do_pu:
                pu_wgts = pu_evaluator(
                    self.pu_lookups,
                    self.parameters,
                    numevents,
                    np.array(df.Pileup.nTrueInt),
                    self.auto_pu,
                )
                weights.add_weight("pu_wgt", pu_wgts, how="all")
                output["wgt_raw_pu"] = pu_wgts["nom"]
            if self.do_l1pw:
                if "L1PreFiringWeight" in df.fields:
                    l1pfw = l1pf_weights(df)
                    weights.add_weight("l1prefiring_wgt", l1pfw, how="all")
                    output["wgt_raw_l1pf"] = l1pfw["nom"]
                else:
                    weights.add_weight("l1prefiring_wgt", how="dummy_vars")
                    output["wgt_raw_l1pf"] = 1.
        else:
            # For Data: apply Lumi mask
            if self.channel == "mu":
                lumi_info = LumiMask(self.parameters["lumimask_UL_mu"])
            else:
                lumi_info = LumiMask(self.parameters["lumimask_UL_el"])
            mask = lumi_info(df.run, df.luminosityBlock)

        # Apply HLT to both Data and MC
        # this is obsolete
        # if self.channel == "mu":
        #     hlt = ak.to_pandas(df.HLT[self.parameters["mu_hlt"]])
        #     hlt = hlt[self.parameters["mu_hlt"]].sum(axis=1)
        # else:
        #     hlt = ak.to_pandas(df.HLT[self.parameters["el_hlt"]])
        #     hlt = hlt[self.parameters["el_hlt"]].sum(axis=1)



        good_pv = ak.to_pandas(df.PV).npvsGood > 0
        flags = ak.to_pandas(df.Flag)
        flags = flags[self.parameters["event_flags"]].product(axis=1)

        # Define baseline event selection
        if not is_mc and "Muon" in dataset:
            output["isDielectron"] = False
        if not is_mc and ("EGamma" in dataset or "DoubleEG" in dataset):
            output["isDimuon"] = False
 
        hasPair = output["isDimuon"] | output["isDielectron"]

        output["event_selection"] = (
            mask
            & hasPair
            & (flags > 0)
            & good_pv
        )
        if self.timer:
            self.timer.add_checkpoint("Applied HLT and lumimask and other event flags")

        # ------------------------------------------------------------#
        # Prepare jets
        # ------------------------------------------------------------#
        prepare_jets(df, is_mc)

        # ------------------------------------------------------------#
        # Apply JEC, get JEC and JER variations
        # ------------------------------------------------------------#

        jets = df.Jet

        muonsForCleaning = ak.mask(df.Muon,cutMuon)        
        electronsForCleaning = ak.mask(df.Electron,cutElectron)        

        closestsMuons = jets.nearest(muonsForCleaning, threshold=0.4)
        jets = jets[ak.is_none(closestsMuons, axis=1)]
        closestsElectrons = jets.nearest(electronsForCleaning, threshold=0.4)
        jets = jets[ak.is_none(closestsElectrons, axis=1)]

        output.columns = pd.MultiIndex.from_product(
            [output.columns, [""]], names=["Variable", "Variation"]
        )

        if self.timer:
            self.timer.add_checkpoint("Jet preparation & event weights")
        
        for v_name in self.pt_variations:
            output_updated = self.jet_loop(
                v_name,
                is_mc,
                df,
                dataset,
                mask,
                mu1,
                mu2,
                el1,
                el2,
                jets,
                jet_branches,
                weights,
                numevents,
                output,
            )
            if output_updated is not None:
                output = output_updated
        
        if self.timer:
            self.timer.add_checkpoint("Jet loop")

        if self.timer:
            self.timer.add_checkpoint("Computed event weights")


        
        # ------------------------------------------------------------#
        # Fill outputs
        # ------------------------------------------------------------#
        output["r"] = None
        output.loc[
            (output["isDimuon"]) & ((abs(output.l1_eta) < 1.2) & (abs(output.l2_eta) < 1.2)), "r"
        ] = "bb"
        output.loc[
            (output["isDimuon"]) & ((abs(output.l1_eta) > 1.2) | (abs(output.l2_eta) > 1.2)), "r"
        ] = "be"
        output.loc[
            (output["isDielectron"]) & ((abs(output.l1_eta) < 1.442) & (abs(output.l2_eta) < 1.442)), "r"
        ] = "bb"
        output.loc[
            (output["isDielectron"]) & (((abs(output.l1_eta) < 1.442) & (abs(output.l2_eta) > 1.566)) | (((abs(output.l1_eta) > 1.566) & (abs(output.l2_eta) < 1.442)))), "r"
        ] = "be"
        output.loc[
            (output["isDielectron"]) & ((abs(output.l1_eta) > 1.566) & (abs(output.l2_eta) > 1.566)), "r"
        ] = "ee"
        output.loc[
            (output["isDielectron"]) & (((abs(output.l1_eta) > 1.442) & (abs(output.l1_eta) < 1.566)) | (((abs(output.l2_eta) > 1.442) & (abs(output.l2_eta) < 1.566)))), "r"
        ] = "gap"

        output["year"] = int(self.year)
        for wgt in weights.df.columns:
            if wgt == "pu_wgt_off":
                output["pu_wgt"] = weights.get_weight(wgt)
            if wgt != "nominal":
                output[f"wgt_{wgt}"] = weights.get_weight(wgt)
        
        if is_mc and "dy" in dataset and self.applykFac:
            mass_bb = output[output["r"] == "bb"].dilepton_mass_gen.to_numpy()
            mass_be = output[output["r"] == "be"].dilepton_mass_gen.to_numpy()
            for key in output.columns:
                if "wgt" not in key[0]:
                    continue
                output.loc[
                    output["r"] == "bb",
                    key[0],
                ] = (
                    output.loc[
                        output["r"] == "bb",
                        key[0],
                    ]
                    * kFac(mass_bb, "bb", self.channel)
                ).values
                output.loc[
                    output["r"] == "be",
                    key[0],
                ] = (
                    output.loc[
                        output["r"] == "be",
                        key[0],
                    ]
                    * kFac(mass_be, "be", self.channel)
                ).values
        
        if is_mc and "dy" in dataset and self.applyNNPDFWeight:
            mass_bb = output[output["r"] == "bb"].dilepton_mass_gen.to_numpy()
            mass_be = output[output["r"] == "be"].dilepton_mass_gen.to_numpy()
            if self.channel == "mu":
                leadingPt_bb = output[output["r"] == "bb"].l1_pt_gen.to_numpy()
                leadingPt_be = output[output["r"] == "be"].l1_pt_gen.to_numpy()
            else:
                leadingPt_bb = output[output["r"] == "bb"].l1_pt_gen.to_numpy()
                leadingPt_be = output[output["r"] == "be"].l1_pt_gen.to_numpy()
 
            for key in output.columns:
                if "wgt" not in key[0]:
                    continue
                output.loc[
                    output["r"] == "bb",
                    key[0],
                ] = (
                    output.loc[
                        output["r"] == "bb",
                        key[0],
                    ]
                    * NNPDFWeight(
                        mass_bb, leadingPt_bb, "bb", self.channel, float(self.year), DY=True
                    )
                ).values
                output.loc[
                    output["r"] == "be",
                    key[0],
                ] = (
                    output.loc[
                        output["r"] == "be",
                        key[0],
                    ]
                    * NNPDFWeight(
                        mass_be, leadingPt_be, "be", self.channel, float(self.year), DY=True
                    )
                ).values

        if is_mc and "ttbar" in dataset and self.applyNNPDFWeight:
            mass_bb = output[output["r"] == "bb"].dilepton_mass_gen.to_numpy()
            mass_be = output[output["r"] == "be"].dilepton_mass_gen.to_numpy()
            if self.channel == "mu":
                leadingPt_bb = output[output["r"] == "bb"].l1_pt_gen.to_numpy()
                leadingPt_be = output[output["r"] == "be"].l1_pt_gen.to_numpy()
            else:
                leadingPt_bb = output[output["r"] == "bb"].l1_pt_gen.to_numpy()
                leadingPt_be = output[output["r"] == "be"].l1_pt_gen.to_numpy()
 
            for key in output.columns:
                if "wgt" not in key[0]:
                    continue
                output.loc[
                    output["r"] == "bb",
                    key[0],
                ] = (
                    output.loc[
                        output["r"] == "bb",
                        key[0],
                    ]
                    * NNPDFWeight(
                        mass_bb, leadingPt_bb, "bb", self.channel, float(self.year), DY=False
                    )
                ).values
                output.loc[
                    output["r"] == "be",
                    key[0],
                ] = (
                    output.loc[
                        output["r"] == "be",
                        key[0],
                    ]
                    * NNPDFWeight(
                        mass_be, leadingPt_be, "be", self.channel, float(self.year), DY=False
                    )
                ).values


        output["dilepton_mass_resUnc"] = output.dilepton_mass.values
        output["dilepton_mass_scaleUncUp"] = output.dilepton_mass.values
        output["dilepton_mass_scaleUncDown"] = output.dilepton_mass.values
        if len(muonP4s) > 0:
            fill_muonUncertainties(self, output, mu1, mu2, is_mc, self.year, weights) 
        if len(electronP4s) > 0:
            output = compute_eleScaleUncertainty(output, el1, el2)

        if self.timer:
            self.timer.add_checkpoint("Corrections and uncertainties")

        output = output.loc[output["event_selection"], :]
        output = output.reindex(sorted(output.columns), axis=1)
        output = output[output.r.isin(self.regions)]
        output.columns = output.columns.droplevel("Variation")
        if self.timer:
            self.timer.add_checkpoint("Filled outputs")
            self.timer.summary()

        if self.apply_to_output is None:
            return output
        else:
            self.apply_to_output(output)
            return self.accumulator.identity()

    def jet_loop(
        self,
        variation,
        is_mc,
        df,
        dataset,
        mask,
        mu1,
        mu2,
        el1,
        el2,
        jets,
        jet_branches,
        weights,
        numevents,
        output,
    ):

        if not is_mc and variation != "nominal":
            return

        variables = pd.DataFrame(index=output.index)
        variables["isDimuon"] = output["isDimuon"]
        variables["isDielectron"] = output["isDielectron"]
        variables = initialize_bjet_var(variables)

        jet_branches_local = copy.copy(jet_branches)

        if is_mc:
            jets["pt_gen"] = jets.matched_gen.pt
            jets["eta_gen"] = jets.matched_gen.eta
            jets["phi_gen"] = jets.matched_gen.phi

            jet_branches_local += [
                "partonFlavour",
                "hadronFlavour",
                "pt_gen",
                "eta_gen",
                "phi_gen",
            ]

        if self.timer:
            self.timer.add_checkpoint("Clean jets from matched leptons")

        preselection = (jets.pt > 30.0) & (abs(jets.eta) < 2.4) & (jets.jetId >= 2)

        if self.do_btag:
            if is_mc:
                jets, _ = btagSF_new(jets, self.year, correction="shape", is_UL=True)
                jets, btag_wp_wgt = btagSF_new(jets, self.year, correction="wp", is_UL=True)
                variables["wgt_nominal"] = (
                    btag_wp_wgt["central"]["wgt"]
                    # ak.prod(jets[preselection]["btag_sf_wp"], axis=1)
                )
                variables["wgt_nominal"] = variables["wgt_nominal"].fillna(1.0)
                variables["wgt_raw_btag"] = variables["wgt_nominal"]
                variables["wgt_nominal"] = variables[
                    "wgt_nominal"
                ] * weights.get_weight("nominal")
                variables["wgt_btag_up"] = (
                    btag_wp_wgt["up"]["wgt"]
                    # ak.prod(jets[preselection]["btag_sf_wp_up"], axis=1)
                )
                variables["wgt_btag_up"] = variables["wgt_btag_up"].fillna(1.0)
                variables["wgt_btag_up"] = variables[
                    "wgt_btag_up"
                ] * weights.get_weight("nominal")
                variables["wgt_btag_down"] = (
                    btag_wp_wgt["down"]["wgt"]
                    # ak.prod(jets[preselection]["btag_sf_wp_down"], axis=1)
                )
                variables["wgt_btag_down"] = variables["wgt_btag_down"].fillna(1.0)
                variables["wgt_btag_down"] = variables[
                    "wgt_btag_down"
                ] * weights.get_weight("nominal")

            else:
                variables["wgt_nominal"] = 1.0
                variables["wgt_raw_btag"] = 1.0
                variables["wgt_btag_up"] = 1.0
                variables["wgt_btag_down"] = 1.0
        else:
            if is_mc:
                variables["wgt_nominal"] = 1.0
                variables["wgt_raw_btag"] = 1.0
                variables["wgt_nominal"] = variables[
                    "wgt_nominal"
                ] * weights.get_weight("nominal")

            else:
                variables["wgt_nominal"] = 1.0

        jets = jets[(jets.pt > 30.0) & (abs(jets.eta) < 2.4) & (jets.jetId >= 2)]
        jets = jets[(~ak.is_none(jets, axis=1))]
        njets = ak.num(jets, axis=1)
        variables["njets"] = njets

        bjets = jets[jets.btagDeepFlavB > parameters["UL_btag_medium"][self.year]]
        bjets = bjets[(~ak.is_none(bjets, axis=1))]
        nbjets = ak.num(bjets, axis=1)
        variables["nbjets"] = nbjets

        jets = ak.pad_none(jets, 2, axis=1)
        jet1 = jets[:, 0]
        jet2 = jets[:, 1]
        Jets = [jet1, jet2]

        muons = [mu1, mu2]
        electrons = [el1, el2]

        fill_jets(output, variables, Jets, njets, muons, electrons, is_mc=is_mc)

        bjets = ak.pad_none(bjets, 2, axis=1)
        bjet1 = bjets[:, 0]
        bjet2 = bjets[:, 1]
        bjets = [bjet1, bjet2]

        fill_bjets(output, variables, bjets, muons, "isDimuon", is_mc=is_mc)
        fill_bjets(output, variables, bjets, electrons, "isDielectron", is_mc=is_mc)

        if self.timer:
            self.timer.add_checkpoint("Filled jet variables")

        # --------------------------------------------------------------#
        # Fill outputs
        # --------------------------------------------------------------#
        # All variables are affected by jet pT because of jet selections:
        # a jet may or may not be selected depending on pT variation.

        for key, val in variables.items():
            output.loc[:, key] = val

        del df
        del jets
        del bjets

        if is_mc:
            del btag_wp_wgt

        return output


    def prepare_lookups(self):
        # Pile-up reweighting
        self.pu_lookups = pu_lookups(self.parameters)

        return

    @property
    def accumulator(self):
        return processor.defaultdict_accumulator(int)

    @property
    def muoncolumns(self):
        return muon_branches

    @property
    def jetcolumns(self):
        return jet_branches

    def postprocess(self, accumulator):
        return accumulator
