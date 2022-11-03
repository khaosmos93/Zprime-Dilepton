import sys

sys.path.append("copperhead/")
import glob
import tqdm
import argparse
import dask
from dask.distributed import Client
import dask.dataframe as dd

from copperhead.python.io import load_dataframe
from doAnalysis.postprocessor import process_partitions

from copperhead.config.mva_bins import mva_bins
from config.variables import variables_lookup

__all__ = ["dask"]


parser = argparse.ArgumentParser()
parser.add_argument(
    "-y", "--years", nargs="+", help="Years to process", default=["2018"]
)
parser.add_argument(
    "-i",
    "--ip",
    dest="ip",
    default=None,
    action="store",
    help="Cluster ip:port (if not specified, " "will create a local cluster)",
)
args = parser.parse_args()

# Dask client settings
use_local_cluster = args.ip is None
node_ip = ""  # "129.13.101.196"

if use_local_cluster:
    ncpus_local = 24
    cluster_ip = ""
    dashboard_address = None  # f"{node_ip}:34875"
else:
    cluster_ip = f"{args.ip}"
    dashboard_address = None  # f"{node_ip}:8787"

# global parameters
parameters = {
    # < general settings >
    "cluster_ip": cluster_ip,
    "global_path": "/work/moh/DileptonBjets/Zprime-Dilepton/output/",
    "years": args.years,
    "label": "v03",
    "channels": [
        "inclusive",
        # "0b0j",
        # "0b1j",
        # "0b2j",
        "0b",
        "1b",
        "2b"
    ],
    "regions": [
        "inclusive",
        # "bb",
        # "be"
    ],
    "syst_variations": [
        "nominal",
        # "btag_up",
        # "btag_down",
        # "recowgt_up",
        # "recowgt_down",
        # "resUnc",
        # "scaleUncUp",
        # "scaleUncDown",
    ],
    # "custom_npartitions": {
    #     "vbf_powheg_dipole": 1,
    # },

    # < settings for histograms >
    "hist_vars": [
        "min_bl_mass",
        "mu1_pt",
        "mu1_eta",
        "mu1_phi",
        "bjet1_pt",
        "bjet1_eta",
        "bjet1_phi",
        "dimuon_mass",
        "dimuon_mass_gen",
        "njets",
        "nbjets",
        "dimuon_cos_theta_cs",
    ],
    "hist_vars_2d": [],  # [["dimuon_mass", "met"]],
    "variables_lookup": variables_lookup,
    "save_hists": True,
    #
    # < settings for unbinned output>
    "tosave_unbinned": {
        "bb": ["dimuon_mass", "event", "wgt_nominal"],
        "be": ["dimuon_mass", "event", "wgt_nominal"],
    },
    "save_unbinned": False,

    # < MVA settings >
    # "models_path": "data/trained_models/",
    # "dnn_models": {},
    # "bdt_models": {},
    # "mva_bins_original": mva_bins,
}
parameters["datasets"] = [
    "data_A",
    "data_B",
    "data_C",
    "data_D",

    "dy_M50",
    "dy_M50_incl",

    "dy0J_M50",
    "dy1J_M50",
    "dy2J_M50",

    "dy0J_M50_incl",
    "dy1J_M50_incl",
    "dy2J_M50_incl",

    "dy0J_M200to400",
    "dy0J_M400to800",
    "dy0J_M800to1400",
    "dy0J_M1400to2300",
    "dy0J_M2300to3500",
    "dy0J_M3500to4500",
    "dy0J_M4500to6000",
    "dy0J_M6000toInf",
    "dy1J_M200to400",
    "dy1J_M400to800",
    "dy1J_M800to1400",
    "dy1J_M1400to2300",
    "dy1J_M2300to3500",
    "dy1J_M3500to4500",
    "dy1J_M4500to6000",
    "dy1J_M6000toInf",
    "dy2J_M200to400",
    "dy2J_M400to800",
    "dy2J_M800to1400",
    "dy2J_M1400to2300",
    "dy2J_M2300to3500",
    "dy2J_M3500to4500",
    "dy2J_M4500to6000",
    "dy2J_M6000toInf",

    "dyInclusive50",
    "Wantitop",
    "tW",

    "ttbar_lep_inclusive",
    "ttbar_lep_M500to800",
    "ttbar_lep_M800to1200",
    "ttbar_lep_M1200to1800",
    "ttbar_lep_M1800toInf",

    "WWinclusive",
    "WW200to600",
    "WW600to1200",
    "WW1200to2500",
    "WW2500toInf",

    "WZ1L1Nu2Q",
    "WZ2L2Q",
    "WZ3LNu",
    "ZZ2L2Nu",
    "ZZ2L2Q",
    "ZZ4L",

    "ttbar_lep_inclusive_nocut",
    "WWinclusive_nocut",

    "bbll_6TeV_M1300To2000_negLL",
    "bbll_6TeV_M2000ToInf_negLL",
    "bbll_6TeV_M300To800_negLL",
    "bbll_6TeV_M800To1300_negLL",
    "bbll_10TeV_M1300To2000_negLL",
    "bbll_10TeV_M2000ToInf_negLL",
    "bbll_10TeV_M300To800_negLL",
    "bbll_10TeV_M800To1300_negLL",

    # "bbll_14TeV_M1300To2000_negLL",
    # "bbll_14TeV_M2000ToInf_negLL",
    # "bbll_14TeV_M300To800_negLL",
    # "bbll_14TeV_M800To1300_negLL",
    # "bbll_18TeV_M1300To2000_negLL",
    # "bbll_18TeV_M2000ToInf_negLL",
    # "bbll_18TeV_M300To800_negLL",
    # "bbll_18TeV_M800To1300_negLL",
    # "bbll_22TeV_M1300To2000_negLL",
    # "bbll_22TeV_M2000ToInf_negLL",
    # "bbll_22TeV_M300To800_negLL",
    # "bbll_22TeV_M800To1300_negLL",
    # "bbll_26TeV_M1300To2000_negLL",
    # "bbll_26TeV_M2000ToInf_negLL",
    # "bbll_26TeV_M300To800_negLL",
    # "bbll_26TeV_M800To1300_negLL",

    # "bbll_4TeV_M400_negLR",
    # "bbll_4TeV_M400_posLL",
    # "bbll_4TeV_M400_posLR",
    # "bbll_8TeV_M1000_negLL",
    # "bbll_8TeV_M1000_negLR",
    # "bbll_8TeV_M1000_posLL",
    # "bbll_8TeV_M1000_posLR",
    # "bbll_8TeV_M400_negLL",
    # "bbll_8TeV_M400_negLR",
    # "bbll_8TeV_M400_posLL",
    # "bbll_8TeV_M400_posLR",

    # "bsll_lambda1TeV_M200to500",
    # "bsll_lambda1TeV_M500to1000",
    # "bsll_lambda1TeV_M1000to2000",
    # "bsll_lambda1TeV_M2000toInf",
    # "bsll_lambda2TeV_M200to500",
    # "bsll_lambda2TeV_M500to1000",
    # "bsll_lambda2TeV_M1000to2000",
    # "bsll_lambda2TeV_M2000toInf",
    # "bsll_lambda4TeV_M200to500",
    # "bsll_lambda4TeV_M500to1000",
    # "bsll_lambda4TeV_M1000to2000",
    # "bsll_lambda4TeV_M2000toInf",
    # "bsll_lambda8TeV_M200to500",
    # "bsll_lambda8TeV_M500to1000",
    # "bsll_lambda8TeV_M1000to2000",
    # "bsll_lambda8TeV_M2000toInf",
]
# using one small dataset for debugging
# parameters["datasets"] = [
#     "dy_M50_incl"
# ]

if __name__ == "__main__":
    # prepare Dask client
    if use_local_cluster:
        print(
            f"Creating local cluster with {ncpus_local} workers."
            f" Dashboard address: {dashboard_address}"
        )
        client = Client(
            processes=True,
            # dashboard_address=dashboard_address,
            n_workers=ncpus_local,
            threads_per_worker=1,
            memory_limit="4GB",
        )
    else:
        print(
            f"Connecting to Slurm cluster at {cluster_ip}."
            f" Dashboard address: {dashboard_address}"
        )
        client = Client(parameters["cluster_ip"])
    parameters["ncpus"] = len(client.scheduler_info()["workers"])
    print(f"Connected to cluster! #CPUs = {parameters['ncpus']}")

    # add MVA scores to the list of variables to create histograms from
    # dnn_models = list(parameters["dnn_models"].values())
    # bdt_models = list(parameters["bdt_models"].values())
    # for models in dnn_models + bdt_models:
    #     for model in models:
    #         parameters["hist_vars"] += ["score_" + model]

    # prepare lists of paths to parquet files (stage1 output) for each year and dataset
    all_paths = {}
    for year in parameters["years"]:
        all_paths[year] = {}
        for dataset in parameters["datasets"]:
            paths = glob.glob(
                f"{parameters['global_path']}/"
                f"{parameters['label']}/stage1_output/{year}/"
                f"{dataset}/*.parquet"
            )
            all_paths[year][dataset] = paths

    # run postprocessing
    for year in parameters["years"]:
        print(f"Processing {year}")
        for dataset, path in tqdm.tqdm(all_paths[year].items()):
            if len(path) == 0:
                continue
            # read stage1 outputs
            df = load_dataframe(client, parameters, inputs=[path], dataset=dataset)
            if not isinstance(df, dd.DataFrame):
                continue

            # run processing sequence (categorization, mva, histograms)
            info = process_partitions(client, parameters, df)
            print(info)
