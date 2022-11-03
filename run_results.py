import argparse
import dask
from dask.distributed import Client
from config.variables import variables_lookup
from produceResults.plotter import plotter

# , plotter2D
# from produceResults.make_templates import to_templates

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
node_ip = "129.13.101.196"

if use_local_cluster:
    ncpus_local = 1
    cluster_ip = ""
    dashboard_address = None  # f"{node_ip}:34875"
else:
    cluster_ip = f"{args.ip}"
    dashboard_address = None  # f"{node_ip}:8787"

# global parameters
parameters = {
    # < general settings >
    "cluster_ip": cluster_ip,
    "years": args.years,
    "global_path": "/work/moh/DileptonBjets/Zprime-Dilepton/output/",
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
    "syst_variations": ["nominal"],

    # < plotting settings >
    "plot_vars": [
        "min_bl_mass",
        "mu1_pt",
        "mu1_eta",
        "mu1_phi",
        "dimuon_mass",
        # "dimuon_mass_gen",
        "bjet1_pt",
        # "bjet1_eta",
        # "bjet1_phi",
        "njets",
        "nbjets",
        "dimuon_cos_theta_cs",
    ],
    "plot_vars_2d": [],  # [["dimuon_mass", "met"]],
    "variables_lookup": variables_lookup,
    "save_plots": True,
    "plot_ratio": True,
    "plots_path": "./plots/v03/datamc/",
    "dnn_models": {},
    "bdt_models": {},

    # < templates and datacards >
    "save_templates": False,
    "templates_vars": [
        # "min_bl_mass",
        # "min_b1l_mass",
        # "min_b2l_mass",
        # "dimuon_mass",
        # "dimuon_mass_gen",
    ],
}

parameters["grouping"] = {
    "data_A": "Data",
    "data_B": "Data",
    "data_C": "Data",
    "data_D": "Data",

    "dy_M50_incl": "DY_incl",

    "dy0J_M50": "DY",
    "dy1J_M50": "DY",
    "dy2J_M50": "DY",
    "dy0J_M200to400": "DY",
    "dy0J_M400to800": "DY",
    "dy0J_M800to1400": "DY",
    "dy0J_M1400to2300": "DY",
    "dy0J_M2300to3500": "DY",
    "dy0J_M3500to4500": "DY",
    "dy0J_M4500to6000": "DY",
    "dy0J_M6000toInf": "DY",
    "dy1J_M200to400": "DY",
    "dy1J_M400to800": "DY",
    "dy1J_M800to1400": "DY",
    "dy1J_M1400to2300": "DY",
    "dy1J_M2300to3500": "DY",
    "dy1J_M3500to4500": "DY",
    "dy1J_M4500to6000": "DY",
    "dy1J_M6000toInf": "DY",
    "dy2J_M200to400": "DY",
    "dy2J_M400to800": "DY",
    "dy2J_M800to1400": "DY",
    "dy2J_M1400to2300": "DY",
    "dy2J_M2300to3500": "DY",
    "dy2J_M3500to4500": "DY",
    "dy2J_M4500to6000": "DY",
    "dy2J_M6000toInf": "DY",

    "ttbar_lep_inclusive": "TT",
    "ttbar_lep_M500to800": "TT",
    "ttbar_lep_M800to1200": "TT",
    "ttbar_lep_M1200to1800": "TT",
    "ttbar_lep_M1800toInf": "TT",

    "WWinclusive": "WW",
    "WW200to600": "WW",
    "WW600to1200": "WW",
    "WW1200to2500": "WW",
    "WW2500toInf": "WW",

    "WZ1L1Nu2Q": "Other",
    "WZ2L2Q": "Other",
    "WZ3LNu": "Other",
    "ZZ2L2Nu": "Other",
    "ZZ2L2Q": "Other",
    "ZZ4L": "Other",

    "Wantitop": "Other",
    "tW": "Other",

    "dyInclusive50": "Other",

    "bbll_6TeV_M1300To2000_negLL": "bbll_6TeV_negLL",
    "bbll_6TeV_M2000ToInf_negLL": "bbll_6TeV_negLL",
    "bbll_6TeV_M300To800_negLL": "bbll_6TeV_negLL",
    "bbll_6TeV_M800To1300_negLL": "bbll_6TeV_negLL",
    "bbll_10TeV_M1300To2000_negLL": "bbll_10TeV_negLL",
    "bbll_10TeV_M2000ToInf_negLL": "bbll_10TeV_negLL",
    "bbll_10TeV_M300To800_negLL": "bbll_10TeV_negLL",
    "bbll_10TeV_M800To1300_negLL": "bbll_10TeV_negLL",
}

parameters["plot_groups"] = {
    "stack": ["DY", "TT", "WW", "Other"],
    "step": [],  # ["bbll_6TeV_negLL", "bbll_10TeV_negLL"],
    "errorbar": ["Data"],
}

# ways to specificy colors for matplotlib are here: https://matplotlib.org/3.5.0/tutorials/colors/colors.html Using the xkcd color survey for now: https://xkcd.com/color/rgb/
parameters["color_dict"] = {
    "DY": "xkcd:water blue",
    "TT": "xkcd:pastel orange",
    "Other": "xkcd:shamrock green",
    "WW": "xkcd:red",
    "DYTauTau": "xkcd:blue",
    "Data": "xkcd:black",
    "bbll_6TeV_negLL": "xkcd:red",
    "bbll_10TeV_negLL": "xkcd:violet",

    "DY_incl": "xkcd:black",
    "DY_jbinned": "xkcd:red",
}

if __name__ == "__main__":
    if use_local_cluster:
        print(
            f"Creating local cluster with {ncpus_local} workers."
            f" Dashboard address: {dashboard_address}"
        )
        client = Client(
            processes=True,
            # dashboard_address=dashboard_address,
            n_workers=1,
            threads_per_worker=1,
            memory_limit="40GB",
        )
    else:
        print(
            f"Connecting to Slurm cluster at {cluster_ip}."
            f" Dashboard address: {dashboard_address}"
        )
        client = Client(parameters["cluster_ip"])
    parameters["ncpus"] = len(client.scheduler_info()["workers"])
    print(f"Connected to cluster! #CPUs = {parameters['ncpus']}")

    # add MVA scores to the list of variables to plot
    # dnn_models = list(parameters["dnn_models"].values())
    # bdt_models = list(parameters["bdt_models"].values())
    # for models in dnn_models + bdt_models:
    #     for model in models:
    #         parameters["plot_vars"] += ["score_" + model]
    #         parameters["templates_vars"] += ["score_" + model]

    parameters["datasets"] = parameters["grouping"].keys()

    # make 1D plots
    yields = plotter(client, parameters)

    # make 2D plots
    # yields2D = plotter2D(client, parameters)

    # save templates to ROOT files
    # yield_df = to_templates(client, parameters)

    # make datacards
    # build_datacards("score_pytorch_test", yield_df, parameters)
