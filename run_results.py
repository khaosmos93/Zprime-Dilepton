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
    "-f", "--flavor", dest="flavor", help="lepton flavor", default="mu"
)


parser.add_argument(
    "-sl",
    "--slurm",
    dest="slurm_port",
    default=None,
    action="store",
    help="Slurm cluster port (if not specified, will create a local cluster)",
)
args = parser.parse_args()

# Dask client settings
use_local_cluster = args.slurm_port is None
node_ip = "128.211.148.60"

if use_local_cluster:
    ncpus_local = 1
    slurm_cluster_ip = ""
    dashboard_address = f"{node_ip}:34875"
else:
    slurm_cluster_ip = f"{node_ip}:{args.slurm_port}"
    dashboard_address = f"{node_ip}:8787"

# global parameters
parameters = {
    # < general settings >
    "slurm_cluster_ip": slurm_cluster_ip,
    "years": args.years,
    "global_path": "/depot/cms/users/schul105/dilepton/Zprime-Dilepton/output/",
    "label": "combinedTest",
    "flavor": args.flavor,
    "channels": ["inclusive", "0b", "1b", "2b"],
    "regions": ["inclusive", "bb", "be"],
    "syst_variations": ["nominal"],
    #
    # < plotting settings >
    "plot_vars": [
        "min_bl_mass",
        "min_b1l_mass",
        "min_b2l_mass",
        "dilepton_mass",
        "dilepton_mass_gen",
        "njets",
        "nbjets",
        "dilepton_cos_theta_cs",
    ],  # "dilepton_mass"],
    "plot_vars_2d": [["dilepton_mass", "met"]],  # "dilepton_mass"],
    "variables_lookup": variables_lookup,
    "save_plots": True,
    "plot_ratio": True,
    "plots_path": "./plots/test_%s/"%args.flavor,
    "dnn_models": {},
    "bdt_models": {},
    #
    # < templates and datacards >
    "save_templates": True,
    "templates_vars": [
        "min_bl_mass",
        "min_b1l_mass",
        "min_b2l_mass",
        "dilepton_mass",
        "dilepton_mass_gen",
    ],  # "dilepton_mass"],
}



parameters["grouping"] = {
    "data_A": "Data",
    "data_B": "Data",
    "data_C": "Data",
    "data_D": "Data",
    # "data_E": "Data",
    # "data_F": "Data",
    # "data_G": "Data",
    # "data_H": "Data",
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
    "ttbar_lep_inclusive": "Top",
    "ttbar_lep_M500to800": "Top",
    "ttbar_lep_M800to1200": "Top",
    "ttbar_lep_M1200to1800": "Top",
    "ttbar_lep_M1800toInf": "Top",
    "tW": "Top",
    "Wantitop": "Top",
    # "tW2" : "Top",
    # "Wantitop2" : "Top",
    "WWinclusive": "WW",
    "WW200to600": "WW",
    "WW600to1200": "WW",
    "WW1200to2500": "WW",
    "WW2500": "Other",
    "WZ2L2Q": "Other",
    "WZ3LNu": "Other",
    "ZZ2L2Nu": "Other",
    "ZZ4L": "Other",
    # "bbll_4TeV_M400_posLL" : "bbll_4TeV_posLL",
    # "bbll_4TeV_M1000_posLL" : "bbll_4TeV_posLL",
    # "bbll_8TeV_M400_posLL" : "bbll_8TeV_posLL",
    # "bbll_8TeV_M1000_posLL" : "bbll_8TeV_posLL",
}

parameters["plot_groups"] = {
    "stack": ["DY", "Top", "Other", "WW"],
    # "step": ["bbll_4TeV_posLL", "bbll_8TeV_posLL"],
    "errorbar": ["Data"],
    # "2D": ["Data","DY","Other","bbll_4TeV_posLL","bbll_8TeV_posLL"],
    # "2D": ["DY","Other"],
}
# ways to specificy colors for matplotlib are here: https://matplotlib.org/3.5.0/tutorials/colors/colors.html Using the xkcd color survey for now: https://xkcd.com/color/rgb/
parameters["color_dict"] = {
    "DY": "xkcd:water blue",
    "Top": "xkcd:pastel orange",
    "Other": "xkcd:shamrock green",
    "WW": "xkcd:red",
    "DYTauTau": "xkcd:blue",
    "Data": "xkcd:black",
    "bbll_4TeV_posLL": "xkcd:red",
    "bbll_8TeV_posLL": "xkcd:violet",
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
            f"Connecting to Slurm cluster at {slurm_cluster_ip}."
            f" Dashboard address: {dashboard_address}"
        )
        client = Client(parameters["slurm_cluster_ip"])
    parameters["ncpus"] = len(client.scheduler_info()["workers"])
    print(f"Connected to cluster! #CPUs = {parameters['ncpus']}")

    # add MVA scores to the list of variables to plot
    dnn_models = list(parameters["dnn_models"].values())
    bdt_models = list(parameters["bdt_models"].values())
    for models in dnn_models + bdt_models:
        for model in models:
            parameters["plot_vars"] += ["score_" + model]
            parameters["templates_vars"] += ["score_" + model]

    parameters["datasets"] = parameters["grouping"].keys()

    # make 1D plots
    yields = plotter(client, parameters)

    # make 2D plots
    # yields2D = plotter2D(client, parameters)

    # save templates to ROOT files
    # yield_df = to_templates(client, parameters)

    # make datacards
    # build_datacards("score_pytorch_test", yield_df, parameters)
