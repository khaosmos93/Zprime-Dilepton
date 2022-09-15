import sys

sys.path.append("copperhead/")
import time
import argparse
import traceback
import datetime
from functools import partial
from coffea.processor import DaskExecutor, Runner
from coffea.nanoevents import NanoAODSchema

# from copperhead.stage1.preprocessor import load_samples
from processNano.preprocessor import load_samples
from copperhead.python.io import mkdir, save_stage1_output_to_parquet
import dask
from dask.distributed import Client
import os

user_name = os.getcwd().split("/")[5]
dask.config.set({"temporary-directory": f"/depot/cms/users/{user_name}/dask-temp/"})
global_path = os.getcwd() + "/output/"
parser = argparse.ArgumentParser()
# Slurm cluster IP to use. If not specified, will create a local cluster
parser.add_argument(
    "-sl",
    "--slurm",
    dest="slurm_port",
    default=None,
    action="store",
    help="Slurm cluster port (if not specified, " "will create a local cluster)",
)
parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="2016",
    action="store",
    help="Year to process (2016, 2017 or 2018)",
)
parser.add_argument(
    "-l",
    "--label",
    dest="label",
    default="test_march",
    action="store",
    help="Unique run label (to create output path)",
)
parser.add_argument(
    "-ch",
    "--chunksize",
    dest="chunksize",
    default=100000,
    action="store",
    help="Approximate chunk size",
)
parser.add_argument(
    "-mch",
    "--maxchunks",
    dest="maxchunks",
    default=-1,
    action="store",
    help="Max. number of chunks",
)
parser.add_argument(
    "-cl",
    "--channel",
    dest="channel",
    default="mu",
    action="store",
    help="the flavor of the final state dilepton",
)


args = parser.parse_args()

node_ip = "128.211.148.60"  # hammer-c000
# node_ip = "128.211.149.135"
# node_ip = "128.211.149.140"
dash_local = f"{node_ip}:34875"


if args.slurm_port is None:
    local_cluster = True
    slurm_cluster_ip = ""
else:
    local_cluster = False
    slurm_cluster_ip = f"{node_ip}:{args.slurm_port}"

mch = None if int(args.maxchunks) < 0 else int(args.maxchunks)
dt = datetime.datetime.now()
local_time = (
    str(dt.year)
    + "_"
    + str(dt.month)
    + "_"
    + str(dt.day)
    + "_"
    + str(dt.hour)
    + "_"
    + str(dt.minute)
    + "_"
    + str(dt.second)
)
parameters = {
    "year": args.year,
    "label": args.label,
    "global_path": global_path,
    "out_path": f"{args.year}_{args.label}_{local_time}",
    # "server": "root://xrootd.rcac.purdue.edu/",
    #"server": "root://cmsxrootd.fnal.gov//",
    "xrootd": False,
    "server": "/mnt/hadoop/",
    "datasets_from": "Zprime",
    "from_das": True,
    "chunksize": int(args.chunksize),
    "maxchunks": mch,
    "save_output": True,
    "local_cluster": local_cluster,
    "slurm_cluster_ip": slurm_cluster_ip,
    "client": None,
    "channel": args.channel,
    "n_workers": 32,
    "do_timer": True,
    "do_btag_syst": False,
    "save_output": True,
}

parameters["out_dir"] = f"{parameters['global_path']}/" f"{parameters['out_path']}"


def saving_func(output, out_dir):
    from dask.distributed import get_worker

    name = None
    for key, task in get_worker().tasks.items():
        if task.state == "executing":
            name = key[-32:]
    if not name:
        return
    for ds in output.s.unique():
        df = output[output.s == ds]
        # df = df.drop_duplicates(subset=["run", "event", "luminosityBlock"])
        if df.shape[0] == 0:
            continue
        mkdir(f"{out_dir}/{ds}")
        df.to_parquet(
            path=f"{out_dir}/{ds}/{name}.parquet",
        )
    del output


def submit_job(parameters):
    # mkdir(parameters["out_path"])
    if parameters["channel"] == "eff_mu":
        out_dir = parameters["global_path"] + parameters["out_path"]
    else:
        out_dir = parameters["global_path"]
    print(out_dir)
    mkdir(out_dir)
    out_dir += "/" + parameters["label"]
    mkdir(out_dir)
    out_dir += "/" + "stage1_output"
    mkdir(out_dir)
    out_dir += "/" + parameters["year"]
    mkdir(out_dir)
    executor_args = {"client": parameters["client"], "retries": 0}
    processor_args = {
        "samp_info": parameters["samp_infos"],
        "do_timer": parameters["do_timer"],
        "do_btag_syst": parameters["do_btag_syst"],
        # "regions": parameters["regions"],
        # "pt_variations": parameters["pt_variations"],
        "apply_to_output": partial(save_stage1_output_to_parquet, out_dir=out_dir),
    }

    if parameters["channel"] == "mu":
        from processNano.dimuon_processor import DimuonProcessor as event_processor
    elif parameters["channel"] == "el":
        from processNano.dielectron_processor import (
            DielectronProcessor as event_processor,
        )
    elif parameters["channel"] == "eff_mu":
        from processNano.dimuon_eff_processor import (
            DimuonEffProcessor as event_processor,
        )
    elif parameters["channel"] == "preselection_mu":
        from processNano.dimuon_preselector import (
            DimuonProcessor as event_processor,
        )
    else:
        print("wrong channel input")

    executor = DaskExecutor(**executor_args)
    run = Runner(
        executor=executor,
        schema=NanoAODSchema,
        chunksize=parameters["chunksize"],
        maxchunks=parameters["maxchunks"],
    )

    try:
        run(
            parameters["samp_infos"].fileset,
            "Events",
            processor_instance=event_processor(**processor_args),
        )

    except Exception as e:
        tb = traceback.format_exc()
        return "Failed: " + str(e) + " " + tb

    return "Success!"


if __name__ == "__main__":
    tick = time.time()
    smp = {
        # 'single_file': [
        #     'test_file',
        # ],
        "data": [
            "data_A",
            "data_B",
            "data_C",
            "data_D",
        ],
        "other_mc": [
            "WZ",
            "WZ3LNu",
            "WZ2L2Q",
            "ZZ",
            #"ZZ2L2Nu",
            "ZZ2L2Nu_ext",
            "ZZ2L2Q",
            "ZZ4L",
            "ZZ4L_ext",
            "WWinclusive",
            "WW200to600",
            "WW600to1200",
            "WW1200to2500",
            "WW2500",
            "dyInclusive50",
            "Wjets",
            "ttbar_lep_inclusive",
            "ttbar_lep_M500to800_ext",
            "WW600to1200",
            "WW1200to2500",
            "ttbar_lep_M1200to1800",
            "ttbar_lep_M1800toInf",
            "Wantitop",
            "Wantitop1",
            "Wantitop2",
            "tW",
            "tW1",
            "tW2",
        ],
        "dy": [
            # "dy50to120",
            "dy120to200",
            "dy200to400",
            "dy400to800",
            "dy800to1400",
            "dy1400to2300",
            "dy2300to3500",
            "dy3500to4500",
            "dy4500to6000",
            "dy6000toInf",
        ],
        "CI": [
            "bbll_4TeV_M1000_negLL",
            "bbll_4TeV_M1000_negLR",
            "bbll_4TeV_M1000_posLL",
            "bbll_4TeV_M1000_posLR",
            "bbll_4TeV_M400_negLL",
            "bbll_4TeV_M400_negLR",
            "bbll_4TeV_M400_posLL",
            "bbll_4TeV_M400_posLR",
            "bbll_8TeV_M1000_negLL",
            "bbll_8TeV_M1000_negLR",
            "bbll_8TeV_M1000_posLL",
            "bbll_8TeV_M1000_posLR",
            "bbll_8TeV_M400_negLL",
            "bbll_8TeV_M400_negLR",
            "bbll_8TeV_M400_posLL",
            "bbll_8TeV_M400_posLR",
        ],
    }
    # prepare Dask client
    if parameters["local_cluster"]:
        # create local cluster
        parameters["client"] = Client(
            processes=True,
            n_workers=24,
            # dashboard_address=dash_local,
            threads_per_worker=1,
            memory_limit="6GB",
        )
    else:
        # connect to existing Slurm cluster
        parameters["client"] = Client(parameters["slurm_cluster_ip"])
    print("Client created")

    datasets_mc = []
    datasets_data = []

    for group, samples in smp.items():
        for sample in samples:
            # if sample not in blackList:
            #    continue
            # if "WWinclusive" not in sample:
            # if "dy200to400" not in sample:
            # if sample != "ttbar_lep_inclusive":
            #    continue
            if "dy200to400" not in sample:
            # if not ("ttbar" in sample or "Wantitop" in sample or "tW" in sample):
                continue

            #if group != "other_mc":
            #    continue
            #if sample not in ["data_A"]:
            #    continue
            # if group != "data":
            #    continue
            if group == "data":
                datasets_data.append(sample)
            else:
                datasets_mc.append(sample)

    timings = {}

    to_process = {"MC": datasets_mc, "DATA": datasets_data}
    for lbl, datasets in to_process.items():
        if len(datasets) == 0:
            continue
        print(f"Processing {lbl}")
        arg_sets = []
        for d in datasets:
            arg_sets.append({"dataset": d})
        tick1 = time.time()
        parameters["samp_infos"] = load_samples(datasets, parameters)
        timings[f"load {lbl}"] = time.time() - tick1

        tick2 = time.time()
        out = submit_job(parameters)
        timings[f"process {lbl}"] = time.time() - tick2

        print(out)

    elapsed = round(time.time() - tick, 3)
    print(f"Finished everything in {elapsed} s.")
    print("Timing breakdown:")
    print(timings)
