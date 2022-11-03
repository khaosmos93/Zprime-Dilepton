import sys

sys.path.append("copperhead/")
import time
import argparse
import traceback
import datetime
from functools import partial
from coffea.processor import DaskExecutor, Runner
from coffea.nanoevents import NanoAODSchema

from processNano.preprocessor import load_samples
from copperhead.python.io import mkdir, save_stage1_output_to_parquet
import dask
from dask.distributed import Client
import os

user_name = os.getcwd().split("/")[2]
dask.config.set({"temporary-directory": f"/ceph/{user_name}/Dilepton/dask-temp/"})
global_path = os.getcwd() + "/output/"
parser = argparse.ArgumentParser()
# Slurm cluster IP to use. If not specified, will create a local cluster
parser.add_argument(
    "-i",
    "--ip",
    dest="ip",
    default=None,
    action="store",
    help="Cluster ip:port (if not specified, " "will create a local cluster)",
)
parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="2018",
    action="store",
    help="Year to process (2016, 2017 or 2018)",
)
parser.add_argument(
    "-l",
    "--label",
    dest="label",
    default="v03",
    action="store",
    help="Unique run label (to create output path)",
)
parser.add_argument(
    "-ch",
    "--chunksize",
    dest="chunksize",
    default=200000,
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

dash_local = None  # f"{node_ip}:34875"


if args.ip is None:
    local_cluster = True
    cluster_ip = ""
else:
    local_cluster = False
    cluster_ip = f"{args.ip}"

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
    "xrootd": False,
    "server": "root://cmsxrootd-redirectors.gridka.de/",
    "datasets_from": "muon",
    "from_das": False,
    "chunksize": int(args.chunksize),
    "maxchunks": mch,
    "local_cluster": local_cluster,
    "cluster_ip": cluster_ip,
    "client": None,
    "channel": args.channel,
    "n_workers": 24,
    "do_timer": False,
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
    # print(out_dir)
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
        "data": [
            # "data_A",
            # "data_B",
            # "data_C",
            # "data_D",
        ],
        "other_mc": [
            "ttbar_lep_inclusive_nocut",
            "WWinclusive_nocut",

            # "dyInclusive50",
            # "Wantitop",
            # "tW",
            # "ttbar_lep_inclusive",
            # "ttbar_lep_M500to800",
            # "ttbar_lep_M800to1200",
            # "ttbar_lep_M1200to1800",
            # "ttbar_lep_M1800toInf",
            # "WWinclusive",
            # "WW200to600",
            # "WW600to1200",
            # "WW1200to2500",
            # "WW2500toInf",
            # "WZ1L1Nu2Q",
            # "WZ2L2Q",
            # "WZ3LNu",
            # "ZZ2L2Nu",
            # "ZZ2L2Q",
            # "ZZ4L",
        ],
        "dy": [
            # "dy_M50_incl",
            # "dy_M50",
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
        ],
        "CI": [
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

            "bbll_6TeV_M1300To2000_negLL",
            "bbll_6TeV_M2000ToInf_negLL",
            "bbll_6TeV_M300To800_negLL",
            "bbll_6TeV_M800To1300_negLL",
            "bbll_10TeV_M1300To2000_negLL",
            "bbll_10TeV_M2000ToInf_negLL",
            "bbll_10TeV_M300To800_negLL",
            "bbll_10TeV_M800To1300_negLL",
            "bbll_14TeV_M1300To2000_negLL",
            "bbll_14TeV_M2000ToInf_negLL",
            "bbll_14TeV_M300To800_negLL",
            "bbll_14TeV_M800To1300_negLL",
            "bbll_18TeV_M1300To2000_negLL",
            "bbll_18TeV_M2000ToInf_negLL",
            "bbll_18TeV_M300To800_negLL",
            "bbll_18TeV_M800To1300_negLL",
            "bbll_22TeV_M1300To2000_negLL",
            "bbll_22TeV_M2000ToInf_negLL",
            "bbll_22TeV_M300To800_negLL",
            "bbll_22TeV_M800To1300_negLL",
            "bbll_26TeV_M1300To2000_negLL",
            "bbll_26TeV_M2000ToInf_negLL",
            "bbll_26TeV_M300To800_negLL",
            "bbll_26TeV_M800To1300_negLL",

            "bbll_4TeV_M1000_negLL",
            "bbll_4TeV_M1000_negLR",
            "bbll_4TeV_M1000_posLL",
            "bbll_4TeV_M1000_posLR",
            "bbll_8TeV_M1000_negLL",
            "bbll_8TeV_M1000_negLR",
            "bbll_8TeV_M1000_posLL",
            "bbll_8TeV_M1000_posLR",
            "bbll_4TeV_M400_negLL",
            "bbll_4TeV_M400_negLR",
            "bbll_4TeV_M400_posLL",
            "bbll_4TeV_M400_posLR",
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
            n_workers=parameters["n_workers"],
            # dashboard_address=dash_local,
            threads_per_worker=1,
            memory_limit="10GB",
        )
        print("Client:", parameters["client"])
        print("dashboard_link:", parameters["client"].dashboard_link)
    else:
        # connect to existing Slurm cluster
        parameters["client"] = Client(parameters["cluster_ip"])
    print("Client created")

    datasets_mc = []
    datasets_data = []

    for group, samples in smp.items():
        for sample in samples:
            if group == "data":
                datasets_data.append(sample)
            else:
                datasets_mc.append(sample)

    timings = {}

    to_process = {"MC": datasets_mc, "DATA": datasets_data}
    # to_process = {"DATA": datasets_data}
    # to_process = {"MC": datasets_mc}
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
