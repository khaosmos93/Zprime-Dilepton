import numpy as np
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import glob
import math
from itertools import repeat
from setFrame import setFrame
import mplhep as hep
import time


def load2df(files):

    df = dd.read_parquet(files)
    field = [
        "dielectron_mass",
        "dielectron_cos_theta_cs",
        "njets",
        "dielectron_mass_gen",
        "pu_wgt",
        "r",
        "event",
        "run",
        "luminosityBlock",
    ]
    out = df[field]
    out.compute()
    return out


def chunk(files, size):

    size = math.ceil(len(files) / float(size))
    file_bag = [
        files[i : min(i + size, len(files))] for i in range(0, len(files), size)
    ]
    return file_bag


def df2hist(var, df, bins, masscut, njets=-1, iscut=False, iswgt=True, scale=True):
    if njets == -1:

        var_array = df.loc[
            (df["dielectron_mass"] > 120) & (df["r"] != "ee"), var
        ].compute()
        if iswgt:

            wgt = df.loc[
                (df["dielectron_mass"] > 120) & (df["r"] != "ee"), "pu_wgt"
            ].compute()
            wgt[wgt < 0] = 0
            if iscut:
                genmass = df.loc[
                    (df["dielectron_mass"] > 120) & (df["r"] != "ee"), "dielectron_mass"
                ].compute()
                wgt[genmass > masscut] = 0
            vals, bins = np.histogram(var_array, bins=bins, weights=wgt)
            vals2, bins = np.histogram(var_array, bins=bins, weights=wgt ** 2)
            errs = np.sqrt(vals2)
        else:
            vals, bins = np.histogram(var_array, bins=bins)
            errs = np.sqrt(vals)

    else:

        var_array = df.loc[
            (df["njets"] == njets) & (df["dielectron_mass"] > 120) & (df["r"] != "ee"),
            var,
        ].compute()

        if iswgt:
            wgt = df.loc[
                (df["njets"] == njets)
                & (df["dielectron_mass"] > 120)
                & (df["r"] != "ee"),
                "pu_wgt",
            ].compute()
            wgt[wgt < 0] = 0
            if iscut:
                genmass = df.loc[
                    (df["dielectron_mass"] > 120)
                    & (df["njets"] == njets)
                    & (df["r"] != "ee"),
                    "dielectron_mass",
                ].compute()
                wgt[genmass > masscut] = 0
            vals, bins = np.histogram(var_array, bins=bins, weights=wgt)
            vals2, bins = np.histogram(var_array, bins=bins, weights=wgt ** 2)
            errs = np.sqrt(vals2)
        else:
            vals, bins = np.histogram(var_array, bins=bins)
            errs = np.sqrt(vals)
    if scale:
        binSize = np.diff(bins)
        vals = vals / binSize
        errs = errs / binSize

    return [vals, errs, bins]


def plots(axes, data, MCs, labels, colors, name):

    bins = data[2]
    MCs_vals = [MC[0] for MC in MCs]
    hep.histplot(
        data[0],
        bins,
        ax=axes[0],
        color="black",
        histtype="errorbar",
        label="$Data$",
        yerr=data[1],
    )
    hep.histplot(
        MCs_vals,
        bins,
        ax=axes[0],
        color=colors,
        histtype="fill",
        label=labels,
        edgecolor=(0, 0, 0),
        stack=True,
    )
    bins_mid = (bins[1:] + bins[:-1]) / 2
    MC_vals = np.zeros(len(MCs[0][0]))
    MC_errs = np.zeros(len(MCs[0][0]))
    for MC in MCs:
        MC_vals += MC[0]
        MC_errs += MC[1] ** 2
    MC_errs = np.sqrt(MC_errs)
    r_vals = data[0] / MC_vals
    r_errs = r_vals * np.sqrt((data[1] / data[0]) ** 2 + (MC_errs / MC_vals) ** 2)
    r_MCerrs = MC_errs / MC_vals

    axes[0].fill_between(
        x=bins[:-1],
        y1=MC_vals - MC_errs,
        y2=MC_vals + MC_errs,
        interpolate=False,
        color="skyblue",
        alpha=0.3,
        step="post",
    )
    axes[0].legend(loc=(0.75, 0.40), fontsize="xx-small")
    hep.histplot(r_vals - 1, bins, ax=axes[1], histtype="errorbar", color="black")
    axes[1].fill_between(
        x=bins_mid,
        y1=-r_MCerrs,
        y2=r_MCerrs,
        interpolate=True,
        color="skyblue",
        alpha=0.3,
    )
    axes[3].savefig(
        f"/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/plots/{name}.pdf"
    )


if __name__ == "__main__":

    client_args = {
        "n_workers": 40,
        "memory_limit": "4.0GB",
        "timeout": 240,
    }

    bins_mass = (
        [j for j in range(120, 150, 5)]
        + [j for j in range(150, 200, 10)]
        + [j for j in range(200, 600, 20)]
        + [j for j in range(600, 900, 30)]
        + [j for j in range(900, 1250, 50)]
        + [j for j in range(1250, 1610, 60)]
        + [j for j in range(1610, 1890, 70)]
        + [j for j in range(1890, 3970, 80)]
        + [j for j in range(3970, 6070, 100)]
        + [6070]
    )

    bins_cs = np.linspace(-1.0, 1.0, 26)
    bins_jets = np.linspace(0, 7, 8)
    path = "/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/output/"
    path_dy = path + "dt_eev3/*/*.parquet"
    dy_files = glob.glob(path_dy)
    path_data = path + "pre-UL_eev2/*/*.parquet"
    data_files = glob.glob(path_data)
    path_tt_inclusive = path + "other_mc_eev3/ttbar_lep/*.parquet"
    tt_inclusive_files = glob.glob(path_tt_inclusive)
    path_tt = path + "other_mc_eev3/ttbar_lep_*/*.parquet"
    tt_files = glob.glob(path_tt)
    tt_files = [file_ for file_ in tt_files if "ext" not in file_]
    path_wz = path + "other_mc_eev3/WZ*/*.parquet"
    wz_files = glob.glob(path_wz)
    path_tw1 = path + "other_mc_eev3/tW/*.parquet"
    path_tw2 = path + "other_mc_eev3/Wantitop/*.parquet"
    tw_files = glob.glob(path_tw1) + glob.glob(path_tw2)
    path_zz = path + "other_mc_eev3/ZZ*/*.parquet"
    zz_files = glob.glob(path_zz)
    zz_files = [file_ for file_ in zz_files if "ext" not in file_]
    path_tau = path + "other_mc_eev3/dyInclusive50/*.parquet"
    tau_files = glob.glob(path_tau)
    path_ww = path + "other_mc_eev3/WW*0/*.parquet"
    ww_files = glob.glob(path_ww)
    path_ww_inclusive = path + "other_mc_eev3/WWinclusive/*.parquet"
    ww_inclusive_files = glob.glob(path_ww_inclusive)

    file_dict = {
        "data": data_files,
        "dy": dy_files,
        "tt": tt_files,
        "tt_inclu": tt_inclusive_files,
        "wz": wz_files,
        "tw": tw_files,
        "zz": zz_files,
        "tau": tau_files,
        "ww": ww_files,
        "ww_inclu": ww_inclusive_files,
    }

    client = Client(LocalCluster(**client_args))
    mass_inclu = {}
    mass_0j = {}
    mass_1j = {}
    mass_2j = {}
    cs_inclu = {}
    cs_0j = {}
    cs_1j = {}
    cs_2j = {}
    nbjets = {}
    for key in [
        "dy",
        "tt",
        "tt_inclu",
        "wz",
        "tw",
        "zz",
        "tau",
        "ww",
        "ww_inclu",
        "data",
    ]:
        file_bag = chunk(file_dict[key], client_args["n_workers"])
        if len(file_bag) == 1:
            df = load2df_mc(file_dict[key])
        else:
            results = client.map(load2df, file_bag)

            dfs = client.gather(results)
            df = dd.concat(dfs)
        if key == "tt_inclu":
            masscut = 500.0
            iscut = True
        elif key == "ww_inclu":
            masscut = 200.0
            iscut = True
        else:
            masscut = 0
            iscut = False

        if key == "data":
            iswgt = False
            client.close()
        else:
            iswgt = True

        mass_inclu[key] = df2hist(
            "dielectron_mass",
            df,
            bins_mass,
            masscut=masscut,
            njets=-1,
            iscut=iscut,
            iswgt=iswgt,
        )
        mass_0j[key] = df2hist(
            "dielectron_mass",
            df,
            bins_mass,
            masscut=masscut,
            njets=0,
            iscut=iscut,
            iswgt=iswgt,
        )
        mass_1j[key] = df2hist(
            "dielectron_mass",
            df,
            bins_mass,
            masscut=masscut,
            njets=1,
            iscut=iscut,
            iswgt=iswgt,
        )
        mass_2j[key] = df2hist(
            "dielectron_mass",
            df,
            bins_mass,
            masscut=masscut,
            njets=2,
            iscut=iscut,
            iswgt=iswgt,
        )

        cs_inclu[key] = df2hist(
            "dielectron_cos_theta_cs",
            df,
            bins_cs,
            masscut=masscut,
            njets=-1,
            iscut=iscut,
            iswgt=iswgt,
            scale=False,
        )
        cs_0j[key] = df2hist(
            "dielectron_cos_theta_cs",
            df,
            bins_cs,
            masscut=masscut,
            njets=0,
            iscut=iscut,
            iswgt=iswgt,
            scale=False,
        )
        cs_1j[key] = df2hist(
            "dielectron_cos_theta_cs",
            df,
            bins_cs,
            masscut=masscut,
            njets=1,
            iscut=iscut,
            iswgt=iswgt,
            scale=False,
        )
        cs_2j[key] = df2hist(
            "dielectron_cos_theta_cs",
            df,
            bins_cs,
            masscut=masscut,
            njets=2,
            iscut=iscut,
            iswgt=iswgt,
            scale=False,
        )

        nbjets[key] = df2hist(
            "njets",
            df,
            bins_jets,
            masscut=masscut,
            njets=-1,
            iscut=iscut,
            iswgt=iswgt,
            scale=False,
        )

    labels = [
        "$\\tau\\tau$",
        "$ZZ$",
        "$WZ$",
        "$WW$",
        "$tW$",
        "$t\\bar{t}$",
        "$\gamma/\mathrm{Z}\\rightarrow e^{+}e^{-}$",
    ]
    colors = ["yellow", "red", "darkred", "brown", "blue", "skyblue", "darkgreen"]

    data_2j = mass_2j["data"]
    data_1j = mass_1j["data"]
    data_0j = mass_0j["data"]
    data_inclu = mass_inclu["data"]

    mass_2j["tt"][0] += mass_2j["tt_inclu"][0]
    mass_2j["tt"][1] = np.sqrt(mass_2j["tt"][1] ** 2 + mass_2j["tt_inclu"][1] ** 2)
    mass_1j["tt"][0] += mass_1j["tt_inclu"][0]
    mass_1j["tt"][1] = np.sqrt(mass_1j["tt"][1] ** 2 + mass_1j["tt_inclu"][1] ** 2)
    mass_0j["tt"][0] += mass_0j["tt_inclu"][0]
    mass_0j["tt"][1] = np.sqrt(mass_0j["tt"][1] ** 2 + mass_0j["tt_inclu"][1] ** 2)
    mass_inclu["tt"][0] += mass_inclu["tt_inclu"][0]
    mass_inclu["tt"][1] = np.sqrt(
        mass_inclu["tt"][1] ** 2 + mass_inclu["tt_inclu"][1] ** 2
    )

    mass_2j["ww"][0] += mass_2j["ww_inclu"][0]
    mass_2j["ww"][1] = np.sqrt(mass_2j["ww"][1] ** 2 + mass_2j["ww_inclu"][1] ** 2)
    mass_1j["ww"][0] += mass_1j["ww_inclu"][0]
    mass_1j["ww"][1] = np.sqrt(mass_1j["ww"][1] ** 2 + mass_1j["ww_inclu"][1] ** 2)
    mass_0j["ww"][0] += mass_0j["ww_inclu"][0]
    mass_0j["ww"][1] = np.sqrt(mass_0j["ww"][1] ** 2 + mass_0j["ww_inclu"][1] ** 2)
    mass_inclu["ww"][0] += mass_inclu["ww_inclu"][0]
    mass_inclu["ww"][1] = np.sqrt(
        mass_inclu["ww"][1] ** 2 + mass_inclu["ww_inclu"][1] ** 2
    )

    MCs = [
        mass_2j["tau"],
        mass_2j["zz"],
        mass_2j["wz"],
        mass_2j["ww"],
        mass_2j["tw"],
        mass_2j["tt"],
        mass_2j["dy"],
    ]
    name = "dielectron_mass_2nbjets_pre"
    axes_mass2j = setFrame(
        "$\mathrm{m}(e^{+}e^{-})$ [GeV]",
        "Events/GeV",
        signal=True,
        ratio=True,
        logx=True,
        logy=True,
        xRange=[120, 6000],
        yRange=[1e-7, 1e5],
        flavor="el",
        year="2018",
    )
    plots(axes_mass2j, data_2j, MCs, labels, colors, name)

    MCs = [
        mass_1j["tau"],
        mass_1j["zz"],
        mass_1j["wz"],
        mass_1j["ww"],
        mass_1j["tw"],
        mass_1j["tt"],
        mass_1j["dy"],
    ]
    name = "dielectron_mass_1nbjets_pre"
    axes_mass1j = setFrame(
        "$\mathrm{m}(e^{+}e^{-})$ [GeV]",
        "Events/GeV",
        signal=True,
        ratio=True,
        logx=True,
        logy=True,
        xRange=[120, 6000],
        yRange=[1e-6, 1e6],
        flavor="el",
        year="2018",
    )
    plots(axes_mass1j, data_1j, MCs, labels, colors, name)

    MCs = [
        mass_0j["tau"],
        mass_0j["zz"],
        mass_0j["wz"],
        mass_0j["ww"],
        mass_0j["tw"],
        mass_0j["tt"],
        mass_0j["dy"],
    ]
    name = "dielectron_mass_0nbjets_pre"
    axes_mass0j = setFrame(
        "$\mathrm{m}(e^{+}e^{-})$ [GeV]",
        "Events/GeV",
        signal=True,
        ratio=True,
        logx=True,
        logy=True,
        xRange=[120, 6000],
        yRange=[1e-5, 1e7],
        flavor="el",
        year="2018",
    )
    plots(axes_mass0j, data_0j, MCs, labels, colors, name)

    MCs = [
        mass_inclu["tau"],
        mass_inclu["zz"],
        mass_inclu["wz"],
        mass_inclu["ww"],
        mass_inclu["tw"],
        mass_inclu["tt"],
        mass_inclu["dy"],
    ]
    name = "dielectron_mass_inclu_pre"
    axes_mass = setFrame(
        "$\mathrm{m}(e^{+}e^{-})$ [GeV]",
        "Events/GeV",
        signal=True,
        ratio=True,
        logx=True,
        logy=True,
        xRange=[120, 6000],
        yRange=[1e-5, 1e7],
        flavor="el",
        year="2018",
    )
    plots(axes_mass, data_inclu, MCs, labels, colors, name)

    data_2j = cs_2j["data"]
    data_1j = cs_1j["data"]
    data_0j = cs_0j["data"]
    data_inclu = cs_inclu["data"]

    cs_2j["tt"][0] += cs_2j["tt_inclu"][0]
    cs_2j["tt"][1] = np.sqrt(cs_2j["tt"][1] ** 2 + cs_2j["tt_inclu"][1] ** 2)
    cs_1j["tt"][0] += cs_1j["tt_inclu"][0]
    cs_1j["tt"][1] = np.sqrt(cs_1j["tt"][1] ** 2 + cs_1j["tt_inclu"][1] ** 2)
    cs_0j["tt"][0] += cs_0j["tt_inclu"][0]
    cs_0j["tt"][1] = np.sqrt(cs_0j["tt"][1] ** 2 + cs_0j["tt_inclu"][1] ** 2)
    cs_inclu["tt"][0] += cs_inclu["tt_inclu"][0]
    cs_inclu["tt"][1] = np.sqrt(cs_inclu["tt"][1] ** 2 + cs_inclu["tt_inclu"][1] ** 2)

    cs_2j["ww"][0] += cs_2j["ww_inclu"][0]
    cs_2j["ww"][1] = np.sqrt(cs_2j["ww"][1] ** 2 + cs_2j["ww_inclu"][1] ** 2)
    cs_1j["ww"][0] += cs_1j["ww_inclu"][0]
    cs_1j["ww"][1] = np.sqrt(cs_1j["ww"][1] ** 2 + cs_1j["ww_inclu"][1] ** 2)
    cs_0j["ww"][0] += cs_0j["ww_inclu"][0]
    cs_0j["ww"][1] = np.sqrt(cs_0j["ww"][1] ** 2 + cs_0j["ww_inclu"][1] ** 2)
    cs_inclu["ww"][0] += cs_inclu["ww_inclu"][0]
    cs_inclu["ww"][1] = np.sqrt(cs_inclu["ww"][1] ** 2 + cs_inclu["ww_inclu"][1] ** 2)

    MCs = [
        cs_2j["tau"],
        cs_2j["zz"],
        cs_2j["wz"],
        cs_2j["ww"],
        cs_2j["tw"],
        cs_2j["tt"],
        cs_2j["dy"],
    ]
    name = "dielectron_cs_2nbjets_pre"
    axes_cs2j = setFrame(
        "$\mathrm{cos}\\theta$",
        "Events",
        signal=True,
        ratio=True,
        logx=False,
        logy=False,
        xRange=[-1.0, 1.0],
        yRange=[0, 3000],
        flavor="el",
        year="2018",
    )
    plots(axes_cs2j, data_2j, MCs, labels, colors, name)

    MCs = [
        cs_1j["tau"],
        cs_1j["zz"],
        cs_1j["wz"],
        cs_1j["ww"],
        cs_1j["tw"],
        cs_1j["tt"],
        cs_1j["dy"],
    ]
    name = "dielectron_cs_1nbjets_pre"
    axes_cs1j = setFrame(
        "$\mathrm{cos}\\theta$",
        "Events",
        signal=True,
        ratio=True,
        logx=False,
        logy=False,
        xRange=[-1.0, 1.0],
        yRange=[0, 15000],
        flavor="el",
        year="2018",
    )
    plots(axes_cs1j, data_1j, MCs, labels, colors, name)

    MCs = [
        cs_0j["tau"],
        cs_0j["zz"],
        cs_0j["wz"],
        cs_0j["ww"],
        cs_0j["tw"],
        cs_0j["tt"],
        cs_0j["dy"],
    ]
    name = "dielectron_cs_0nbjets_pre"
    axes_cs0j = setFrame(
        "$\mathrm{cos}\\theta$",
        "Events",
        signal=True,
        ratio=True,
        logx=False,
        logy=False,
        xRange=[-1.0, 1.0],
        yRange=[0, 75000],
        flavor="el",
        year="2018",
    )
    plots(axes_cs0j, data_0j, MCs, labels, colors, name)

    MCs = [
        cs_inclu["tau"],
        cs_inclu["zz"],
        cs_inclu["wz"],
        cs_inclu["ww"],
        cs_inclu["tw"],
        cs_inclu["tt"],
        cs_inclu["dy"],
    ]
    name = "dielectron_cs_inclusive_pre"
    axes_cs = setFrame(
        "$\mathrm{cos}\\theta$",
        "Events",
        signal=True,
        ratio=True,
        logx=False,
        logy=False,
        xRange=[-1.0, 1.0],
        yRange=[0, 75000],
        flavor="el",
        year="2018",
    )
    plots(axes_cs, data_inclu, MCs, labels, colors, name)

    data = nbjets["data"]

    nbjets["tt"][0] += nbjets["tt_inclu"][0]
    nbjets["tt"][1] = np.sqrt(nbjets["tt"][1] ** 2 + nbjets["tt_inclu"][1] ** 2)

    nbjets["ww"][0] += nbjets["ww_inclu"][0]
    nbjets["ww"][1] = np.sqrt(nbjets["ww"][1] ** 2 + nbjets["ww_inclu"][1] ** 2)

    MCs = [
        nbjets["tau"],
        nbjets["zz"],
        nbjets["wz"],
        nbjets["ww"],
        nbjets["tw"],
        nbjets["tt"],
        nbjets["dy"],
    ]
    name = "dielectron_nbjets_pre"
    axes = setFrame(
        "Number of b-jets",
        "Events",
        signal=True,
        ratio=True,
        logx=False,
        logy=True,
        xRange=[0, 7.0],
        yRange=[1e-5, 1e11],
        flavor="el",
        year="2018",
    )
    plots(axes, data, MCs, labels, colors, name)
