
dy_factor = 5765.4 / (5129.0 + 951.5 + 361.4)

cross_sections = {
    "test": 1,

    "dy_M50_incl": 5765.4,
    "dy_M50": 5765.4,

    "dy0J_M50": 5129.0 * dy_factor,
    "dy1J_M50": 951.5 * dy_factor,
    "dy2J_M50": 361.4 * dy_factor,

    "dy0J_M50_incl": 5129.0 * dy_factor,
    "dy1J_M50_incl": 951.5 * dy_factor,
    "dy2J_M50_incl": 361.4 * dy_factor,

    "dy0J_M200to400": 5.738e+00 * dy_factor,
    "dy0J_M400to800": 4.292e-01 * dy_factor,
    "dy0J_M800to1400": 2.503e-02 * dy_factor,
    "dy0J_M1400to2300": 1.796e-03 * dy_factor,
    "dy0J_M2300to3500": 1.008e-04 * dy_factor,
    "dy0J_M3500to4500": 3.968e-06 * dy_factor,
    "dy0J_M4500to6000": 4.040e-07 * dy_factor,
    "dy0J_M6000toInf":  2.788e-08 * dy_factor,
   
    "dy1J_M200to400": 2.211e+00 * dy_factor,
    "dy1J_M400to800": 2.416e-01 * dy_factor,
    "dy1J_M800to1400": 1.841e-02 * dy_factor,
    "dy1J_M1400to2300": 1.509e-03 * dy_factor,
    "dy1J_M2300to3500": 8.850e-05 * dy_factor,
    "dy1J_M3500to4500": 3.448e-06 * dy_factor,
    "dy1J_M4500to6000": 3.542e-07 * dy_factor,
    "dy1J_M6000toInf":  2.479e-08 * dy_factor,

    "dy2J_M200to400": 9.159e-01 * dy_factor,
    "dy2J_M400to800": 1.079e-01 * dy_factor,
    "dy2J_M800to1400": 9.395e-03 * dy_factor,
    "dy2J_M1400to2300": 8.550e-04 * dy_factor,
    "dy2J_M2300to3500": 5.525e-05 * dy_factor,
    "dy2J_M3500to4500": 2.377e-06 * dy_factor,
    "dy2J_M4500to6000": 2.744e-07 * dy_factor,
    "dy2J_M6000toInf": 2.049e-08 * dy_factor,

    "dy50to120": 2112.90,
    "dy120to200": 20.56,
    "dy200to400": 2.89,
    "dy400to800": 0.252,
    "dy800to1400": 1.71e-2,
    "dy1400to2300": 1.37e-3,
    "dy2300to3500": 8.178e-5,
    "dy3500to4500": 3.191e-6,
    "dy4500to6000": 2.787e-7,
    "dy6000toInf": 9.56e-9,
    "tW": 19.47,
    "Wantitop": 19.47,
    "tW1": 19.47,
    "Wantitop1": 19.47,
    "tW2": 19.47,
    "Wantitop2": 19.47,
    "Wjets": 61526.7,
    "WW200to600": 1.385,
    "WW600to1200": 0.0566,
    "WW1200to2500": 0.003557,
    "WW2500toInf": 0.00005395,
    "WWinclusive": 12.178,
    "WWinclusive_nocut": 12.178,
    "dyInclusive50": 5765.4,
    # ~ "WWinclusive":118.7,
    "WZ1L1Nu2Q": 10.73,
    "WZ_ext": 47.13,
    "ZZ_ext": 16.523,
    "ZZ": 16.523,
    "WZ3LNu": 4.42965,
    "WZ2L2Q": 6.331,
    "ZZ4L": 1.212,
    "ZZ4L_ext": 1.212,
    "ZZ2L2Nu": 0.564,
    "ZZ2L2Nu_ext": 0.564,
    "ZZ2L2Q": 1.999,
    "ttbar_lep_inclusive_nocut": 87.31,
    "ttbar_lep_inclusive": 87.31,
    "ttbar_lep50to500": 87.31,
    "ttbar_lep_M500to800": 0.32611,
    "ttbar_lep_M500to800_ext": 0.32611,
    "ttbar_lep_M800to1200": 0.03265,
    "ttbar_lep_M1200to1800": 0.00305,
    "ttbar_lep_M1800toInf": 0.00017,

    ### bbll samples from private production:
    "bbll_4TeV_M1000_negLL": 1.298e-2,
    "bbll_4TeV_M1000_negLR": 1.415e-2,
    "bbll_4TeV_M1000_posLL": 1.516e-2,
    "bbll_4TeV_M1000_posLR": 1.368e-2,
    "bbll_4TeV_M400_negLL": 3.251e-2,
    "bbll_4TeV_M400_negLR": 5.787e-2,
    "bbll_4TeV_M400_posLL": 7.394e-2,
    "bbll_4TeV_M400_posLR": 4.794e-2,
    "bbll_8TeV_M1000_negLL": 6.223e-4,
    "bbll_8TeV_M1000_negLR": 9.233e-4,
    "bbll_8TeV_M1000_posLL": 1.124e-3,
    "bbll_8TeV_M1000_posLR": 8.140e-4,
    "bbll_8TeV_M400_negLL": 1.750e-3,
    "bbll_8TeV_M400_negLR": 4.523e-3,
    "bbll_8TeV_M400_posLL": 8.489e-3,
    "bbll_8TeV_M400_posLR": 2.072e-3,
    ### bsll samples from private production
    "bsll_lambda1TeV_M1000to2000": 0.03227000,
    "bsll_lambda2TeV_M1000to2000": 0.00201400,
    "bsll_lambda4TeV_M1000to2000": 0.00012590,
    "bsll_lambda8TeV_M1000to2000": 0.00000791,
    "bsll_lambda1TeV_M2000toInf": 0.00643600,
    "bsll_lambda2TeV_M2000toInf": 0.00040430,
    "bsll_lambda4TeV_M2000toInf": 0.00002524,
    "bsll_lambda8TeV_M2000toInf": 0.00000156,
    "bsll_lambda1TeV_M200to500": 0.13690000,
    "bsll_lambda2TeV_M200to500": 0.00857600,
    "bsll_lambda4TeV_M200to500": 0.00053590,
    "bsll_lambda8TeV_M200to500": 0.00003345,
    "bsll_lambda1TeV_M500to1000": 0.07736000,
    "bsll_lambda2TeV_M500to1000": 0.00481600,
    "bsll_lambda4TeV_M500to1000": 0.00030010,
    "bsll_lambda8TeV_M500to1000": 0.00001894,
    ### bbll samples from official production
    "bbll_6TeV_M1300To2000_negLL": 8.979e-04,
    "bbll_6TeV_M2000ToInf_negLL": 2.485e-04,
    "bbll_6TeV_M300To800_negLL": -0.005252436882944,
    "bbll_6TeV_M800To1300_negLL": 2.126e-03,

    "bbll_10TeV_M1300To2000_negLL": 3.800e-04,
    "bbll_10TeV_M2000ToInf_negLL": 1.373e-04,
    "bbll_10TeV_M300To800_negLL": -0.005045158676077999,
    "bbll_10TeV_M800To1300_negLL": 7.733e-05,

    "bbll_14TeV_M1300To2000_negLL": 5.417e-05,
    "bbll_14TeV_M2000ToInf_negLL": 3.048e-05,
    "bbll_14TeV_M300To800_negLL": -0.0029449024720912,
    "bbll_14TeV_M800To1300_negLL": -2.106e-04,

    "bbll_18TeV_M1300To2000_negLL": -4.6936163158e-07,
    "bbll_18TeV_M2000ToInf_negLL": 8.547e-06,
    "bbll_18TeV_M300To800_negLL": -0.0018851545396580999,
    "bbll_18TeV_M800To1300_negLL": -5.90717325258e-05,

    "bbll_22TeV_M1300To2000_negLL": -3.1630243580479998e-06,
    "bbll_22TeV_M2000ToInf_negLL": 2.387e-06,
    "bbll_22TeV_M300To800_negLL": -0.0013085742388313,
    "bbll_22TeV_M800To1300_negLL": -4.753660083629e-05,

    "bbll_26TeV_M1300To2000_negLL": -3.378014398961e-06,
    "bbll_26TeV_M2000ToInf_negLL": 3.324e-07,
    "bbll_26TeV_M300To800_negLL": -0.0009505105447133999,
    "bbll_26TeV_M800To1300_negLL": -3.7536927547050005e-05,



}
