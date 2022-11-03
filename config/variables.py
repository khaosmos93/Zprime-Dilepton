class Variable(object):
    def __init__(
        self,
        name_,
        caption_,
        nbins_,
        xmin_,
        xmax_,
        ymin_,
        ymax_,
        binning_=[],
        norm_to_bin_width_=False,
        xminPlot_=None,
        xmaxPlot_=None,
        logx_=False,
    ):
        self.name = name_
        self.caption = caption_
        self.nbins = nbins_
        self.xmin = xmin_
        self.xmax = xmax_
        self.ymin = ymin_
        self.ymax = ymax_
        self.logx = logx_
        if xminPlot_ is not None:
            self.xminPlot = xminPlot_
        else:
            self.xminPlot = xmin_
        if xmaxPlot_ is not None:
            self.xmaxPlot = xmaxPlot_
        else:
            self.xmaxPlot = xmax_
        self.binning = binning_
        self.norm_to_bin_width = norm_to_bin_width_


variables = []

n_mass_bins = 60
mass_min = 60.
mass_max = 6500.
mass_r = (mass_max/mass_min)**(1./n_mass_bins)
massBinningMuMu = (
    [
        60, 64, 70, 75, 81, 
        88, 95, 103, 112, 120, 

        # BFF Z' signal region: 150-500 GeV -> 50 GeV bins ?
        131, 141, 153, 165, 179, 
        193, 209, 226, 244, 264, 
        286, 309, 334, 361, 390, 
        422, 456, 494, 534, 577, 

        # 131, 141, 150, 200, 250, 
        # 300, 350, 400, 450, 500,
        # 577,

        624, 675, 730, 789, 853, 
        922, 997, 1078, 1166, 1261, 
        1363, 1474, 1593, 1723, 1863, 
        2014, 2178, 2355, 2546, 2753, 
        3000, 3218, 3480, 3762, 4068, 
        4398, 4756, 5142, 5560, 6011, 
        6500
    ]

    # [mass_min*(int(mass_r**float(j))) for j in range(n_mass_bins+1)]

    # [j for j in range(0, 7000, 100)]
    # + [7000]

    # [j for j in range(120, 150, 5)]
    # + [j for j in range(150, 200, 10)]
    # + [j for j in range(200, 600, 20)]
    # + [j for j in range(600, 900, 30)]
    # + [j for j in range(900, 1250, 50)]
    # + [j for j in range(1250, 1610, 60)]
    # + [j for j in range(1610, 1890, 70)]
    # + [j for j in range(1890, 3970, 80)]
    # + [j for j in range(3970, 6070, 100)]
    # + [6070]
)

variables.append(
    Variable(
        "dimuon_mass",
        r"$m_{\mu\mu}$ [GeV]",
        len(massBinningMuMu) - 1,
        120,
        3000,
        1e-5,
        1e5,
        binning_=massBinningMuMu,
        norm_to_bin_width_=True,
        logx_=True,
    )
)
variables.append(
    Variable(
        "dimuon_mass_resUnc",
        r"$m_{\mu\mu}$ [GeV] (res. unc.)",
        len(massBinningMuMu) - 1,
        120,
        3000,
        1e-5,
        1e5,
        binning_=massBinningMuMu,
        norm_to_bin_width_=True,
        logx_=True,
    )
)
variables.append(
    Variable(
        "dimuon_mass_scaleUncUp",
        r"$m_{\mu\mu}$ [GeV] (scale unc. up)",
        len(massBinningMuMu) - 1,
        120,
        3000,
        1e-5,
        1e5,
        binning_=massBinningMuMu,
        norm_to_bin_width_=True,
        logx_=True,
    )
)
variables.append(
    Variable(
        "dimuon_mass_scaleUncDown",
        r"$m_{\mu\mu}$ [GeV] (scale unc. down)",
        len(massBinningMuMu) - 1,
        120,
        3000,
        1e-5,
        1e5,
        binning_=massBinningMuMu,
        norm_to_bin_width_=True,
        logx_=True,
    )
)
variables.append(
    Variable(
        "dimuon_mass_gen",
        r"generated $m_{\mu\mu}$ [GeV]",
        len(massBinningMuMu) - 1,
        120,
        3000,
        1e-5,
        1e5,
        binning_=massBinningMuMu,
        norm_to_bin_width_=True,
        logx_=True,
    )
)

variables.append(
    Variable("bmmj1_mass", r"m(\ell\ell b) [GeV]", 200, 0, 4000, 1e-5, 1e5)
)
variables.append(Variable("min_bl_mass", r"min m(l,b) [GeV]", 100, 0, 600, 1e-5, 1e5))
variables.append(
    Variable("min_b1l_mass", r"min m(l,leading b) [GeV]", 100, 0, 600, 1e-5, 1e5)
)
variables.append(
    Variable("min_b2l_mass", r"min m(l,trailing b) [GeV]", 100, 0, 600, 1e-5, 1e5)
)

variables.append(Variable("njets", r"$N_{jet}$", 10, -0.5, 9.5, 0.5, 1e6))
variables.append(Variable("nbjets", r"$N_{b-tagged jet}$", 10, -0.5, 9.5, 0.5, 1e6))
variables.append(
    Variable("met", r"$E_{\mathrm{T}}^{\mathrm{miss}} [GeV]$", 20, 0, 200, 0.5, 1e6)
)
variables.append(
    Variable("dimuon_cos_theta_cs", r"$cos\theta_{\mathrm{CS}}$", 20, -1, 1, 0.5, 1e6)
)
variables.append(
    Variable("bjet1_pt", r"Leading b-jet $p_{T}$ [GeV]",100, 0, 500, 1e-5, 1e5)
)
variables.append(
    Variable("bjet2_pt", r"Subleading b-jet $p_{T}$ [GeV]",100, 0, 500, 1e-5, 1e5)
)
variables.append(
    Variable("mu1_pt", r"Leading muon $p_{T}$ [GeV]",100, 0, 500, 1e-5, 1e5)
)
variables.append(
    Variable("mu1_eta", r"Leading muon $\eta$", 50, -2.5, 2.5, 1e-5, 1e5)
)
variables.append(
    Variable("mu1_phi", r"Leading muon $\phi$", 50, -3.2, 3.2, 1e-5, 1e5)
)
variables.append(
    Variable("mu2_pt", r"Subleading muon $p_{T}$ [GeV]",100, 0, 500, 1e-5, 1e5)
)   
variables_lookup = {}
for v in variables:
    variables_lookup[v.name] = v
