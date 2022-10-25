import numpy as np


def calcMassScaleUncert(mass, isBB, year, up=True):

    scaleUnc = 0.0

    if (year == 2016):
        if (isBB):
            scaleUnc = -0.02
        else:
            scaleUnc = -0.01

    elif (year == 2017):
        if (isBB):
            scaleUnc = -0.02
        else:
            scaleUnc = -0.01

    elif (year == 2018):
        if (isBB):
            scaleUnc = -0.02
        else:
            scaleUnc = -0.01

    if up:
        scaleUnc = 1 - scaleUnc
    else:
        scaleUnc = 1 + scaleUnc

    return scaleUnc


def electronScaleUncert(masses, isBB, year, up=True):

    result = np.array([calcMassScaleUncert(mass, isBB, int(year), up) for mass in masses])
    return result
