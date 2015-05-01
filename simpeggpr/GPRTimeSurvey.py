from SimPEG import *
from SimPEG.Utils import sdiag, mkvc, sdInv, speye

class SurveyGPRTime(Survey.BaseSurvey):
    """
        **SurveyAcousitc**

        Geophysical Acoustic Wave data.

    """

    def __init__(self, srcList,**kwargs):
        self.srcList = srcList
        Survey.BaseSurvey.__init__(self, **kwargs)

    def projectFields(self, u):
        data = {}

        for i, src in enumerate(self.srcList):
            for rx in src.rxList:

                Proj = rx.getP(self.prob.mesh)
                if rx.rxtype.find('E') >= 0:
                    flag = 'E'
                elif rx.rxtype.find('H') >= 0:
                    flag = 'H'
                data[src, rx] = (Proj*u[flag, src])

        return data
