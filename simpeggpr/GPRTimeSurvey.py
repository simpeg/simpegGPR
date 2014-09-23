from SimPEG import *
from SimPEG.Utils import sdiag, mkvc, sdInv, speye

class SurveyGPRTime(Survey.BaseSurvey):
    """
        **SurveyAcousitc**

        Geophysical Acoustic Wave data.

    """

    def __init__(self, txList,**kwargs):
        self.txList = txList
        Survey.BaseSurvey.__init__(self, **kwargs)

    def projectFields(self, u):
        data = {}

        for i, tx in enumerate(self.txList):
            for rx in tx.rxList:

                Proj = rx.getP(self.prob.mesh)
                if rx.rxtype.find('E') >= 0:
                    flag = 'E'
                elif rx.rxtype.find('H') >= 0:
                    flag = 'H'
                data[tx, rx] = (Proj*u[flag, tx])

        return data
