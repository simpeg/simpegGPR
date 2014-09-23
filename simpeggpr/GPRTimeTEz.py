from SimPEG import *
from SimPEG.Utils import sdiag, mkvc, sdInv, speye
import matplotlib.pyplot as plt
from time import clock
from scipy.constants import mu_0, epsilon_0
from GPRTimeSurvey import SurveyGPRTime

class GPRTEzTx(Survey.BaseTx):


    def __init__(self, loc, time, rxList, txType='Mz', **kwargs):

        self.dt = time[1]-time[0]
        self.time = time
        self.loc = loc
        self.rxList = rxList
        self.txType = txType
        self.kwargs = kwargs


    def RickerWavelet(self):

        """
            Generating Ricker Wavelet

            .. math ::

        """
        tlag = self.kwargs['tlag']
        fmain = self.kwargs['fmain']
        t = self.time
        self.wave = np.exp(-2*fmain**2*(t-tlag)**2)*np.cos(np.pi*fmain*(t-tlag))
        return self.wave

    def Wave(self, tInd):

        """
            Generating Ricker Wavelet

            .. math ::

        """
        tlag = self.kwargs['tlag']
        fmain = self.kwargs['fmain']
        t = self.time[tInd]
        self.wave = np.exp(-2*fmain**2*(t-tlag)**2)*np.cos(np.pi*fmain*(t-tlag))
        return self.wave

    def getq(self, mesh):

        if self.txType=='Mz':

            txind = Utils.closestPoints(mesh, self.loc, gridLoc='CC')
            jm = np.zeros(mesh.nC)
            jm[txind] = 1./mesh.vol[txind]
            return np.r_[jm, jm]*0.5, np.zeros(mesh.nE)

        elif self.txType=='Jx':
            txind = Utils.closestPoints(mesh, self.loc, gridLoc='Ex')
            je = np.zeros(mesh.nE)
            je[txind] = 1./mesh.edge[txind]

            return np.zeros(2*mesh.nC), je

        elif self.txType=='Jy':
            txind = Utils.closestPoints(mesh, self.loc, gridLoc='Ey')
            je = np.zeros(mesh.nE)
            je[txind] = 1./mesh.edge[txind]
            return np.zeros(2*mesh.nC), je

        else:
            Exception("Not implemented!!")

class GPRTEzRx(Survey.BaseRx):

    def __init__(self, locs, rxtype, **kwargs):
        self.locs = locs
        self.rxtype = rxtype
        self._Ps = {}
    @property
    def nD(self):
        """ The number of data in the receiver."""
        return self.locs.shape[0]

    def getP(self, mesh):
        # TODO: need to be changed: do not generate every time
        if self.rxtype == 'Ex':
            P = mesh.getInterpolationMat(self.locs, 'Ex')
        elif self.rxtype == 'Ey':
            P = mesh.getInterpolationMat(self.locs, 'Ey')
        elif self.rxtype == 'Hz':
            P = mesh.getInterpolationMat(self.locs, 'CC')
        return P

class GPR2DTEzProblemPML(Problem.BaseProblem):
    """

    """
    surveyPair = SurveyGPRTime
    Solver     = Solver
    storefield = True
    verbose = False
    stability = False
    sigx = False

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh)
        Utils.setKwargs(self, **kwargs)

    def setPMLBC(self, npad, dt, sm=3., Rth=1e-8):
        ax = self.mesh.vectorCCx[-npad]
        ay = self.mesh.vectorCCy[-npad]

        indy = np.logical_or(self.mesh.gridCC[:,1]<=-ay, self.mesh.gridCC[:,1]>=ay)
        indx = np.logical_or(self.mesh.gridCC[:,0]<=-ax, self.mesh.gridCC[:,0]>=ax)

        tempx = np.zeros_like(self.mesh.gridCC[:,0])
        tempx[indx] = (abs(self.mesh.gridCC[:,0][indx])-ax)**2
        tempx[indx] = tempx[indx]-tempx[indx].min()
        tempx[indx] = tempx[indx]/tempx[indx].max()
        tempy = np.zeros_like(self.mesh.gridCC[:,1])
        tempy[indy] = (abs(self.mesh.gridCC[:,1][indy])-ay)**2
        tempy[indy] = tempy[indy]-tempy[indy].min()
        tempy[indy] = tempy[indy]/tempy[indy].max()

        self.tempx = tempx
        self.tempy = tempy

        self.Lx = self.mesh.hx[-npad:].sum()
        self.Ly = self.mesh.hy[-npad:].sum()
        self.sm = sm
        self.Rth= Rth


    def stabilitycheck(self, epsilon, mu, sig0, time, fmain, sigs=0.):

        self.epsilon = epsilon
        self.mu = mu
        self.sig0 = sig0
        self.dxmin = min(self.mesh.hx.min(), self.mesh.hy.min())
        self.c = 1./np.sqrt(self.epsilon*self.mu)
        self.topt = self.dxmin/self.c.max()*0.5
        self.dt = time[1]-time[0]
        self.fmain = fmain
        self.wavelen = self.c.min()/self.fmain
        self.G = self.wavelen/self.dxmin
        self.sigm = -(self.epsilon.max()*self.c.max()/(0.5*(self.Lx+self.Ly)))/(1.+self.sm*(1./3+2./(np.pi**2)))*np.log(self.Rth)
        self.sm_ref = self.wavelen/(2*self.mesh.hx.min())-1.
        self.sigx = self.sigm*np.sin(0.5*np.pi*np.sqrt(self.tempx))**2
        self.sigy = self.sigm*np.sin(0.5*np.pi*np.sqrt(self.tempy))**2
        self.sx0 = 1.+self.sm*self.tempx
        self.sy0 = 1.+self.sm*self.tempy
        self.sigs = np.ones_like(self.sigx)*sigs


        if self.dt > self.topt:
            print "Warning: dt is greater than topt"
            self.stability = False
        elif self.G < 0.1:
            print "Warning: Wavelength per cell (G) should be greater than 0.5"
            self.stability = False
        elif self.sm < self.sm_ref:
            self.stability = False
            print ("sm should be smaller than %5.2e") % (self.sm_ref)
        else:
            print "You are good to go:)"
            self.stability = True

        print ">> Stability information"
        print ("   dt: %5.2e s")%(self.dt)
        print ("   Optimal dt: %5.2e s")%(self.topt)
        print ("   Cell per wavelength (G): %5.2e")%(self.G)
        print ("   Optimal G: %5.2e")%(1.2)
        print ('>> sm: %5.2e, lamda: %5.2e, sigm: %5.2e') % (self.sm, self.wavelen, self.sigm)

    def fields(self, epsilon, mu, sig0):

        Smu = sp.block_diag([sdiag(mu*self.sy0), sdiag(mu*self.sx0)])
        SmuI = sp.block_diag([sdiag(1./(mu*self.sy0)), sdiag(1./(mu*self.sx0))])
        Smuisig = sp.block_diag([sdiag(self.sigs*self.sigy*(1./epsilon)*self.sy0), sdiag(self.sigs*self.sigx*(1./epsilon)*self.sx0)])
        Ssig = sp.block_diag([sdiag((self.sigy*mu/epsilon+self.sigs)*self.sy0), sdiag((self.sigx*mu/epsilon+self.sigs)*self.sx0)])
        Mesepsisig = sdiag(mesh.aveE2CCV.T*np.r_[1./epsilon*sig0*self.sigy*self.sy0, 1./epsilon*sig0*self.sigx*self.sx0])
        Messig = sdiag(mesh.aveE2CCV.T*np.r_[(sig0+self.sigy)*self.sy0, (sig0+self.sigx)*self.sx0])
        Meseps = sdiag(mesh.aveE2CCV.T*np.r_[epsilon*self.sy0, epsilon*self.sx0])
        MesepsI = sdInv(Meseps)
        Icc = sp.hstack((speye(mesh.nC), speye(mesh.nC)))
        curl = mesh.edgeCurl
        curlvec = sp.block_diag((curl[:,:mesh.nEx], curl[:,mesh.nEx:]))

        if self.stability==False:
            raise Exception("Stability condition is not satisfied!!")
        elif self.sigx is False:
            print "Warning: Absorbing boundary condition was not set yet!!"
        start = clock()
        print ""
        print "***** Start Computing Electromagnetic Wave *****"
        print ""
        print (">> dt: %5.2e s")%(self.dt)
        print (">> Optimal dt: %5.2e s")%(self.topt)
        print (">> Main frequency, fmain: %5.2e Hz")%(self.fmain)
        print (">> Cell per wavelength (G): %5.2e")%(self.G)


        if self.storefield==True:
            self._Fields ={}
            #TODO: parallize in terms of sources
            ntx = len(self.survey.txList)
            for itx, tx in enumerate(self.survey.txList):
                print ("  Tx at (%7.2f, %7.2f): %4i/%4i")%(tx.loc[0], tx.loc[0], itx+1, ntx)
                hd0 = np.zeros(2*mesh.nC)
                hd1 = np.zeros(2*mesh.nC)
                hId0 = np.zeros(2*mesh.nC)
                hId1 = np.zeros(2*mesh.nC)
                e0 = np.zeros(mesh.nE)
                e1 = np.zeros(mesh.nE)
                eI0 = np.zeros(mesh.nE)
                eI1 = np.zeros(mesh.nE)
                time = tx.time
                dt = tx.dt
                jm, je = tx.getq(self.mesh)
                h = np.zeros((mesh.nC, time.size))
                e = np.zeros((mesh.nE, time.size))

                for i in range(time.size-1):
                    eI0 = eI1.copy()
                    eI1 = eI0 + dt*e0
                    e1 = e0 + MesepsI*dt*(curl.T*(Icc*hd1)-Messig*e0-Mesepsisig*eI1-je*wave[i])
                    e0 = e1.copy()
                    e[:,i] = e1
                    hId0 = hId1.copy()
                    hId1 = hId0 + dt*hd0
                    hd1 = hd0 - SmuI*dt*(curlvec*e0+Ssig*hd0+Smuisig*hId1+jm*wave[i]*0.5)
                    hd0 = hd1.copy()
                    h[:,i] = Icc*hd1

                self._Fields['E', tx]= e
                self._Fields['H', tx]= h
            elapsed = clock()-start
            print (">>Elapsed time: %5.2e s")%(elapsed)

            return self._Fields


        elif self.storefield==False:
            Data = {}
            ntx = len(self.survey.txList)
            for itx, tx in enumerate(self.survey.txList):
                print ("  Tx at (%7.2f, %7.2f): %4i/%4i")%(tx.loc[0], tx.loc[0], itx+1, ntx)
                hd0 = np.zeros(2*mesh.nC)
                hd1 = np.zeros(2*mesh.nC)
                hId0 = np.zeros(2*mesh.nC)
                hId1 = np.zeros(2*mesh.nC)
                e0 = np.zeros(mesh.nE)
                e1 = np.zeros(mesh.nE)
                eI0 = np.zeros(mesh.nE)
                eI1 = np.zeros(mesh.nE)
                time = tx.time
                dt = tx.dt
                jm, je = tx.getq(self.mesh)
                h = np.zeros((mesh.nC, time.size))
                e = np.zeros((mesh.nE, time.size))

                for i in range(time.size-1):
                    eI0 = eI1.copy()
                    eI1 = eI0 + dt*e0
                    e1 = e0 + MesepsI*dt*(curl.T*(Icc*hd1)-Messig*e0-Mesepsisig*eI1-je*wave[i])
                    e0 = e1.copy()
                    e[:,i] = e1
                    hId0 = hId1.copy()
                    hId1 = hId0 + dt*hd0
                    hd1 = hd0 - SmuI*dt*(curlvec*e0+Ssig*hd0+Smuisig*hId1+jm*wave[i])
                    hd0 = hd1.copy()
                    h[:,i] = Icc*hd1

                for rx in tx.rxList:
                    Proj = rx.getP(self.mesh)
                    if rx.rxtype.find('E') >= 0:
                        flag = 'E'
                        Data[tx, rx] = (Proj*e)
                    elif rx.rxtype.find('H') >= 0:
                        flag = 'H'
                        Data[tx, rx] = (Proj*h)

            elapsed = clock()-start
            print (">>Elapsed time: %5.2e s")%(elapsed)

            return Data


if __name__ == '__main__':


    dt = 1e-11
    fmain = 3e9
    time = np.arange(650)*dt
    options={'tlag':50*dt, 'fmain':fmain}
    rx = GPRTEzRx(np.r_[0, 0.], 'Ex')
    tx = GPRTEzTx(np.r_[0, 0.], time, [rx], txType='Mz', **options)
    survey = SurveyGPRTime([tx])
    wave = tx.RickerWavelet()
    cs =  1.0*1e-2
    hx = np.ones(200)*cs
    hy = np.ones(200)*cs
    mesh = Mesh.TensorMesh([hx, hy], 'CC')
    prob = GPR2DTEzProblemPML(mesh)
    prob.pair(survey)
    epsilon = epsilon_0*np.ones(mesh.nC)*1.
    epsilon[mesh.gridCC[:,1]<0.5] = epsilon_0*2.
    mu = mu_0*np.ones(mesh.nC)
    sighalf = 1e-3
    sig0 = sighalf*np.ones(mesh.nC)
    prob.setPMLBC(30, dt)
    prob.stabilitycheck(epsilon, mu, sig0, time, fmain, sigs=0.)
    storefield = True
    if storefield == False:
        prob.storefield = False
        Data = prob.fields(epsilon, mu, sig0)
        plt.plot(time, Utils.mkvc(Data[tx,rx]))

    elif storefield == True:
        Fields = prob.fields(epsilon, mu, sig0)
        icount = 600
        extent = [mesh.vectorCCx.min(), mesh.vectorCCx.max(), mesh.vectorCCy.min(), mesh.vectorCCy.max()]

        plt.imshow(np.flipud(Fields['H', tx][:,icount].reshape((mesh.nCx, mesh.nCy), order = 'F').T), cmap = 'RdBu', extent=extent)
        plt.show()

        data = survey.projectFields(Fields)
        plt.plot(time, Utils.mkvc(data[tx, rx]))

    plt.show()
