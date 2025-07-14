# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

from mantid.simpleapi import *
Ei = 701.42
t0 = -7.3821
Ein= 696.08
t0n= -3.4702
Load(Filename='/SNS/ARCS/IPTS-27751/nexus/ARCS_201616.nxs.h5', OutputWorkspace='ARCS_201616.nxs', LoadMonitors=True)
ChangeBinOffset(InputWorkspace='ARCS_201616.nxs', OutputWorkspace='ARCS_201616_s_NE', Offset=t0n)
ConvertUnits(InputWorkspace='ARCS_201616_s_NE', OutputWorkspace='ARCS_201616_NE_E', Target='DeltaE', 
             EMode='Direct', EFixed=Ein)
Rebin(InputWorkspace='ARCS_201616_NE_E', OutputWorkspace='ARCS_201616_NE_E', Params='-200,0.5,600')
hE_NE_sum = SumSpectra(InputWorkspace='ARCS_201616_NE_E', IncludeMonitors=False)

ConvertToMD(InputWorkspace='ARCS_201616_NE_E', QDimensions='|Q|', Q3DFrames='Q_sample', 
            QConversionScales='Q in lattice units',  OutputWorkspace='ARCS_201616_E_MD', MaxRecursionDepth=5)
SaveMD('ARCS_201616_E_MD', FileName='201616-exported.nxs',SaveHistory=False,SaveInstrument=False,SaveSample=False,SaveLogs=False)