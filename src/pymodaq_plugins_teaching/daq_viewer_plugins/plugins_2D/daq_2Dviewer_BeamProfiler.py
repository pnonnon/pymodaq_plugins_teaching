from numba.cuda.cudadrv.driver import profiling
from qtpy.QtCore import QThread, Slot, QRectF
from qtpy import QtWidgets
import numpy as np
from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.parameter import Parameter
import laserbeamsize as lbs
from pymodaq_plugins_mockexamples.daq_viewer_plugins.plugins_2D.daq_2Dviewer_BSCamera import DAQ_2DViewer_BSCamera

class DAQ_2DViewer_BeamProfiler(DAQ_2DViewer_BSCamera):
    live_mode_available = False
    def grab_data(self, Naverage=1, **kwargs):
        """Start a grab from the detector

        Parameters
        ----------
        Naverage: int
            Number of hardware averaging (if hardware averaging is possible, self.hardware_averaging should be set to
            True in class preamble and you should code this implementation)
        kwargs: dict
            others optionals arguments
        """
        dte = self.average_data(Naverage)  # hardware averaging
        dwa_xy, dwa_dxy, dwa_phi = self.profiling(dte.data[0][0])

        dte.append(dwa_xy)
        dte.append(dwa_dxy)
        dte.append(dwa_phi)

        self.dte_signal.emit(dte)

    def profiling(self, data_array : np.ndarray):
        x,y,dx,dy, phi = lbs.beam_size(data_array)
        dwa_xy = DataFromPlugins('xy',
                                 data=[np.array([x]), np.array([y])],
                                 labels = ['X0, Y0']
                                 )
        dwa_dxy = DataFromPlugins('dxy',
                                  data=[np.array([dx]), np.array([dy])],
                                  labels=['dX, dY']
                                  )
        dwa_phi = DataFromPlugins('phi',
                                  data=[np.array([phi])],
                                  labels=['Phi'])
        return dwa_xy, dwa_dxy, dwa_phi



if __name__ == '__main__':
    main(__file__)
