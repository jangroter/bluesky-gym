''' This module is deprecated and will be removed in a later version of BlueSky-gym. '''
from warnings import deprecated
from bluesky.simulation import ScreenIO


@deprecated('Subclassing ScreenIO in detached simulations is no longer necessary for BlueSky>=1.0.0')
class ScreenDummy(ScreenIO):
    """
    Dummy class for the screen. Inherits from ScreenIO to make sure all the
    necessary methods are there. This class is there to reimplement the echo
    method so that console messages are ignored.
    """
    def echo(self, text='', flags=0):
        pass