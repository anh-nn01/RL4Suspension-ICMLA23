import random
import numpy as np

"""======================================
    SIMULATE SINE ROAD PROFILES 
======================================"""
def generate_road(TIME, MAX_BUMP, mode='hump'):
    """ Create array of road profile"""
    road_profile = [t for t in range(len(TIME))]
    for t_idx, t in enumerate(TIME):
        # road_profile[t_idx]= random.uniform(-0.005, 0.005) # No roughness by default
        road_profile[t_idx] = 0.0

        """ Model Speed bumps"""
        if mode == 'hump':
            if t >= 1 and t <= 1.5:
                # road_profile[t_idx]=MAX_BUMP
                t0 = t-1
                road_profile[t_idx]= -(16*MAX_BUMP) * (t0)**2 + 8*MAX_BUMP * (t0)
#             if t >= 2 and t <= 2.5:
#                 # road_profile[t_idx]=MAX_BUMP
#                 t0 = t-2
#                 road_profile[t_idx]= -(16*MAX_BUMP) * (t0)**2 + 8*MAX_BUMP * (t0)
#             if t >= 3 and t <= 3.5:
#                 # road_profile[t_idx]=MAX_BUMP
#                 t0 = t-3
#                 road_profile[t_idx]= -(16*MAX_BUMP) * (t0)**2 + 8*MAX_BUMP * (t0)
#             if t >= 4 and t <= 4.5:
#                 # road_profile[t_idx]=MAX_BUMP
#                 t0 = t-4
#                 road_profile[t_idx]= -(16*MAX_BUMP) * (t0)**2 + 8*MAX_BUMP * (t0)
#             if t >= 5 and t <= 5.5:
#                 # road_profile[t_idx]=MAX_BUMP
#                 t0 = t-5
#                 road_profile[t_idx]= -(16*MAX_BUMP) * (t0)**2 + 8*MAX_BUMP * (t0)
            # if t >= 2 and t <= 2.2:
            #     road_profile[t_idx]=MAX_BUMP
            # elif t >= 3 and t <= 3.2:
            #     road_profile[t_idx]=MAX_BUMP
            # elif t >= 4 and t <= 4.2:
            #     road_profile[t_idx]=MAX_BUMP
            # elif t >= 23 and t <= 23.2:
            #     road_profile[t_idx]=MAX_BUMP

            # """ Model trough"""
            # if t >= 6 and t <= 6.2:
            #     road_profile[t_idx]=-MAX_BUMP
        elif mode == 'rectangular':
            if t >= 1 and t <= 5:
                road_profile[t_idx] = 1.5

    return road_profile
    
"""======================================
    ISO Road Profile Generator
======================================"""
class RoadProfile(object):

    '''
    Based on method described in:
        Da Silva, J. G. S. "Dynamical performance of highway bridge
        decks with irregular pavement surface."
        Computers & structures 82.11 (2004): 871-881.

    Attributes:
        Gdn0 (float): Gd(n0)
        n0 (float): reference spatial frequency
        n_max (float): max spatial frequency
        n_min (float): min spatial frequency
        w (int): set according to the value in page 14 of ISO
    '''

    n_min = 0.0078  # min spatial frequency
    n_max = 40.  # max spatial frequency
    n0 = 0.1  # reference spatial frequency
    w = 2  # set according to the value in page 14 of ISO
    iso_Gdn0 = {"A": 32E-6,
                "B": 128E-6,
                "C": 512E-6,
                "D": 2048E-6,
                "E": 8192E-6,
                "F": 32768E-6}

    def __init__(self, Gdn0=32E-6):
        '''Gdn0, displacement power spectral density (m**3)

        Args:
            Gdn0 (float, optional): Gd(n0)
        '''
        self.Gdn0 = Gdn0

    def generate(self, L=100., dx=0.1, center = False):
        """Summary

        Args:
            L (float, optional): Length of the road profile
            dx (float, optional): Interval between two points
            center (bool, optional): Center the profile mean to zero

        Returns:
            array : Road profile
        """
        x = np.arange(0, L+dx/2., dx)
        components = int(L/dx/2)
        ns = np.linspace(self.n_min, self.n_max, components)

        Gd = np.sqrt(self.Gdn0 * (ns/self.n0)**(-self.w) *
                     2*(self.n_max - self.n_min)/components)

        profile = np.zeros(np.size(x))
        phase = np.random.rand(components)*2*np.pi
        for i in range(len(x)):
            profile[i] = np.sum(Gd*np.cos(2*np.pi*ns*x[i]-phase))

        if center:
            profile -= np.mean(profile)

        return [x,profile]

    def set_profile_class(self, profile_class="A"):
        """Summary

        Args:
            profile_class (str, optional): A-F
        """
        try:
            self.Gdn0 = self.iso_Gdn0[profile_class.upper()]
        except:
            raise ValueError("Profile name is not predefined.")

    def get_profile_by_class(self, profile_class="A", L=100, dx=0.1, center = False):
        """Summary

        Args:
            profile_class (str, optional): A-F
            L (int, optional): Length of the road profile
            dx (float, optional): Interval between two points
            center (bool, optional): Center the profile mean to zero

        Returns:
            array: Road profile

        """
        try:
            self.Gdn0 = self.iso_Gdn0[profile_class.upper()]
            return self.generate(L, dx)
        except:
            raise ValueError("Profile name is not predefined.")