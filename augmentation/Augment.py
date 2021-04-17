#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


# In[2]:


class Augment():

    """
    Augmentation methods. Adjusts labels after applying methods.

    Parameters
    ----------

    dim: tuple
    Dimension of input traces.

    n_channels: int, default=3
    Number of channels.

    stretching_gap: int, default=3
    Stretching scale of time series

    """

    def __init__(self,
                 dim=6000,
                 n_channels=3,
                 stretching_gap=3):
        self.dim = dim
        self.n_channels = n_channels
        self.stretching_gap = stretching_gap

    def _stretch(self, data, stretching_gap):
        """

        Stretch data with the scale of stretching gap by using
        pandas.DataFrame.interpole "polynomial" method.

        Parameters
        ----------
        data: time series data
        stretching_gap: int, default = 3,
                        the scale of stretching

        Returns
        --------
        Returns numpy array with the stretched shape of the input data

        """

        org_len = data.shape[0]
        org_channels = data.shape[1]
        new_len = org_len * stretching_gap

        if (new_len - 1) % stretching_gap != 0:
            new_len = new_len - (new_len - 1) % stretching_gap

        temp = np.zeros((new_len, org_channels))

        for channel in range(3):
            org_sample = 0
            for sample in range(new_len):

                if sample % stretching_gap == 0:
                    temp[sample][channel] = data[org_sample][channel]
                    org_sample += 1
                else:
                    temp[sample][channel] = np.nan

            temp = pd.DataFrame(temp)

            stretch_data = temp.interpolate(method='polynomial', order=2).to_numpy()

            return stretch_data

    def _stretchLabel(self, spt, sst, coda_end, stretching_gap):
        """
        Adjust P,S wave arrival and coda end times proportional to the stretching scale.

        Parameters
        ----------
        spt: P wave arrival time
        sst: S wave arrival time
        coda_end: coda end time
        stretching_gap: int, default = 3,
                        the scale of stretching

        Returns
        --------
        Returns integer values of adjusted P,S wave arrival and coda end times.

        """

        if spt:
            spt = spt * stretching_gap

        if sst:
            sst = sst * stretching_gap

        if coda_end:
            coda_end = coda_end * stretching_gap

        return spt, sst, coda_end

    def _slideCut(self, data, dim):
        """
        Slide data 5 seconds before P wave arrival time
        and cut as long as original shape.

        Parameters
        ----------
        data: time series data
        dim: int, default = 6000,
             Length of original data

        Returns
        --------
        Returns numpy array with the original shape of the data

        """

        org_len = data.shape[0]
        new_len = dim
        slide = spt - 500  # Slide data 5 seconds(500 sample) before P wave arrival time.

        if slide >= 0 and slide + new_len < org_len:
            slide_data = data[slide: slide + new_len]

        else:
            slide_data = data[0: new_len]

        return slide_data

    def _slideCutLabel(self, spt, sst, coda_end, dim):
        """
        Adjust P,S wave arrival and coda end times proportional to the slideCut function.

        Parameters
        ----------
        spt: P wave arrival time
        sst: S wave arrival time
        coda_end: coda end time
        dim: int, default = 6000,
             Length of original data

        Returns
        --------
        Returns integer values of adjusted P,S wave arrival and coda end times.

        """

        new_len = dim
        new_spt = new_sst = new_coda_end = None
        slide = spt - 500  # Slide data 5 seconds(500 sample) before P wave arrival time.

        if spt - slide >= 0 and spt - slide < new_len:
            new_spt = spt - slide
        else:
            new_spt = None

        if sst - slide >= 0 and sst - slide < new_len:
            new_sst = sst - slide
        else:
            new_sst = None

        if coda_end - slide < new_len:
            new_coda_end = coda_end - slide
        else:
            new_coda_end = new_len  # If the coda_end doesn't fit the conditions, coda_end is defined as the last sample

        if new_spt and new_sst:
            spt = new_spt
            sst = new_sst
            coda_end = new_coda_end

        return spt, sst, coda_end

    def _magnitudeWarp(self, data, sigma=0.2, knot=4):
        """
        Multiplies the magnitude of each time series by a curve that is created by cubic spline

        Parameters
        ----------
        data: time series data
        sigma: scale of gaussian distribution, default is 0.2
        knot: number of joint points of cubic polynomials, default is 4

        Returns
        --------
        Returns numpy array with the sampe shape of the input data

        """

        org_len = data.shape[0]
        org_channels = data.shape[1]

        # independent variables in increasing order.
        ch_generator = np.ones((org_channels, 1))
        random_points = np.arange(0, org_len, (org_len - 1) / (knot + 1))
        independent_vars = (ch_generator * random_points).transpose()  # (6,3)

        # dependent variables
        dependent_vars = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, org_channels))  # (6,3)
        x_range = np.arange(org_len)

        cs_data = []

        for channel in range(org_channels):
            cs = CubicSpline(independent_vars[:, channel], dependent_vars[:, channel])
            cs_data.append(cs(x_range))

        warp_data = data * np.array(cs_data).transpose()

        return warp_data

# In[ ]:
