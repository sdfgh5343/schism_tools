import re
import numpy as np
import pandas as pd
import matplotlib.tri as mtri

def __init__(self,filename):
    """
    Loading hgrid.ll file information
    Parameters
    ----------
    filename : Filename including path.
    Returns lon, lat, dep, elem storing into class
    -------
    """
    self.path = filename
    with open(self.path,'r') as rf:
        rf.readline()
        elem,node=re.findall('(\d+)',rf.readline())
        elem=int(elem);node=int(node)
    nodeAll=pd.read_csv(self.path,sep='\s+',header=None,skiprows=2,nrows=node)
    self.lon = np.float64(nodeAll[1])
    self.lat = np.float64(nodeAll[2])
    self.depth   = np.float64(nodeAll[3])
    elemAll = pd.read_csv(self.path,sep='\s+',header=None,skiprows=2+node,nrows=elem)
    self.element = np.int64([elemAll[2],elemAll[3],elemAll[4]]).transpose()-1
def triang(self):
    return mtri.Triangulation(self.lon,self.lat,self.element)
