import os
os.environ['PROJ_LIB'] = '/home/sdfgh5343/anaconda3/share/proj'
import re
import warnings
import traceback
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from   cmocean import cm
from   scipy.interpolate import griddata
from   mpl_toolkits.basemap import Basemap

class grid():
    """Loading hgrid.ll file information"""
    def __init__(self,filename):
        """
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
class temperature:
    """Assigning the setting of plotting Temperature figure"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.var   = 'Temperature'
        self.shade = [20,30,21]
        self.cbar  = [20,30.1,11]
        self.cmap  = cm.thermal
        self.unit  = '℃'
        self.date = None
class salinity:
    """Assigning the setting of plotting Salinity figure"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.var   = 'Salinity'
        self.shade = [34,37,21]
        self.cbar  = [34,37.1,11]
        self.cmap  = cm.haline
        self.unit  = 'psu'
        self.date = None
class velocity:
    """Assigning the setting of plotting Current Velocity figure"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.var   = 'Current Velocity'
        self.shade = [0,1.5,21]
        self.cbar  = [0,1.5,11]
        self.cmap  = "rainbow"
        self.unit  = 'm/s'
        self.date = None
class wind:
    """Assigning the setting of plotting Wind Velocity figure"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.var   = 'Wind Velocity'
        self.shade = [0,50,21]
        self.cbar  = [0,50,11]
        self.cmap  = "rainbow"
        self.unit  = 'm/s'
        self.date = None
class variable_reset(temperature,salinity,velocity,wind):
    temperature()
    salinity()
    velocity()
    wind()
class self_defining:
    """
    Assigning the setting of plotting Wind Velocity figure.
    Date assigning is optional.
    """
    def __init__(self,var, shade, cbar, cmap, unit, date=None):
        """
        -------
        var : Variable, string ; 
        shade: Shade range in form of [0,50,space] ; 
        cbar  : Colorbar range in form of [0,50,sapce] ; 
        cmap  : Colormap ; 
        unit : Variable unit.
        -------
        """
        var   = var
        shade = shade
        cbar  = cbar
        cmap  = cmap
        unit  = unit
        date = date


class Interpolation:
    """
    Requring grid data loading from schism_tools.grid, xarray, 
    yarray. Data is defaulted to depth if data is not assigned.
    Parameters
    ----------
    class_grid : Grid information.
    xarray : x-corrdinate in 2D array.
    yarray : y-corrdinate in 2D array.
    data : 1D array for interpolation, optional.
    -------
    """
    def __init__(self, class_grid, xarray, yarray):
        """
        """
        self.lon = class_grid.lon
        self.lat = class_grid.lat
        self.depth = class_grid.depth
        self.xarray = xarray
        self.yarray = yarray
        self.element = class_grid.element
        self.triang  = mtri.Triangulation(self.lon,self.lat,self.element)
        self.location_cal()
        self.weight_cal()
        
    def location_cal(self):
        func = self.triang.get_trifinder()
        loc = func(self.xarray, self.yarray)
        self.loc = np.int64(loc)
        
    def weight_cal(self):
        loc = self.element[self.loc,:]
        ax1 = self.lon[loc[:,:,0]]
        ax2 = self.lon[loc[:,:,1]]
        ax3 = self.lon[loc[:,:,2]]
        ay1 = self.lat[loc[:,:,0]]
        ay2 = self.lat[loc[:,:,1]]
        ay3 = self.lat[loc[:,:,2]]
        upw1 = (ay2-ay3)*(self.xarray-ax3)+(ax3-ax2)*(self.yarray-ay3)
        low1 = (ay2-ay3)*(ax1-ax3)+(ax3-ax2)*(ay1-ay3)
        upw2 = (ay3-ay1)*(self.xarray-ax3)+(ax1-ax3)*(self.yarray-ay3)
        low2 = (ay2-ay3)*(ax1-ax3)+(ax3-ax2)*(ay1-ay3)
        aw1 = upw1/low1; aw2 = upw2/low2; aw3 = 1-aw2-aw1
        self.weight = np.array([aw1,aw2,aw3])
        
    def interp2structure(self,data):
        """Interpolating from unstructure1D to structure2D"""
        struc = self.weight[0,:,:]*data[self.element[self.loc[:],0]]+\
                self.weight[1,:,:]*data[self.element[self.loc[:],1]]+\
                self.weight[2,:,:]*data[self.element[self.loc[:],2]]
        struc[np.where(self.loc==-1)]=np.nan
        return struc


class Angle:
    """Solving Wind direction and the angle between two vectors"""
    def vector_direction(u,v):
        return (270-np.rad2deg(np.arctan2(v,u)))%360 
    
    def vel_direction(u,v):
        return (360+np.rad2deg(np.arctan2(v,u)))%360
    
    def vecters_angle(u1,u2,v1,v2):
        vel1 = np.sqrt(u1**2+v1**2)
        vel2 = np.sqrt(u2**2+v2**2)
        upper = u1*u2+v1*v2
        bottom = vel1*vel2
        return np.rad2deg(np.arccos(upper/bottom))


class plot:
    """
    Assigning varaiable, shade range, colorbar range, cmap range
    Date and title is deflaulted to None.
    """
    def __init__(self,var):
        """
        Parameters
        ----------
        var : "temperature", "salinity", "velocity, "wind"
        
        Returns variable, shade, cabr, cmap, date 
        -------
        """
        self.variable = var.var
        self.shade    = np.linspace(var.shade[0],var.shade[1],var.shade[2])
        self.cbar     = np.linspace(var.cbar[0],var.cbar[1],var.cbar[2])
        self.cmap     = var.cmap
        self.date     = var.date
        self.llcrnr = [115, 20]
        self.urcrnr = [125, 30]
        
    def show_init(self):
        print('Variable:\n  ',self.variable)
        print('Shade range:\n  ',self.shade)
        print('Colorbar range:\n  ',self.cbar)
        print('Colormap:\n  ',self.cmap)
        print('Date:\n  ',self.date)
        print("Region:\n",self.llcrnr,self.urcrnr)
        
    def set_region(self,lonmin=115,lonmax=125,latmin=20,latmax=30):
        self.llcrnr = [lonmin, latmin]
        self.urcrnr = [lonmax, latmax]
        
    def plot_unstruc_grid(self,triang,data,title=None,save=None,land='full'):
        warnings.filterwarnings("ignore", category=UserWarning)
        plt.figure(figsize=(12,10),dpi=300)
        m = Basemap(llcrnrlat = self.llcrnr[1], urcrnrlat = self.urcrnr[1],\
                    llcrnrlon = self.llcrnr[0], urcrnrlon = self.urcrnr[0],\
                    resolution = 'l')
        cs = plt.tricontourf(triang, data, self.shade,cmap=self.cmap,extend='both')
        cbar = m.colorbar(cs,ticks=self.cbar, location='right',pad= 0.4,size="3%")
        cbar.ax.tick_params(labelsize = 15)

        m.drawparallels(np.linspace(self.llcrnr[1], self.urcrnr[1],11),\
                        labels= [1,0,0,0],fontsize= 15,linewidth= 0.0)
        m.drawmeridians(np.linspace(self.llcrnr[0], self.urcrnr[0],11),\
                        labels= [0,0,0,1],fontsize= 15,linewidth= 0.0)

        if land=='full':
            m.drawcoastlines()
            m.fillcontinents(color = 'gray', lake_color = 'gray')
        elif land=='line':
            m.drawcoastlines()
        elif land is None:
            pass
        
        if title is None:
            plt.title(self.variable,fontsize=30,pad=30)
        else:
            plt.title(title,fontsize=30,pad=30)
        
        if self.date is not None:
            plt.text(self.urcrnr[0],self.urcrnr[1]+(self.urcrnr[1]-self.urcrnr[1])/100,\
                     "Date:%s"%self.date.strftime("%Y%m%d"),fontsize=20)
        20
        if save is None:
            plt.show()
        else:
            plt.savefig(save)
        plt.close()
        
    def plot_struc_grid(self,X,Y,data,title=None,save=None,land='full'):
        warnings.filterwarnings("ignore", category=UserWarning)
        plt.figure(figsize=(12,10),dpi=300)
        m = Basemap(llcrnrlat = self.llcrnr[1], urcrnrlat = self.urcrnr[1],\
                    llcrnrlon = self.llcrnr[0], urcrnrlon = self.urcrnr[0],\
                    resolution = 'l')
        cs = plt.contourf(X,Y,data, self.shade,cmap=self.cmap,extend='both')
        cbar = m.colorbar(cs,ticks=self.cbar, location='right',pad= 0.4,size="3%")
        cbar.ax.tick_params(labelsize = 15)

        m.drawparallels(np.linspace(self.llcrnr[1], self.urcrnr[1],11),\
                        labels= [1,0,0,0],fontsize= 15,linewidth= 0.0)
        m.drawmeridians(np.linspace(self.llcrnr[0], self.urcrnr[0],11),\
                        labels= [0,0,0,1],fontsize= 15,linewidth= 0.0)
        if land=='full':
            m.drawcoastlines()
            m.fillcontinents(color = 'gray', lake_color = 'gray')
        elif land=='line':
            m.drawcoastlines()
        elif land is None:
            pass
        
        if title is None:
            plt.title(self.variable,fontsize=30,pad=30)
        else:
            plt.title(title,fontsize=30,pad=30)
        
        if self.date is not None:
            plt.text(self.urcrnr[0],self.urcrnr[1]+(self.urcrnr[1]-self.urcrnr[1])/100,\
                     "Date:%s"%self.date.strftime("%Y%m%d"),fontsize=20)
        if save is None:
            plt.show()
        else:
            plt.savefig(save)
        plt.close()

if __name__ == '__main__':
    temperature()
    salinity()
    velocity()
    wind()
