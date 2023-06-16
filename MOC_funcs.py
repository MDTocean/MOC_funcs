import matplotlib.pyplot as plt
from matplotlib import gridspec
import xarray as xr
import numpy as np
import sectionate
from numba import njit
from glob import glob
import momlevel
from cmip_basins.basins import generate_basin_codes
from xhistogram.xarray import histogram
import sys
import os
from functools import reduce

def MOC_xsec_nodes_online(
    dir_vars,
    dir_grid,
    section_node_lons,
    section_node_lats,
    file_str_identifier,
    z_layer_var="rho2_l",
    z_inter_var="rho2_i",
    u_transport_var="umo",
    v_transport_var="vmo",
    time_limits=[],
    x_hpoint_1Dvar="xh",
    x_qpoint_1Dvar="xq",
    y_hpoint_1Dvar="yh",
    y_qpoint_1Dvar="yq",
    grid_x_hpoint_1Dvar="xh",
    grid_x_qpoint_1Dvar="xq",
    grid_y_hpoint_1Dvar="yh",
    grid_y_qpoint_1Dvar="yq",
    time_var="time",
    lons_tpoint="geolon",
    lats_tpoint="geolat",
    lons_cpoint="geolon_c",
    lats_cpoint="geolat_c",
    plot_flag=True,
    decode_times_flag=True,
    dmgetout=False,
    zarr_dir=''
):
    """Calculate MOC along a specified cross section  (from section_node_lons etc). 
    Defaults are set to calculate density-space AMOC from online density-space transports, 
    but the directory and variable names can be changed to select any other variable. 
    See description of the inputs below -- also check out the readme to the SigmaZ_funcs package, 
    which provides a README that explains some of the choices of these input types. 
    The function requires Raphael Dussin's sectionate tool (https://github.com/MDTocean/sectionate/tree/dev_MDT), 
    as well as John Krasting's momlevel (https://github.com/MDTocean/momlevel) and cmip_basins tools (https://github.com/MDTocean/cmip_basins)
    And some other packages that can be installed with github. 
    
    Args:
        dir_vars (string): directory containing the output variable files. 
        dir_grid (string): directory cotaining the grid info file
        section_node_lons (vector): 1D vector of longitude nodes that (together with section_node_lats) define a cross section. E.g.: [60.3000, 58.8600, 58.0500, 58.0000, 56.5000]
        section_node_lats (vector): 1D vector of latitude nodes that define the cross section. For use in Raphael Dussin's sectionate tool, which finds all grid cell locations along straight lines between each lon/lat node. E.g [-44.9000, -30.5400, -28.0000, -14.7000, -5.9300]
        file_str_identifier (string): string that identifies a particular output file or set of output files. E.g. ".0196*", which will identify all strings containing '.0196'
        z_layer_var (string): variable name for the vertical coordinate at the grid centre
        z_inter_var (string): variable name for the vertical coordinate at the grid interface
        u_transport_var (string): variable name for the zonal transport
        v_transport_var (string): variable name for the meridional transport
        time_limits (vector): a 2-element vector of the type [time_start,time_limit], using the same units as in the data, that allows the data to be limited within these time margins 
        x_hpoint_1Dvar (string) etc: coordinate names of the 1D lon and lat q and h points
        lons_tpoint (string) etc: variable names for the 2D lon and lat grid at the grid centre
        lons_cpoint (string) etc: variable names for the 2D lon and lat grid at the grid corner
        time_var (string): coordinate name of the time dimension
        plot_flag (logical): if True then the routine will make some basic plots
        decode_times_flag (logical): use metadata to convert time to a date
        dmgetout(logical): GFDL-specific flag that exits the function and outputs the dmget commands that first need to be run in a terminal to retrieve the data. 
        zarr_dir ('string'): If an empty string then it will do nothing. If given a directory, the code will: 1) save the data as equivalent zarr data to that directory if the zarr data doesn't already exist; 2) read the data from that zarr directory if the zarr data does already exist there. Zarr files will be saved separately for every file read in, and saved at '/zarr_dir/dir_vars/' (where dir_vars is given above as input). 
        
    output:
        dsT (xarray dataset):  output from Raphael Dussin's sectionate tool, containing transport and grid information along the specified cross-section
        MOC_mean (xarray dataset): MOC streamfunction calculated along the specified cross section
        MOC_ts (xarray dataset): timeseries of maximum MOC along the specified cross section

    """
    
    ########################
    ### create a list of files to load:

    umo_vars_str=dir_vars+"*"+file_str_identifier+"*."+u_transport_var+".nc"
    vmo_vars_str=dir_vars+"*"+file_str_identifier+"*."+v_transport_var+".nc"
    files_timestep=glob(f"{umo_vars_str}")
    files_timestep+=glob(f"{vmo_vars_str}")
    if dmgetout: 
        print(f"{'dmget '+umo_vars_str+' &'}")
        print(f"{'dmget '+vmo_vars_str+' &'}")
        return [],[],[],[],[]

    ########################
    ### If a zarr_dir is given, modify the list of files to read to/write from there instead:

    if zarr_dir:
        for count, filename in enumerate(files_timestep):
            files_timestep[count] = zarr_dir+filename+'.zarr'
            if os.path.isdir(zarr_dir+filename+'.zarr')==False:
                ds_filename=xr.open_dataset(filename,decode_times=decode_times_flag, chunks={'xh' : 100, 'xq' : 100, 'yh' : 100, 'yq' : 100})
                ds_filename.to_zarr(zarr_dir+filename+'.zarr')

                
    ########################
    ### Read the data from the list of files:
    
    ds = xr.open_mfdataset(files_timestep, decode_times=decode_times_flag)
    grid=xr.open_dataset(dir_grid)
    
    ### If there are not the same number of dimensions of yq and yh, cut off the extra dimension:
    if len(ds[y_qpoint_1Dvar])==len(ds[y_hpoint_1Dvar])+1:
        ds = ds.isel( { y_qpoint_1Dvar: slice(1,len(ds[y_qpoint_1Dvar])) } )
        grid = grid.isel( { y_qpoint_1Dvar: slice(1,len(ds[y_qpoint_1Dvar])) } )
    elif len(ds[y_qpoint_1Dvar])==len(ds[y_hpoint_1Dvar]):
        'this is fine'
    else:
        raise Exception("yq must be more positive than yh or be 1 element larger than yh")
    if len(ds[x_qpoint_1Dvar])==len(ds[x_hpoint_1Dvar])+1:
        ds = ds.isel( { x_qpoint_1Dvar: slice(1,len(ds[x_qpoint_1Dvar])) } )
        grid = grid.isel( { x_qpoint_1Dvar: slice(1,len(ds[x_qpoint_1Dvar])) } )
    elif len(ds[x_qpoint_1Dvar])==len(ds[x_hpoint_1Dvar]):
        'this is fine'
    else:
        raise Exception("xq must be more positive than xh or be 1 element larger than xh")

    #######################
    ### If the 1D coordinate names are not xh,xq, etc, then rename them to x_hpoint_1Dvar, x_qpoint_1Dvar etc (The sectionate tool requries it):
    
    if y_hpoint_1Dvar != "yh":
        ds = ds.rename({y_hpoint_1Dvar: 'yh'})
    if y_qpoint_1Dvar != "yq":
        ds = ds.rename({y_qpoint_1Dvar: 'yq'})
    if x_hpoint_1Dvar != "xh":
        ds = ds.rename({x_hpoint_1Dvar: 'xh'})
    if x_qpoint_1Dvar != "xq":
        ds = ds.rename({x_qpoint_1Dvar: 'xq'})
    if time_var != "time":
        ds = ds.rename({time_var: 'time'})
    
    ### If the grid coordinate names are not xh,xq, etc, then rename them to the specified x_hpoint_1Dvar, x_qpoint_1Dvar etc:
    if grid_y_hpoint_1Dvar != "yh":
        grid = grid.rename({grid_y_hpoint_1Dvar: 'yh'})
    if grid_y_qpoint_1Dvar != "yq":
        grid = grid.rename({grid_y_qpoint_1Dvar: 'yq'})
    if grid_x_hpoint_1Dvar != "xh":
        grid = grid.rename({grid_x_hpoint_1Dvar: 'xh'})
    if grid_x_qpoint_1Dvar != "xq":
        grid = grid.rename({grid_x_qpoint_1Dvar: 'xq'})
    if time_var != "time":
        grid = grid.rename({time_var: 'time'})
    
    ########################
    ### Reduce the spatial domain to fit around the chosen section coordinates (with a 10 deg buffer around it):
    
    lat_range_min=np.abs(ds.yh-(min(section_node_lats)-1)).argmin()
    lat_range_max=np.abs(ds.yh-(max(section_node_lats)+1)).argmin()
    lon_range_min=np.abs(ds.xh-(min(section_node_lons)-1)).argmin()
    lon_range_max=np.abs(ds.xh-(max(section_node_lons)+10)).argmin()
    
    if np.any(time_limits):
        ds_subpolar = ds.sel(yq=slice(ds.yq[lat_range_min],ds.yq[lat_range_max]),xh=slice(ds.xh[lon_range_min],ds.xh[lon_range_max]),yh=slice(ds.yh[lat_range_min],ds.yh[lat_range_max]),xq=slice(ds.xq[lon_range_min],ds.xq[lon_range_max]),time=slice(time_limits[0],time_limits[1]))
    else: 
        ds_subpolar = ds.sel(yq=slice(ds.yq[lat_range_min],ds.yq[lat_range_max]),xh=slice(ds.xh[lon_range_min],ds.xh[lon_range_max]),yh=slice(ds.yh[lat_range_min],ds.yh[lat_range_max]),xq=slice(ds.xq[lon_range_min],ds.xq[lon_range_max]))
    grid_subpolar = grid.sel(yq=slice(ds.yq[lat_range_min],ds.yq[lat_range_max]),xh=slice(ds.xh[lon_range_min],ds.xh[lon_range_max]),yh=slice(ds.yh[lat_range_min],ds.yh[lat_range_max]),xq=slice(ds.xq[lon_range_min],ds.xq[lon_range_max]))
    
    ########################
    ### Run Raf's sectionate tool to grid section coordinates:    #####################
    ### Use sectionate with the xarray dataset of variables to get T,S and rho (if specified in the inputs) along the chosen cross-section:

    
    isec, jsec, xsec, ysec = sectionate.create_section_composite(grid_subpolar['geolon_c'],
                                                                 grid_subpolar['geolat_c'],
                                                                 section_node_lons,
                                                                 section_node_lats)
    
    [corner_offset1,corner_offset2]=sectionate.find_offset_center_corner(grid_subpolar[lons_tpoint], grid_subpolar[lats_tpoint], grid_subpolar[lons_cpoint], grid_subpolar[lats_cpoint])
    
    #####################
    ### Use sectionate with the xarray dataset of variables to get the transport along the chosen cross-section:
    
    dsT = sectionate.MOM6_normal_transport(ds_subpolar, isec, jsec,utr=u_transport_var,vtr=v_transport_var,layer=z_layer_var,interface=z_inter_var,offset_center_x=corner_offset1,offset_center_y=corner_offset2,old_algo=True)
    transp_vals=dsT.uvnormal
    
    ######################
    ### Approximate the grid cell width:
    
    earth_radius=6371000  # in km
    section_gridwidth=earth_radius*sectionate.distance_on_unit_sphere(ysec[0:-1],xsec[0:-1],ysec[1:],xsec[1:])
    
    #####################
    ### calculate the MOC:
    
    MOC_mean=dsT.uvnormal.mean(dim=time_var).sum(dim='sect',skipna=True).cumsum(dim=z_layer_var)/1030/1e6
    MOC_ts=dsT.uvnormal.sum(dim='sect',skipna=True).cumsum(dim=z_layer_var).max(dim=z_layer_var)/1030/1e6

    
    ####################
    ### make simple plots if plog_flag is True:
    
    if plot_flag==1:
        plt.figure(figsize=(8,6)),MOC_mean.plot(y=z_layer_var,ylim=[1033,1037.5])
        plt.figure(figsize=(8,6)),MOC_ts.plot()
        
    return dsT, MOC_mean, MOC_ts, xsec, ysec

def MOC_basin_latrange_online(
    dir_vars,
    dir_grid,
    file_str_identifier,
    lat_range=[],
    lon_range=[],
    depth_range=[],
    basin_list=[],
    z_layer_var="rho2_l",
    z_inter_var="rho2_i",
    v_transport_var="vmo",
    time_limits=[],
    x_hpoint_1Dvar="xh",
    x_qpoint_1Dvar="xq",
    y_hpoint_1Dvar="yh",
    y_qpoint_1Dvar="yq",
    time_var="time",
    lons_tpoint="geolon",
    lats_tpoint="geolat",
    lons_cpoint="geolon_c",
    lats_cpoint="geolat_c",
    decode_times_flag=True,
    dmgetout=False,
    zarr_dir=''
):
    """Calculate MOC over the given range of latitudes (according to lat_range variable). 
    Defaults are set to calculate density-space AMOC from online density-space transports, 
    but the directory and variable names can be changed to select any e.g. offline variables. 
    
    Additional Args (see the MOC_xsec_nodes_online descriptions above):
        lat_range (list): 1-element, or 2-element vector (i.e. [start_lat,end_lat]). Can be left empty (i.e. global). Provide the latitudes, and not the grid indices. Must be in square brackets
        lon_range (list): 1-element, or 2-element vector (i.e. [start_lon,end_lon]). Can be left empty (i.e. global). Provide the longitudes, and not the grid indices. Must be in square brackets
        depth_range (list): 1-element, or 2-element vector (i.e. [start_depth,end_depth]).  Can be left empty. Provide the depths (or densities!), and not the indices. Must be in square brackets
        basin_list (list): list of basins according to John Krasting's generate_basin_codes, e.g. [2,3,5] is [Atl,Pac,Ind]. Others: 1=SO; 4=Arc; 6=Med; 7=Black Sea; 8=Hud Bay(?): 9=Balt. Sea(?); 10=Red Sea(?). 

    output:
        ds (xarray dataset):  dataset of meridional transport for the specific basin(s) (if any) and specified ranges
        grid (xarray dataset):  grid information 
        moc (xarray dataset): MOC streamfunction calculated for the specific basin(s) (if any) and specified ranges
        vmo_zonmean (xarray dataset): zonally-integrated meridional transports across the specific basin(s) (if any) and specified ranges

    """
    
    ########################
    ### create a list of files to load:

    vmo_vars_str=dir_vars+"*"+file_str_identifier+"*."+v_transport_var+".nc"
    files_timestep=glob(f"{vmo_vars_str}")
    if dmgetout: 
        print(f"{'dmget '+vmo_vars_str+' &'}")
        return [],[],[],[]

    ########################
    ### If a zarr_dir is given, modify the list of files to read to/write from there instead:

    if zarr_dir:
        for count, filename in enumerate(files_timestep):
            files_timestep[count] = zarr_dir+filename+'.zarr'
            if os.path.isdir(zarr_dir+filename+'.zarr')==False:
                ds_filename=xr.open_dataset(filename,decode_times=decode_times_flag, chunks={'xh' : 100, 'yh' : 100, 'yq' : 100})
                ds_filename.to_zarr(zarr_dir+filename+'.zarr')

    ########################
    ### Read the data from the list of files:

    ds = xr.open_mfdataset(files_timestep, decode_times=decode_times_flag)
    grid=xr.open_dataset(dir_grid)
    
    ### Cut off the final yq index, since the grid file has a problem there in CM4_hires
    ds = ds.isel( { y_qpoint_1Dvar: slice(0,-1) } )
    grid = grid.isel( { y_qpoint_1Dvar: slice(0,-1) } )
    
    #######################
    ### If the 1D coordinate names are not xh,xq, etc, then rename them to x_hpoint_1Dvar, x_qpoint_1Dvar etc (The sectionate tool requries it):

    if y_hpoint_1Dvar != "yh":
        ds = ds.rename({y_hpoint_1Dvar: 'yh'})
        grid = grid.rename({y_hpoint_1Dvar: 'yh'})
    if y_qpoint_1Dvar != "yq":
        ds = ds.rename({y_qpoint_1Dvar: 'yq'})
        grid = grid.rename({y_qpoint_1Dvar: 'yq'})
    if x_hpoint_1Dvar != "xh":
        ds = ds.rename({x_hpoint_1Dvar: 'xh'})
        grid = grid.rename({x_hpoint_1Dvar: 'xh'})
    if x_qpoint_1Dvar != "xq":
        ds = ds.rename({x_qpoint_1Dvar: 'xq'})
        grid = grid.rename({x_qpoint_1Dvar: 'xq'})
    if time_var != "time":
        ds = ds.rename({time_var: 'time'})
        grid = grid.rename({time_var: 'time'})    

    #######################
    ###bit of a dirty fix -- sometimes the grid and variable latitudes don't match (e.g. for the odiv209 _d2 grid):
    
    if ds['yq'].values[0]!=grid['yq'].values[0]: 
        grid['yq']=ds['yq'] 
    if ds['xh'].values[0]!=grid['xh'].values[0]: 
        grid['xh']=ds['xh']
        
    #######################
    ### Use John Krasting's generate_basin_codes script to create ocean masks:
    
    if basin_list:
        basincodes = generate_basin_codes(grid, lon="geolon", lat="geolat", mask="wet")
        basincodes_v = generate_basin_codes(grid, lon="geolon_v", lat="geolat_v", mask="wet_v")
        
        
    #######################
    ### Cut down the region to according to lat_range, lon_range and depth_range

    if len(lat_range)==2:
        ds=ds.sel(yq=slice(lat_range[0],lat_range[1]))
        grid=grid.sel(yq=slice(lat_range[0],lat_range[1]) , yh=slice(lat_range[0],lat_range[1]))
        grid=grid.sel(yq=slice(lat_range[0],lat_range[1]) , yh=slice(lat_range[0],lat_range[1]))
        if basin_list: 
            basincodes_v=basincodes_v.sel(yq=slice(lat_range[0],lat_range[1]))
            basincodes=basincodes.sel(yh=slice(lat_range[0],lat_range[1]))
    elif len(lat_range)==1:
        ds=ds.sel(yq=lat_range,method='nearest')
        grid=grid.sel(yq=lat_range,method='nearest')
        grid=grid.sel(yh=lat_range,method='nearest')
        if basin_list: 
            basincodes_v=basincodes_v.sel(yq=lat_range,method='nearest')
            basincodes=basincodes.sel(yh=lat_range,method='nearest')
    elif len(lat_range)>2:
        raise Exception("lat_range can have a length of zero (i.e. empty), or 1 or 2 elements")
    
    if len(lon_range)==2:
        ds=ds.sel(xh=slice(lon_range[0],lon_range[1]))
        grid=grid.sel(xh=slice(lon_range[0],lon_range[1]) , xq=slice(lat_range[0],lat_range[1]))
        if basin_list: 
            basincodes_v=basincodes_v.sel(xh=slice(lon_range[0],lon_range[1]))
            basincodes=basincodes.sel(xh=slice(lon_range[0],lon_range[1]))
    elif len(lon_range)==0: 'then do nothing'
    else:
        raise Exception("lon_range must be empty or have 2 elements")
        
    if len(depth_range)==2:
        ds=ds.sel(z_l=slice(depth_range[0],depth_range[1]))
    elif len(depth_range)==0: 'then do nothing'
    else:
        raise Exception("depth_range must either be empty or have 2 elements")
    
    ########################
    ### Use the basincodes to select only the ocean basins listed in basin_list:
    
    if len(basin_list)>0:
        conditions_ds = [basincodes_v==x for x in basin_list]
        conditions_grid = [basincodes==x for x in basin_list]
        combined_condition_ds=reduce(lambda x,y : x|y, conditions_ds)
        combined_condition_grid=reduce(lambda x,y : x|y, conditions_grid)
        ds=ds.where(combined_condition_ds).drop('lon')
        grid['deptho']=grid['deptho'].where(combined_condition_grid).drop('lon')
    
    ########################
    ### Calculate the AMOC etc for the region of choice
    
    vmo_zonmean=ds[v_transport_var].sum(x_hpoint_1Dvar)
    vmo_zonmean=vmo_zonmean/1030/1e6
    moc=vmo_zonmean.cumsum(z_layer_var) 

    
    
    return ds, grid, moc, vmo_zonmean

def MOC_basin_latrange_offline(
    dir_vars,
    dir_grid,
    file_str_identifier,
    nrho=500,
    rholims=[21,28.1],
    ref_pres=0,
    lat_range=[],
    lon_range=[],
    depth_range=[],
    basin_list=[],
    z_layer_var="z_l",
    z_inter_var="z_i",
    v_transport_var="vmo",
    theta_var="thetao",
    salt_var="so",
    rho_var="",
    time_limits=[],
    x_hpoint_1Dvar="xh",
    x_qpoint_1Dvar="xq",
    y_hpoint_1Dvar="yh",
    y_qpoint_1Dvar="yq",
    time_var="time",
    lons_tpoint="geolon",
    lats_tpoint="geolat",
    lons_cpoint="geolon_c",
    lats_cpoint="geolat_c",
    decode_times_flag=True,
    dmgetout=False,
    zarr_dir='',
):
    """This is an "offline" version of the "online" function above. By "offline", 
    I mean that a density-space AMOC calculation will be done offline, using depth space variables". 
    
    Additional Args (see the MOC_xsec_nodes_online descriptions above):
        nrho (integer): number of density bins in the range set by rholims. 
        rholims (vector): a 2-element vector of the type [lowest_rho,highest_rho], that specifies the range of density-space density levels to calculate. 
        ref_pres (scalar): Reference pressure in dB
        lat_range (list): 1-element, or 2-element vector (i.e. [start_lat,end_lat]). Can be left empty (i.e. global). Provide the latitudes, and not the grid indices. Must be in square brackets
        lon_range (list): 1-element, or 2-element vector (i.e. [start_lon,end_lon]). Can be left empty (i.e. global). Provide the longitudes, and not the grid indices. Must be in square brackets
        depth_range (list): 1-element, or 2-element vector (i.e. [start_depth,end_depth]).  Can be left empty. Provide the depths (or densities!), and not the indices. Must be in square brackets
        basin_list (list): list of basins according to John Krasting's generate_basin_codes, e.g. [2,3,5] is [Atl,Pac,Ind]. Others: 1=SO; 4=Arc; 6=Med; 7=Black Sea; 8=Hud Bay(?): 9=Balt. Sea(?); 10=Red Sea(?). 
        theta_var (string): variable name for reading potential temperature from file. If an empty string is given, it will not read temperature
        salt_var (string): variable name for reading salt from file. If an empty string is given, it will not read salt
        rho_var (string): variable name for reading rho from file -- if an empty string is given, but strings are given for temp and salt, then the script will calculate rho from T and S using a reference of "ref_pres"

    output:
        ds (xarray dataset):  dataset of meridional transport for the specific basin(s) (if any) and specified ranges
        grid (xarray dataset):  grid information 
        MOCsig_offline (xarray dataset): density-space MOC streamfunction calculated for the specific basin(s) (if any) and specified ranges
        rhopoto (xarray dataset): Density field for the chosen area and basin

    """
    
    ########################
    ### create a list of files to load:

    vmo_vars_str=dir_vars+"*"+file_str_identifier+"*."+v_transport_var+".nc"
    files_timestep=glob(f"{vmo_vars_str}")
    if dmgetout: 
        print(f"{'dmget '+vmo_vars_str+' &'}")
    if rho_var:
        rho_vars_str=dir_vars+"*"+file_str_identifier+"*."+rho_var+".nc"
        files_timestep+=glob(f"{rho_vars_str}")
        if dmgetout: 
            print(f"{'dmget '+rho_vars_str+' &'}")
            return [],[],[],[]
    if theta_var:
        theta_vars_str=dir_vars+"*"+file_str_identifier+"*."+theta_var+".nc"
        files_timestep+=glob(f"{theta_vars_str}")
        if dmgetout: 
            print(f"{'dmget '+theta_vars_str+' &'}")
    if salt_var:
        salt_vars_str=dir_vars+"*"+file_str_identifier+"*."+salt_var+".nc"
        files_timestep+=glob(f"{salt_vars_str}")
        if dmgetout: 
            print(f"{'dmget '+salt_vars_str+' &'}")
            return [],[],[],[]
       
    ########################
    ### If a zarr_dir is given, modify the list of files to read to/write from there instead:

    if zarr_dir:
        for count, filename in enumerate(files_timestep):
            files_timestep[count] = zarr_dir+filename+'.zarr'
            if os.path.isdir(zarr_dir+filename+'.zarr')==False:
                ds_filename=xr.open_dataset(filename,decode_times=decode_times_flag, chunks={x_hpoint_1Dvar : 100, x_qpoint_1Dvar : 100, y_hpoint_1Dvar : 100, y_qpoint_1Dvar : 100})
                if ".nc" in file_str_identifier:
                    variables2keep=[u_transport_var,v_transport_var,theta_var,salt_var,rho_var,z_layer_var,z_inter_var]
                    variables2keep = [i for i in variables2keep if i]
                    ds_filename=ds_filename[variables2keep]
                ds_filename.to_zarr(zarr_dir+filename+'.zarr')
    
    ########################
    ### Read the data from the list of files:

    ds = xr.open_mfdataset(files_timestep, decode_times=decode_times_flag)#, chunks={'xh' : 100, 'yh' : 100, 'yq' : 100})
    grid=xr.open_dataset(dir_grid)

    ## If there are not the same number of dimensions of yq and yh, cut off the extra dimension:
    if len(ds[y_qpoint_1Dvar])==len(ds[y_hpoint_1Dvar])+1:
        ds = ds.isel( { y_qpoint_1Dvar: slice(1,len(ds[y_qpoint_1Dvar])) } )
        grid = grid.isel( { y_qpoint_1Dvar: slice(1,len(ds[y_qpoint_1Dvar])) } )
    elif len(ds[y_qpoint_1Dvar])==len(ds[y_hpoint_1Dvar]):
        'this is fine'
    else:
        raise Exception("yq must be 1-half-grid more positive (asymmetric) than yh or be 1 element larger than yh (symmetric)")
            
    ##########################
    ### Cut off the final yq index, since the grid file seems to have a problem there in CM4_hires
    ds = ds.isel( { y_qpoint_1Dvar: slice(0,-1),  y_hpoint_1Dvar: slice(0,-1)} )
    grid = grid.isel( { y_qpoint_1Dvar: slice(0,-1),  y_hpoint_1Dvar: slice(0,-1)} )
    
    
    #######################
    ### Use John Krasting's generate_basin_codes script to create ocean masks:
    
    if basin_list:
        basincodes = generate_basin_codes(grid, lon="geolon", lat="geolat", mask="wet")
        basincodes_v = generate_basin_codes(grid, lon="geolon_v", lat="geolat_v", mask="wet_v")

    #######################
    ### Cut down the region to according to lat_range, lon_range and depth_range

    if len(lat_range)==2:
        ds=ds.sel(yq=slice(lat_range[0],lat_range[1]))
        grid=grid.sel(yq=slice(lat_range[0],lat_range[1]))
        ds=ds.sel(yh=slice(lat_range[0],lat_range[1]))
        grid=grid.sel(yh=slice(lat_range[0],lat_range[1]))        
        if basin_list: 
            basincodes_v=basincodes_v.sel(yq=slice(lat_range[0],lat_range[1]))
            basincodes=basincodes.sel(yh=slice(lat_range[0],lat_range[1]))
    elif len(lat_range)==1:
        nearest_lat=ds.indexes['yq'].get_loc(lat_range[0],method="nearest")
        ds=ds.isel(yq=nearest_lat)
        grid=grid.isel(yq=nearest_lat)
        ds=ds.isel(yh=slice(nearest_lat,nearest_lat+2))
        grid=grid.isel(yh=slice(nearest_lat,nearest_lat+2))
        if basin_list: 
            basincodes_v=basincodes_v.sel(yq=lat_range,method='nearest')
            basincodes=basincodes.sel(yh=lat_range,method='nearest')
    elif len(lat_range)>2:
        raise Exception("lat_range can have a length of zero (i.e. empty), or 1 or 2 elements")
    
    if len(lon_range)==2:
        ds=ds.sel(xh=slice(lon_range[0],lon_range[1]))
        grid=grid.sel(xh=slice(lon_range[0],lon_range[1]))
        if basin_list: basincodes_v=basincodes_v.sel(xh=slice(lon_range[0],lon_range[1]))
    elif len(lon_range)==0: 'then do nothing'
    else:
        raise Exception("lon_range must be empty or have 2 elements")
        
    if len(depth_range)==2:
        ds=ds.sel(z_l=slice(depth_range[0],depth_range[1]))
    elif len(depth_range)==0: 'then do nothing'
    else:
        raise Exception("depth_range must be empty or have 2 elements")
    
    #######################
    ### Read T,S,Rho from file, or calculate density from T and S if rho is not given: 
    
    vmo=ds[v_transport_var]
    so=[]
    thetao=[]
    if salt_var:
        ds[salt_var]=ds[salt_var].interp(yh=ds[y_qpoint_1Dvar]).drop_vars(y_hpoint_1Dvar)
    if theta_var:
        ds[theta_var]=ds[theta_var].interp(yh=ds[y_qpoint_1Dvar]).drop_vars(y_hpoint_1Dvar)
    if rho_var:
        rhopoto=ds[rho_var]-1000
        rhopoto=rhopoto.interp(yh=ds[y_qpoint_1Dvar]).drop_vars(y_hpoint_1Dvar)
    else:
        if theta_var and salt_var:   
            rhopoto = momlevel.derived.calc_rho(ds[theta_var],ds[salt_var],ref_pres*1e4)-1000
            rhopoto=rhopoto.rename('rhopoto')
        else:
            error('Either a density variable must be given, or the salt and temperature variables must be given (or all three)')

    ########################
    ### Use the basincodes to select only the ocean basins listed in basin_list:

    if len(basin_list)>0:
        conditions_ds = [basincodes_v==x for x in basin_list]
        conditions_grid = [basincodes==x for x in basin_list]
        combined_condition_ds=reduce(lambda x,y : x|y, conditions_ds)
        combined_condition_grid=reduce(lambda x,y : x|y, conditions_grid)    
        ds=ds.where(combined_condition_ds)
        vmo=vmo.where(combined_condition_ds)
        rhopoto=rhopoto.where(combined_condition_ds).drop('lon')
        grid['deptho']=grid.deptho.where(combined_condition_grid).drop('lon')


    ########################
    ### calculate a SigmaZ diagram and extract the density-space AMOC from it
    
    dens_bins=np.linspace(rholims[0],rholims[1],nrho)
    hist_transpweight=histogram(rhopoto,bins=dens_bins,weights=vmo,dim=['xh'])/1030/1e6
    
    MOCsig_offline=hist_transpweight.mean(dim='time').sum(z_layer_var).cumsum('rhopoto_bin')
    
    return ds, grid, MOCsig_offline, rhopoto

def MOC_basin_computeMFTMHT(ds,grid,rho,rholims,nrho,rebin_depth=[],rho_dim='rhopoto_bin',z_dim='z_l',dist_var='xh',annual_mean_flag=False,plot_flag=True):
    """ Calculates Heat and Freshwater transports normal to the cross-section, using input cross-sections of transport and density. SigmaZ arrays of each of the properties are created so that heat and freshwater transports can be output in both depth- and density- coordinates. "Gyre" and "AMOC" components (in both depth- and density- space) of MHT and MFT are also calculated. 
    
    Args:
        ds (ND xarray): cross-sectional dataset produced by the MOC_basin_latrange_offline function above.  
        grid (NC xarray): grid information matching the transp 
        rho (ND xarray): z-space cross-section of potential density. 
        nrho (integer): number of density bins in the range set by rholims. 
        rholims (vector): a 2-element vector of the type [lowest_rho,highest_rho], that specifies the range of density-space density levels to calculate.         
        rebin_depth (1D array): Rebin the depth vector onto this vector. Helpful to convert an original depth vector with uneven grid spacing to a regular-spaced vertical grid. If left empty then it won't rebin. 
        rho_dim (string): name of the density coordinate. 
        z_dim (string): name of the depth coordinate. 
        dist_var (string): name of the horizontal coordinate. 
        annual_mean_flag (logical): If True then output will be converted from monthly to annual means. Must be set to False if not using monthly data
        plot_flag (logical): makes a simple plot of the output timeseries of MHT, MFT, and their overturning and gyre components. 
        
    output: 
         MOCsig_MHTMFT (xarray Dataset): dataset containing all density-space variables. Descriptions are provided in the output metadata        
         MOCz_MHTMFT (xarray Dataset): dataset containing all depth-space variables. Descriptions are provided in the output metadata
         MOCsigz_MHTMFT (xarray Dataset): dataset containing the SigmaZ diagrams of transport-weighted temperature and salinity. Descriptions in the metadata
        
    """
    
    Cp=3850 # heat capacity of seawater      

    #########################
    ### Specify cell-centre and cell-edges of a density-based vertical coordinate: 
    
    rho0_ref=rholims[0]+np.arange(0,nrho-1)*((rholims[1]-rholims[0])/nrho); 
    rho0_bounds=rholims[0]-(rho0_ref[1]-rho0_ref[0])/2+np.arange(0,nrho)*((rholims[1]-rholims[0])/nrho); 
    
    ########################
    ### calculate all tracer SigmaZ matrices:
    
    ty_z_rho = histogram(rho,bins=[rho0_bounds],weights=ds.vmo.fillna(0.),dim=[dist_var]).squeeze()
    thetao_z_rho = histogram(rho,bins=[rho0_bounds],weights=(ds.thetao*grid.dxCv).fillna(0.),dim=[dist_var]).squeeze()
    so_z_rho = histogram(rho,bins=[rho0_bounds],weights=(ds.so*grid.dxCv).fillna(0.),dim=[dist_var]).squeeze()
    cellarea_z_rho = histogram(rho,bins=[rho0_bounds],weights=(grid.dxCv).fillna(0.),dim=[dist_var]).squeeze()
    thetao_z_rho_mean = thetao_z_rho/cellarea_z_rho  # The mean temperature in each sigma_z cell for OSNAP 
    so_z_rho_mean = so_z_rho/cellarea_z_rho   # The mean salinity in each sigma_z cell for OSNAP 
    
    ######################
    ### if rebin_depth is specified, remap the depth coordinate from the native coordinate to the specified one.  
    if len(rebin_depth)>0:
        model_depth=ds[z_dim].isel({ z_dim : slice(1,len(ds[z_dim]))})
        depth_diff=ds[z_dim].diff(z_dim)
        thetao_z_rho=rebin_sigma_z(model_depth.values,depth_diff.values,rebin_depth,thetao_z_rho.values)
        thetao_z_rho=xr.DataArray(data=thetao_z_rho, dims=('time',z_dim,'rhopoto_bin'),coords={'time' : ty_z_rho['time'],z_dim : rebin_depth, 'rhopoto_bin' : ty_z_rho['rhopoto_bin']})
        so_z_rho=rebin_sigma_z(model_depth.values,depth_diff.values,rebin_depth,so_z_rho.values)
        so_z_rho=xr.DataArray(data=so_z_rho, dims=('time',z_dim,'rhopoto_bin'),coords={'time' : ty_z_rho['time'],z_dim : rebin_depth, 'rhopoto_bin' : ty_z_rho['rhopoto_bin']})
        cellarea_z_rho=rebin_sigma_z(model_depth.values,depth_diff.values,rebin_depth,cellarea_z_rho.values)
        cellarea_z_rho=xr.DataArray(data=cellarea_z_rho, dims=('time',z_dim,'rhopoto_bin'),coords={'time' : ty_z_rho['time'],z_dim : rebin_depth, 'rhopoto_bin' : ty_z_rho['rhopoto_bin']})
        ty_z_rho=rebin_sigma_z(model_depth.values,depth_diff.values,rebin_depth,ty_z_rho.values)
        ty_z_rho=xr.DataArray(data=ty_z_rho, dims=('time',z_dim,'rhopoto_bin'),coords={'time' : thetao_z_rho['time'],z_dim : rebin_depth, 'rhopoto_bin' : thetao_z_rho['rhopoto_bin']})

    #### Calculate z- and rho- space zonal means of T,S and V, for use in calculating overturning component of MFT and MHT. 
    S_bar=so_z_rho.fillna(0).sum()/cellarea_z_rho.fillna(0).sum() 
    S_zm_z = so_z_rho.fillna(0).sum(dim=rho_dim)/cellarea_z_rho.fillna(0).sum(dim=rho_dim) 
    S_zm_rho = so_z_rho.fillna(0).sum(dim=z_dim)/cellarea_z_rho.fillna(0).sum(dim=z_dim) # rho-space zonal mean salinity
    theta_zm_z = thetao_z_rho.fillna(0).sum(dim=rho_dim)/cellarea_z_rho.fillna(0).sum(dim=rho_dim) 
    theta_zm_rho = thetao_z_rho.fillna(0).sum(dim=z_dim)/cellarea_z_rho.fillna(0).sum(dim=z_dim) 
    ty_zm_z = ty_z_rho.fillna(0).sum(dim=rho_dim)
    ty_zm_rho = ty_z_rho.fillna(0).sum(dim=z_dim)

    ######################
    ### calculate MOC MFT:
    
    MFT_MOCrho =- ty_zm_rho*(S_zm_rho-S_bar)/S_bar
    MFT_MOCz =- ty_zm_z*(S_zm_z-S_bar)/S_bar
    MFT_MOCrho_sum = MFT_MOCrho.fillna(0).sum(dim=rho_dim)
    MFT_MOCz_sum = MFT_MOCz.fillna(0).sum(dim=z_dim)
    
    
    ######################
    ### calculate MOC MHT
    
    MHT_MOCrho = Cp*ty_zm_rho*theta_zm_rho
    MHT_MOCz = Cp*ty_zm_z*theta_zm_z
    MHT_MOCrho_sum = MHT_MOCrho.fillna(0).sum(dim=rho_dim)
    MHT_MOCz_sum = MHT_MOCz.fillna(0).sum(dim=z_dim)

    #######################
    ### calculate the transport*tracer SigmaZ matrices, and calculate (zonal- and total-) integrated MFT :
    
    Vthetao_z_rho = histogram(rho,bins=[rho0_bounds],weights=(ds.thetao*ds.vmo).fillna(0.),dim=[dist_var]).squeeze()
    Vso_z_rho = -histogram(rho,bins=[rho0_bounds],weights=((ds.so-S_bar)*ds.vmo/S_bar).fillna(0.),dim=[dist_var]).squeeze()
    
    ######################
    ### if rebin_depth is specified, remap the depth coordinate from the native coordinate to the specified one.  
    if len(rebin_depth)>0:
        Vthetao_z_rho=rebin_sigma_z(model_depth.values,depth_diff.values,rebin_depth,Vthetao_z_rho.values)
        Vthetao_z_rho=xr.DataArray(data=Vthetao_z_rho, dims=('time',z_dim,'rhopoto_bin'),coords={'time' : ty_z_rho['time'],z_dim : rebin_depth, 'rhopoto_bin' : ty_z_rho['rhopoto_bin']})
        Vso_z_rho=rebin_sigma_z(model_depth.values,depth_diff.values,rebin_depth,Vso_z_rho.values)
        Vso_z_rho=xr.DataArray(data=Vso_z_rho, dims=('time',z_dim,'rhopoto_bin'),coords={'time' : ty_z_rho['time'],z_dim : rebin_depth, 'rhopoto_bin' : ty_z_rho['rhopoto_bin']})

    ######################
    ### calculate the GYRE component of MFT and MHT:

    MFT_zonmean_rho=Vso_z_rho.fillna(0).sum(dim=z_dim)
    MFT_zonmean_z=Vso_z_rho.fillna(0).sum(dim=rho_dim)
    MFT_GYRErho=MFT_zonmean_rho-MFT_MOCrho
    MFT_GYREz=MFT_zonmean_z-MFT_MOCz
    MFT_GYRErho_sum=MFT_GYRErho.fillna(0).sum(dim=rho_dim)
    MFT_GYREz_sum=MFT_GYREz.fillna(0).sum(dim=z_dim)
    MFT_sum=Vso_z_rho.fillna(0.).sum(dim=[z_dim,rho_dim])
    MHT_zonmean_rho=Cp*Vthetao_z_rho.fillna(0).sum(dim=z_dim)
    MHT_zonmean_z=Cp*Vthetao_z_rho.fillna(0).sum(dim=rho_dim)
    MHT_GYRErho=MHT_zonmean_rho-MHT_MOCrho
    MHT_GYREz=MHT_zonmean_z-MHT_MOCz
    MHT_GYRErho_sum=MHT_GYRErho.fillna(0).sum(dim=rho_dim)
    MHT_GYREz_sum=MHT_GYREz.fillna(0).sum(dim=z_dim)
    MHT_sum=Cp*Vthetao_z_rho.fillna(0.).sum(dim=[z_dim,rho_dim])

    #######################
    ### Calculate depth- and density- space overturning:
    
    MOCrho =ty_z_rho.sum(dim=z_dim).cumsum(dim=rho_dim).max(dim=rho_dim)
    MOCz = ty_z_rho.sum(dim=rho_dim).cumsum(dim=z_dim).max(dim=z_dim)
    
    ###################
    ### Combine output into single datasets for sigma-, z-, and SigmaZ space:
    
    MOCsig_MHTMFT = xr.Dataset()
    MOCsig_MHTMFT['MOCrho']=MOCrho
    MOCsig_MHTMFT['MHT_zonmean_rho']=MHT_zonmean_rho
    MOCsig_MHTMFT['MHT_sum']=MHT_sum
    MOCsig_MHTMFT['MHT_MOCrho']=MHT_MOCrho
    MOCsig_MHTMFT['MHT_MOCrho_sum']=MHT_MOCrho_sum
    MOCsig_MHTMFT['MHT_GYRErho']=MHT_GYRErho
    MOCsig_MHTMFT['MHT_GYRErho_sum']=MHT_GYRErho_sum
    MOCsig_MHTMFT['MFT_zonmean_rho']=MFT_zonmean_rho
    MOCsig_MHTMFT['MFT_sum']=MFT_sum
    MOCsig_MHTMFT['MFT_MOCrho']=MFT_MOCrho
    MOCsig_MHTMFT['MFT_MOCrho_sum']=MFT_MOCrho_sum
    MOCsig_MHTMFT['MFT_GYRErho']=MFT_GYRErho
    MOCsig_MHTMFT['MFT_GYRErho_sum']=MFT_GYRErho_sum

    MOCz_MHTMFT = xr.Dataset()
    MOCz_MHTMFT['MOCz']=MOCz
    MOCz_MHTMFT['MHT_zonmean_z']=MHT_zonmean_z
    MOCz_MHTMFT['MHT_sum']=MHT_sum
    MOCz_MHTMFT['MHT_MOCz']=MHT_MOCz
    MOCz_MHTMFT['MHT_MOCz_sum']=MHT_MOCz_sum
    MOCz_MHTMFT['MHT_GYREz']=MHT_GYREz
    MOCz_MHTMFT['MHT_GYREz_sum']=MHT_GYREz_sum
    MOCz_MHTMFT['MFT_zonmean_z']=MFT_zonmean_z
    MOCz_MHTMFT['MFT_sum']=MFT_sum
    MOCz_MHTMFT['MFT_MOCz']=MFT_MOCz
    MOCz_MHTMFT['MFT_MOCz_sum']=MFT_MOCz_sum
    MOCz_MHTMFT['MFT_GYREz']=MFT_GYREz
    MOCz_MHTMFT['MFT_GYREz_sum']=MFT_GYREz_sum

    MOCsigz_MHTMFT = xr.Dataset()
    MOCsigz_MHTMFT['Vthetao_z_rho']=Vthetao_z_rho
    MOCsigz_MHTMFT['Vso_z_rho']=Vso_z_rho 
    
    ######################
    ## temporally rebin the monthly offline timeseries into annual ones:
    if annual_mean_flag:
        MOCsig_MHTMFT=MOCsig_MHTMFT.coarsen(time=12).mean()
        MOCz_MHTMFT=MOCz_MHTMFT.coarsen(time=12).mean()
        MOCsigz_MHTMFT=MOCsigz_MHTMFT.coarsen(time=12).mean()
    
    ######################
    ### Plot some data: 
    
    if plot_flag:
        rho0=1030
        fig = plt.figure(figsize=(10,12))
        ax = fig.add_subplot(3,1,1)
        ax.plot(MOCsig_MHTMFT.MOCrho.time,MOCsig_MHTMFT.MOCrho/rho0/1e6)
        ax.set_title('MOC in rho-space')
        ax.set_ylabel('Sv')
        ax = fig.add_subplot(3,1,2)
        ax.plot(MOCsig_MHTMFT.MOCrho.time,(MOCsig_MHTMFT.MHT_sum)/1e15)
        ax.plot(MOCsig_MHTMFT.MOCrho.time,(MOCsig_MHTMFT.MHT_MOCrho_sum)/1e15)
        ax.plot(MOCsig_MHTMFT.MOCrho.time,(MOCsig_MHTMFT.MHT_GYRErho_sum)/1e15)
        ax.set_title('MHT in rho-space')
        ax.set_ylabel('PW')
        ax = fig.add_subplot(3,1,3)
        ax.plot(MOCsig_MHTMFT.MOCrho.time,(MOCsig_MHTMFT.MFT_sum)/rho0/1e6,label='TOTrho')
        ax.plot(MOCsig_MHTMFT.MOCrho.time,(MOCsig_MHTMFT.MFT_MOCrho_sum)/rho0/1e6,label='MOCrho')
        ax.plot(MOCsig_MHTMFT.MOCrho.time,(MOCsig_MHTMFT.MFT_GYRErho_sum)/rho0/1e6,label='GYRErho')
        ax.legend(loc="lower left")
        ax.set_title('MFT in rho-space')
        ax.set_ylabel('Sv')
        ax.set_xlabel('time (months)')
            
            
            
    return MOCsig_MHTMFT, MOCz_MHTMFT, MOCsigz_MHTMFT

@njit
def rebin_sigma_z(model_depth,depth_diff,rebin_depth,ty_z_rho):
    ty_z_rho_rebin=np.zeros((np.shape(ty_z_rho)[0],np.shape(rebin_depth)[0],np.shape(ty_z_rho)[2]))
    rebin_depth_index=0;
    for ii in range(0,np.shape(ty_z_rho)[1]-1):
        V=ty_z_rho[:,ii,:]
        depth_range=depth_diff[ii]
        if model_depth[ii]<rebin_depth[rebin_depth_index]:
            ty_z_rho_rebin[:,rebin_depth_index,:]=ty_z_rho_rebin[:,rebin_depth_index,:]+V
        elif model_depth[ii]==rebin_depth[rebin_depth_index]:
            ty_z_rho_rebin[:,rebin_depth_index,:]=ty_z_rho_rebin[:,rebin_depth_index,:]+V
            rebin_depth_index=rebin_depth_index+1
        elif model_depth[ii]>rebin_depth[rebin_depth_index]:
            top_frac=V*(rebin_depth[rebin_depth_index]-model_depth[ii-1])/depth_range
            ty_z_rho_rebin[:,rebin_depth_index,:]=ty_z_rho_rebin[:,rebin_depth_index,:]+top_frac
            if model_depth[ii]<rebin_depth[rebin_depth_index+1]:
                rebin_depth_index=rebin_depth_index+1
                ty_z_rho_rebin[:,rebin_depth_index,:]=ty_z_rho_rebin[:,rebin_depth_index,:]+(V-top_frac)
            else:
                jjj=0
                while model_depth[ii]>rebin_depth[rebin_depth_index+1]:
                    rebin_depth_index=rebin_depth_index+1
                    middle_frac=V*(rebin_depth[rebin_depth_index]-rebin_depth[rebin_depth_index-1])/depth_range
                    ty_z_rho_rebin[:,rebin_depth_index,:]=ty_z_rho_rebin[:,rebin_depth_index,:]+middle_frac
                    jjj=jjj+1
                rebin_depth_index=rebin_depth_index+1
                ty_z_rho_rebin[:,rebin_depth_index,:]=ty_z_rho_rebin[:,rebin_depth_index,:]+(V-top_frac-(jjj*middle_frac))
    return ty_z_rho_rebin

