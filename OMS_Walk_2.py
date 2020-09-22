import pandana, time, os, pandas as pd, numpy as np
from pandana.loaders import osm
from pandana.loaders import osm
import osmnx as ox
import pandana as pdna
import  random
import pandas as pd
import numpy as np
from descartes import PolygonPatch
from shapely.geometry import Point, LineString
import h5py

# configure search at a max distance of 1 km for up to the 10 nearest points-of-interest
#amenities = ['hospital', 'clinic', 'pharmacy']
amenities = ['school','bank','cafe','office','restaurant','bus_stop','place_of_worship','kindergarten','atm','community_centre']
distance = 3000
num_pois = 10
num_categories = len(amenities) + 1 #one for each amenity, plus one extra for all of them combined

# bounding box as a list of llcrnrlat, llcrnrlng, urcrnrlat, urcrnrlng
#bbox = [12.9422,77.6289,12.9863,77.6636] #lat-long bounding box for Indiranagar
#bbox = [12.9451,77.7210,12.9899,77.7681] #lat long for Whitefield
bbox = [12.8799,77.4879,13.0808,77.7825] #lat-long bounding box for all bangalore
#bbox = [12.9154,77.6596,12.9300,77.6918] #lat long for Bellandur
#bbox = [12.9330,77.6849,12.9652,77.7096] #lat long for Marathalli
#bbox = [12.9710,77.5480,13.0161,77.5903] #lat long for Malleshwaram
#lat_min=bbox[0],lng_min= bbox[1],lat_max= bbox[2], lng_max=bbox[3])
# configure filenames to save/load POI and network datasets
bbox_string = '_'.join([str(x) for x in bbox])
net_filename = 'network_{}.h5'.format(bbox_string)
poi_filename = 'pois_{}_{}.csv'.format('_'.join(amenities), bbox_string)

# keyword arguments to pass for the matplotlib figure
bbox_aspect_ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
fig_kwargs = {'facecolor':'w',
              'figsize':(10, 10 * bbox_aspect_ratio)}

# keyword arguments to pass for scatter plots
plot_kwargs = {'s':5,
               'alpha':0.9,
               'cmap':'viridis_r',
               'edgecolor':'none'}

# network aggregation plots are the same as regular scatter plots, but without a reversed colormap
agg_plot_kwargs = plot_kwargs.copy()
agg_plot_kwargs['cmap'] = 'viridis'

# keyword arguments to pass for hex bin plots
hex_plot_kwargs = {'gridsize':60,
                   'alpha':0.9,
                   'cmap':'viridis_r',
                   'edgecolor':'none'}

# keyword arguments to pass to make the colorbar
cbar_kwargs = {}

# keyword arguments to pass to basemap
bmap_kwargs = {}

# color to make the background of the axis
bgcolor = 'k'

start_time = time.time()
if os.path.isfile(poi_filename):
    # if a points-of-interest file already exists, just load the dataset from that
    pois = pd.read_csv(poi_filename)
    method = 'loaded from CSV'
else:
    # otherwise, query the OSM API for the specified amenities within the bounding box
    osm_tags = '"amenity"~"{}"'.format('|'.join(amenities))
    pois = osm.node_query(bbox[0], bbox[1], bbox[2], bbox[3], tags=osm_tags)
    pois.to_csv('Maps/Excel/Medical-POIS-Whole_Bangalore.csv') #Printing the POIS selected in the given Map out to a CSV
    # using the '"amenity"~"school"' returns preschools etc, so drop any that aren't just 'school' then save to CSV
    pois = pois[pois['amenity'].isin(amenities)]
    #pois.to_csv(poi_filename, index=False, encoding='utf-8')
    method = 'downloaded from OSM'

print('{:,} POIs {} in {:,.2f} seconds'.format(len(pois), method, time.time() - start_time))
print(pois[['amenity', 'name', 'lat', 'lon']].head())

print(pois['amenity'].value_counts())

start_time = time.time()
# west, south, east, north = (77.4879, 12.8799, 77.7825, 13.0808)
# G = ox.graph_from_bbox(north, south, east, west, network_type='walk')
# ox.plot_graph(G)
# G_df = nodes_df = pd.DataFrame(ox.graph_to_gdfs(G, edges=False))
# print(G_df)
# G_df.to_csv("Maps/Excel/Nodes-Whole_Bangalore.csv") #Printing the nodes of the given map out to a CSV

if os.path.isfile(net_filename):
    # if a street network file already exists, just load the dataset from that
    network = pandana.network.Network.from_hdf5(net_filename)
    print(network.nodes_df)
    method = 'loaded from HDF5'
    store = pd.HDFStore(net_filename, "r")
    nodes = store.nodes
    nodes.to_csv("Maps/Excel/Nodes-Medical-Whole_Bangalore.csv")
    #G_df.to_csv("Maps/Excel/Nodes-Whole_Bangalore.csv")  # Printing the nodes of the given map out to a CSV
else:
    # otherwise, query the OSM API for the street network within the specified bounding box
    network = osm.pdna_network_from_bbox(lat_min=bbox[0],lng_min= bbox[1],lat_max= bbox[2], lng_max=bbox[3])
    print("######################TESTING############################")
    method = 'downloaded from OSM'
    #print(network)
    # identify nodes that are connected to fewer than some threshold of other nodes within a given distance
    lcn = network.low_connectivity_nodes(impedance=1000, count=10, imp_name='distance')
    network.save_hdf5(net_filename, rm_nodes=lcn)  # remove low-connectivity nodes and save to h5

print(method)
print('Network with {:,} nodes {} in {:,.2f} secs'.format(len(network.node_ids), method, time.time() - start_time))

# so, as long as you use a smaller distance, cached results will be used
network.precompute(distance + 1)
# initialize the underlying C++ points-of-interest engine
#network.init_pois(num_categories=num_categories, max_dist=distance, max_pois=num_pois)

# initialize a category for all amenities with the locations specified by the lon and lat columns
network.set_pois(category='all',maxdist = distance, maxitems=num_pois, x_col=pois['lon'], y_col=pois['lat'])

# searches for the n nearest amenities (of all types) to each node in the network
all_access = network.nearest_pois(distance=distance, category='all', num_pois=num_pois)

# it returned a df with the number of columns equal to the number of POIs that are requested
# each cell represents the network distance from the node to each of the n POIs
print('{:,} nodes'.format(len(all_access)))
all_access.to_csv("Maps/Excel/Daily-Whole_Bangalore.csv")  #Printing the distance nodes to it's nearest POIS of the given map out to a CSV
print(all_access)

final = pd.merge(nodes, all_access, how='inner', on=None, left_on='id', right_on='id',
left_index=True, sort=True)
final.to_csv("Maps/Excel/NodeID-Lat-Long-Distance-All_Bangalore.csv")

# distance to the nearest amenity of any type
n = 5
bmap, fig, ax = network.plot(all_access[n], bbox=bbox,plot_type='scatter',  plot_kwargs=plot_kwargs,
                             fig_kwargs=fig_kwargs, bmap_kwargs=bmap_kwargs, cbar_kwargs=cbar_kwargs)
ax.set_facecolor('k')
ax.set_title('Walking distance (m) to nearest amenity', fontsize=15)
fig.savefig('Maps/Whole_Bangalore-scatter-Daily.png', dpi=400, bbox_inches='tight')

bmap, fig1, ax1 = network.plot(all_access[n], bbox=bbox,plot_type='hexbin',  plot_kwargs=hex_plot_kwargs,
                             fig_kwargs=fig_kwargs, bmap_kwargs=bmap_kwargs, cbar_kwargs=cbar_kwargs)
ax1.set_facecolor('k')
ax1.set_title('Walking distance (m) to nearest amenity', fontsize=15)
fig1.savefig('Maps/Whole_Bangalore-hex-Daily.png', dpi=400, bbox_inches='tight')