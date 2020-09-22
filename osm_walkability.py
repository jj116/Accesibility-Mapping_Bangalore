import osmnx as ox
from pandana.loaders import osm
import osmnx as ox
import pandana as pdna
import  random
import pandas as pd
import numpy as np
from descartes import PolygonPatch
from shapely.geometry import Point, LineString

bbox = [12.8881,77.5051,13.0450,77.7750]
# west, south, east, north = (77.6636, 12.9169, 77.6093, 12.9880)
# #The bound area that i want to contruct the heatmap for
# G = ox.graph_from_bbox(north, south, east, west, network_type='walk')
# bbox_aspect_ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
# fig_kwargs = {'facecolor':'w',
#               'figsize':(10, 10 * bbox_aspect_ratio)}
#
# # keyword arguments to pass for scatter plots
# plot_kwargs = {'s':30,
#                'alpha':0.9,
#                'cmap':'viridis_r',
#                'edgecolor':'none'}
#
# # network aggregation plots are the same as regular scatter plots, but without a reversed colormap
# agg_plot_kwargs = plot_kwargs.copy()
# agg_plot_kwargs['cmap'] = 'viridis'
#
# # keyword arguments to pass for hex bin plots
# hex_plot_kwargs = {'gridsize':60,
#                    'alpha':0.9,
#                    'cmap':'viridis_r',
#                    'edgecolor':'none'}
#
# # keyword arguments to pass to make the colorbar
# cbar_kwargs = {}
#
# # keyword arguments to pass to basemap
# bmap_kwargs = {}
#
# # color to make the background of the axis
# bgcolor = 'k'
#
# # define your selected amenities and bounding box
# amenities = ['hospital', 'clinic', 'restaurant','cafe','school','bank','pharmacy','park']
# osm_tags = '"amenity"="restaurant"'
# # request them from the OpenStreetMap API (Overpass)
# pois2 = osm.node_query(lat_min=bbox[0], lng_min=bbox[1], lat_max=bbox[2], lng_max=bbox[3], tags=osm_tags)
# #pois = pois[pois['amenity'].isin(amenities)]
# #print(pois)
# #List how many we downloaded
#
# # print(pois.amenity.value_counts())
# # print(pois[['amenity', 'name', 'lat', 'lon']].head(100))
#
# num_pois = pois2.amenity.value_counts()
#
# intersection_count = 100
#
# pois_df = pd.DataFrame(pois2)
# print(pois_df.head(100))
# # this function makes intersections into Shapely points
# def make_n_pois(): # Changes to give points from pois from the osm.nodequery
#     print(len(pois_df))
#     for i in range(len(pois_df)):
#         #print(pois_df.lon.iloc[i])
#         x = pois_df.lon.iloc[i]
#         y = pois_df.lat.iloc[i]
#         yield Point(x, y)
#
#
#
# pois2 = list(make_n_pois())
# pois_df2 = pois_df[['lat','lon']].copy()
# pois_df2.rename(columns={'lat':'y','lon':'x'},inplace=True)
# #print(pois_df2)
# fig, ax = ox.plot_graph(G, show=True, close=False,
#                         edge_color='#777777')
#
# # Instead, let's first update the network with these new random POI
# for point in pois2:
#     patch = PolygonPatch(point.buffer(0.0004), fc='#ff0000', ec='k', linewidth=0, alpha=0.5, zorder=-1)
#     ax.add_patch(patch)
#
# # Given a graph, generate a dataframe (df)
# # representing all graph nodes
# def create_nodes_df(G):
#     # first make a df from the nodes
#     # and pivot the results so that the
#     # individual node ids are listed as
#     # row indices
#     nodes_df = pd.DataFrame(ox.graph_to_gdfs(G, edges=False))
#     print("Nodes_DF",nodes_df)
#     # preserve these indices as a column values, too
#     nodes_df['id'] = nodes_df.index
#     # and cast it as an integer
#     nodes_df['id'] = nodes_df['id'].astype(int)
#     nodes_df.to_csv("Nodes_DF.csv")
#     return nodes_df
#
#
# nodes_df = create_nodes_df(G)
#
#
# # Given a graph, generate a dataframe (df)
# # representing all graph edges
# def create_edges_df(G):
#     # First, we must move the nested objects
#     # to a signle top level dictionary
#     # that can be consumed by a Pandas df
#     edges_ref = {}
#
#     # move through first key (origin node)
#     for e1 in G.adj.keys():
#         e1_dict = G.adj[e1]
#
#         # and then get second key (destination node)
#         for e2 in e1_dict.keys():
#             # always use the first key here
#             e2_dict = e1_dict[e2][0]
#
#             # update the sub-dict to include
#             # the origin and destination nodes
#             e2_dict['st_node'] = e1
#             e2_dict['en_node'] = e2
#
#             # ugly, and unnecessary but might as
#             # well name the index something useful
#             name = '{}_{}'.format(e1, e2)
#
#             # udpate the top level reference dict
#             # with this new, prepared sub-dict
#             edges_ref[name] = e2_dict
#
#     # let's take the resulting dict and convert it
#     # to a Pandas df, and pivot it as with the nodes
#     # method to get unique edges as rows
#     edges_df = pd.DataFrame(edges_ref).T
#     print(edges_df)
#     # udpate the edge start and stop nodes as integers
#     # which is necessary for Pandana
#     edges_df['st_node'] = edges_df['st_node'].astype('int64')
#     edges_df['en_node'] = edges_df['en_node'].astype('int64')
#
#     # for the purposes of this example, we are not going
#     # to both with impedence along edge so they all get
#     # set to the same value of 1
#     edges_df['weight'] = 1
#     edges_df.to_csv("Edges_DF.csv")
#
#     return edges_df
#
# edges_df = create_edges_df(G)
#
#
# net = pdna.Network(nodes_df['x'], nodes_df['y'], edges_df['st_node'], edges_df['en_node'],
#                    edges_df[['weight']])
#
# near_ids = net.get_node_ids(pois_df2['x'],
#                             pois_df2['y'],
#                             mapping_distance=5)
# pois_df2['nearest_node_id'] = near_ids
#
# nearest_to_pois = pd.merge(pois_df2,
#                            nodes_df,
#                            left_on='nearest_node_id',
#                            right_on='id',
#                            how='left',
#                            sort=False,
#                            suffixes=['_from', '_to'])
# print("##############################MERGED DATAFRAME#######################")
# print(nearest_to_pois)
# nearest_to_pois.to_csv("Merged_dataframe.csv")
#
# nearest_to_pois.dropna(inplace=True)
#
# for row_id, row in nearest_to_pois.iterrows():
#     # Draw a circle on the nearest graph node
#     point = Point(row.x_to, row.y_to)
#     patch = PolygonPatch(point.buffer(0.0001),
#                          fc='#0073ef',
#                          ec='k',
#                          linewidth=0,
#                          alpha=0.5,
#                          zorder=-1)
#     ax.add_patch(patch)
#
#     # Sloppy way to draw a line because I don't want to Google Matplotlib API
#     # stuff anymore right now
#     linestr = LineString([(row['x_from'], row['y_from']),
#                           (row['x_to'], row['y_to'])]).buffer(0.000001)
#     new_line = PolygonPatch(linestr,
#                             alpha=0.4,
#                             fc='#b266ff',
#                             zorder=1)
#     ax.add_patch(new_line)
#     fig.savefig("final.png")
#
#
# # function to plot distance to selected amenity
# # -- default: distance to nearest amenity
# # -- if a parameter n is supplied, distance to the nth nearest amenity
# distance = 50
# num_pois = 10
# net.set_pois(category='restaurant',maxdist = distance, maxitems=num_pois, x_col=pois_df['lon'], y_col=pois_df['lat'])
# def plot_nearest_amenity(amenity, n):
#     accessibility = net.nearest_pois(distance=distance, category=amenity, num_pois=num_pois)
#     print(accessibility[1])
#     accessibility.to_csv("Accessibilty.csv")
#     fig, ax = net.plot(accessibility[1], bbox=bbox,plot_kwargs=plot_kwargs, fig_kwargs=fig_kwargs,
#                              bmap_kwargs=bmap_kwargs, cbar_kwargs=cbar_kwargs)
#     ax.set_facecolor('k')
#     ax.set_title('Pedestrian accessibility in Bangalore (Walking distance to {}, meters (n = {}))'.format(amenity, n),
#                  fontsize=14);
#
#
# plot_nearest_amenity('restaurant', 3)



import pandana, time, os, pandas as pd, numpy as np
from pandana.loaders import osm

# configure search at a max distance of 1 km for up to the 10 nearest points-of-interest
#amenities = ['hospital', 'clinic', 'pharmacy']
amenities = ['school','bank','cafe','office','restaurant','office']
distance = 3000
num_pois = 10
num_categories = len(amenities) + 1 #one for each amenity, plus one extra for all of them combined
west, south, east, north = (77.4879, 12.8799, 77.7825, 13.0808)
G = ox.graph_from_bbox(north, south, east, west, network_type='walk')
ox.plot_graph(G)
G_df = nodes_df = pd.DataFrame(ox.graph_to_gdfs(G, edges=False))
print(G_df)
G_df.to_csv("Maps/Excel/Nodes-Whole_Bangalore.csv") #Printing the nodes of the given map out to a CSV
# bounding box as a list of llcrnrlat, llcrnrlng, urcrnrlat, urcrnrlng
#bbox = [12.9422,77.6289,12.9863,77.6636] #lat-long bounding box for Indiranagar

# bbox = [12.9584,77.6258,12.9828,77.6505]#lat-long bounding box for Indiranagar
#bbox = [12.9154,77.6596,12.9300,77.6918] #lat long for Bellandur
#bbox = [12.9330,77.6849,12.9652,77.7096] #lat long for Marathalli
bbox = [12.8799,77.4879,13.0808,77.7825] #lat-long bounding box for all bangalore
#bbox = [12.9710,77.5480,13.0161,77.5903] #lat long for Malleshwaram
#bbox = [12.9451,77.7210,12.9899,77.7681] #Whitefield
# bbox = [12.9584,77.6258,12.9828,77.6505]
# bbox = [12.9584,77.6258,12.9828,77.6505]
#lat_min=bbox[0],lng_min= bbox[1],lat_max= bbox[2], lng_max=bbox[3])
# configure filenames to save/load POI and network datasets
bbox_string = '_'.join([str(x) for x in bbox])
net_filename = 'network_{}.h5'.format(bbox_string)
poi_filename = 'pois_{}_{}.csv'.format('_'.join(amenities), bbox_string)

# keyword arguments to pass for the matplotlib figure
bbox_aspect_ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
fig_kwargs = {'facecolor':'w',
              'figsize':(10, 10 * bbox_aspect_ratio)}

west, south, east, north = (77.6636, 12.9422, 77.6289, 12.9863)
G = ox.graph_from_bbox(north, south, east, west, network_type='walk')
G_df = nodes_df = pd.DataFrame(ox.graph_to_gdfs(G, edges=False))
print(G_df)
G_df.to_csv("Maps/Excel/Nodes-Whole_Bangalore.csv") #Printing the nodes of the given map out to a CSV
# keyword arguments to pass for scatter plots
plot_kwargs = {'s':30,
               'alpha':0.9,
               'cmap':'viridis_r',
               'edgecolor':'none'}

# keyword arguments to pass for hex bin plots
hex_plot_kwargs = {'gridsize':60,
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

    # using the '"amenity"~"school"' returns preschools etc, so drop any that aren't just 'school' then save to CSV
    pois = pois[pois['amenity'].isin(amenities)]
    #pois.to_csv(poi_filename, index=False, encoding='utf-8')
    method = 'downloaded from OSM'

print('{:,} POIs {} in {:,.2f} seconds'.format(len(pois), method, time.time() - start_time))
print(pois[['amenity', 'name', 'lat', 'lon']].head())

print(pois['amenity'].value_counts())

start_time = time.time()



if os.path.isfile(net_filename):
    # if a street network file already exists, just load the dataset from that
    network = pandana.network.Network.from_hdf5(net_filename)
    method = 'loaded from HDF5'
else:
    # otherwise, query the OSM API for the street network within the specified bounding box
    network = osm.pdna_network_from_bbox(lat_min=bbox[0],lng_min= bbox[1],lat_max= bbox[2], lng_max=bbox[3])
    method = 'downloaded from OSM'

    # identify nodes that are connected to fewer than some threshold of other nodes within a given distance
    lcn = network.low_connectivity_nodes(impedance=1000, count=10, imp_name='distance')
    network.save_hdf5(net_filename, rm_nodes=lcn)  # remove low-connectivity nodes and save to h5

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
all_access.to_csv("Maps/Excel/Daily-Whole_Bangalore.csv")
print(all_access)

# distance to the nearest amenity of any type
n = 1
bmap, fig, ax = network.plot(all_access[n], bbox=bbox,plot_type='hexbin',  plot_kwargs=hex_plot_kwargs,
                             fig_kwargs=fig_kwargs, bmap_kwargs=bmap_kwargs, cbar_kwargs=cbar_kwargs)
# ax.set_facecolor('k')
# ax.set_title('Walking distance (m) to nearest amenity around Indiranagar', fontsize=15)
# fig.savefig('Maps/accessibility-all-Bangalore.png', dpi=200, bbox_inches='tight')