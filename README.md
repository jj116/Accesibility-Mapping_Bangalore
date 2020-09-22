# Methodology
The pedestrian accessibility score determines how easy it is for a citizen to walk for different daily requirements. We break-down pedestrian accessibility into two factors - 
Permeability of the area (the density of the street network)
Availability of amenities catering to diverse requirements (mixed-use availability) and public transit, i.e. proximity to bus stop and metro stations

Hence, we define the “Pedestrian accessibility score” as the distance to the sixth closest amenity, to suitably account for both permeability and mixed-use availability. This score does not however consider other factors that affect pedestrian walkability such as comfort, shading, infrastructure, etc.

## Data
Based on data available from Open data maps, we have been able to calculate a “Pedestrian Accessibility score”. The amenities considered are: 
‘School'
'Bank'
'Cafe'
'Office'
'Restaurant'
'Office'
'Bus_stop'
'Place_of_worship'
'Kindergarten'
'Atm'
'Community_centre'

We were able to extract around 6,200 amenities under the above-mentioned categories. Subsequently, we calculated the Pedestrian accessibility scores for 1.06 lac nodes for the city. The nodes were selected as all major street intersections in Bangalore to account for the difference in the radial distance and the actual distance while walking.

We are to visualize the areas which are walkable and where there is a dearth of amenities using this map. 
