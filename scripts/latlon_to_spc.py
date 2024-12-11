from pyproj import CRS, Transformer

# Define the SPCS projections for Minnesota zones
# Minnesota Central (EPSG:26917), Minnesota North (EPSG:26915), Minnesota South (EPSG:26916)
# Example here is for Minnesota Central (Zone 17)
def feet_to_meters(feet):
    return feet * 0.3048

def meters_to_feet(meters):
    return meters / 0.3048

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3594", always_xy=True)
# Example SPCS coordinates (easting, northing) in Minnesota Central zone (EPSG:26917)
y, x = 1015643.4029999994,2814423.254999998 # Replace with actual coordinates
lat,lon = 44.8837613,-93.26788027
# Convert from SPCS to latitude and longitude
longitude,latitude = transformer.transform(x,y)
# Convert from latitude and longitude to SPCS
x1, y1 = transformer.transform(lon,lat)

print(f"Latitude: {latitude}, Longitude: {longitude}")
print(latitude,",",longitude)
print(f"SPCS Coordinates: Easting = {x1}, Northing = {y1}")