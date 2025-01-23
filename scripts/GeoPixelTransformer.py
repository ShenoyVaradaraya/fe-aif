import cv2
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer,enums

def prepare_latlon(file_path):
    data = pd.read_excel(file_path)
    latitudes = data['latitude'].values
    longitudes = data['longitude'].values
    return latitudes, longitudes

class GeoPixelTransformer:
    def __init__(self, img_shape, latitudes, longitudes):
        self.img_height, self.img_width, _ = img_shape

        # Define WGS84 (EPSG:4326), Web Mercator (EPSG:3857), and US Survey Feet (EPSG:26943) CRS
        self.geo_crs = CRS("EPSG:4326")  # WGS84 for lat/lon
        self.mercator_crs = CRS("EPSG:3857")  # Web Mercator

        # Initialize transformers
        self.transformer = Transformer.from_crs(self.geo_crs, self.mercator_crs, always_xy=True)
        # Convert latitude and longitude to Web Mercator (EPSG:3857) coordinates
        mercator_coords = [self.transformer.transform(lon, lat) for lat, lon in zip(latitudes, longitudes)]
        self.min_merc_x, self.max_merc_x = min(coord[0] for coord in mercator_coords), max(coord[0] for coord in mercator_coords)
        self.min_merc_y, self.max_merc_y = min(coord[1] for coord in mercator_coords), max(coord[1] for coord in mercator_coords)

        self.latlon_to_xy_tf = Transformer.from_crs("EPSG:4326", "EPSG:26915", always_xy=True)
        self.xy_to_spc_tf = Transformer.from_crs("EPSG:26915", "ESRI:103389", always_xy=True)
        self.cam_pole_lat, self.cam_pole_lon = 44.88354957, -93.26783915

    def feet_to_meters(self,feet):
        return feet * 1200 / 3937

    def meters_to_feet(self,meters):
        return meters / 1200 * 3937
    
    def pixel_to_mercator(self, x_pixel, y_pixel):
        x_norm = x_pixel / self.img_width
        y_norm = y_pixel / self.img_height

        x_merc = self.min_merc_x + x_norm * (self.max_merc_x - self.min_merc_x)
        y_merc = self.max_merc_y - y_norm * (self.max_merc_y - self.min_merc_y)  # Invert Y-axis

        return x_merc, y_merc

    def mercator_to_pixel(self, x_merc, y_merc):
        x_norm = (x_merc - self.min_merc_x) / (self.max_merc_x - self.min_merc_x)
        y_norm = (self.max_merc_y - y_merc) / (self.max_merc_y - self.min_merc_y)  # Invert Y-axis

        x_pixel = int(x_norm * self.img_width)
        y_pixel = int(y_norm * self.img_height)

        return x_pixel, y_pixel

    def pixel_to_latlon(self, x_pixel, y_pixel):
        x_merc, y_merc = self.pixel_to_mercator(x_pixel, y_pixel)
        # Use the same transformer to convert Mercator back to Lat/Lon (EPSG:4326)
        longitude, latitude = self.transformer.transform(x_merc, y_merc, direction=enums.TransformDirection.INVERSE)
        return latitude, longitude

    def latlon_to_pixel(self, latitude, longitude):
        try:
            x_merc, y_merc = self.transformer.transform(longitude, latitude)

            # Skip invalid coordinates
            if x_merc is None or y_merc is None or np.isinf(x_merc) or np.isinf(y_merc):
                print(f"Invalid Mercator for lat/lon ({latitude}, {longitude})")
                return None, None

            x_pixel, y_pixel = self.mercator_to_pixel(x_merc, y_merc)
            if x_pixel is None or y_pixel is None:
                print(f"Lat/Lon Out of bounds: ({latitude}, {longitude})")
            return x_pixel, y_pixel
        except Exception as e:
            print(f"Error in transformation: ({latitude}, {longitude}): {e}")
            return None, None

    def latlon_to_xy(self,lat, lon):
        x, y = self.latlon_to_xy_tf.transform(lon, lat)
        return x, y

    def xy_to_spc(self,x, y):
        easting, northing = self.xy_to_spc_tf.transform(x, y)
        return self.meters_to_feet(easting), self.meters_to_feet(northing)

    def latlon_to_spc(self,lat, lon):
        x, y = self.latlon_to_xy(lat, lon)
        return self.xy_to_spc(x, y)

    def get_spc_relative_to_cam_pole(self,lat, lon):
        cam_pole_x, cam_pole_y = self.latlon_to_spc(self.cam_pole_lat, self.cam_pole_lon)
        x, y = self.latlon_to_spc(lat, lon)
        return x - cam_pole_x, y - cam_pole_y
    
    def get_relative_spc_from_pixel(self,x_pixel, y_pixel):
        lat, lon = self.pixel_to_latlon(x_pixel, y_pixel)
        return self.get_spc_relative_to_cam_pole(lat, lon)


# Main Execution
if __name__ == "__main__":
    # Load the first frame of the video for image dimensions
    pano_video = cv2.VideoCapture('./panoramic.avi')
    fe_video = cv2.VideoCapture('/fe8k-data/sensornet/Tue_2017-03-14_073002.avi')
    ret, frame = pano_video.read()
    fe_ret, fe_frame = fe_video.read()
    img_shape = frame.shape

    # Load latitude and longitude data
    file_path = 'Portland_66th_GE_Points.xlsx'
    latitudes, longitudes = prepare_latlon(file_path)

    # Initialize the transformer
    transformer = GeoPixelTransformer(img_shape, latitudes, longitudes)

    # Example pixel points
    pixel_points = [(1070, 6320)]


    lat,lon = transformer.pixel_to_latlon(x, y)
    print(f"Latitude: {lat}, Longitude: {lon}")
    spc_e, spc_n = transformer.latlon_to_spc(lat, lon)
    print(f"SPC Coordinates: Easting = {spc_e}, Northing = {spc_n}")
    easting, northing = transformer.get_spc_relative_to_cam_pole(lat, lon)
    print(f"Relative SPC Coordinates: Easting = {easting}, Northing = {northing}")

