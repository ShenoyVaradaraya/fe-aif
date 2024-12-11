import cv2
import numpy as np
import pandas as pd
import pyproj


class GeoPixelTransformer:
    def __init__(self, img_shape, latitudes, longitudes):
        self.img_height, self.img_width, _ = img_shape

        # Define WGS84 (EPSG:4326), Web Mercator (EPSG:3857), and US Survey Feet (EPSG:26943) CRS
        self.geo_crs = pyproj.CRS("EPSG:4326")  # WGS84 for lat/lon
        self.mercator_crs = pyproj.CRS("EPSG:3857")  # Web Mercator
        self.survey_feet_crs = pyproj.CRS("EPSG:26915")  # NAD83 / StatePlane California III FIPS 0403 (US Survey Feet)
        
        # Initialize transformers
        self.transformer = pyproj.Transformer.from_crs(self.geo_crs, self.mercator_crs, always_xy=True)
        self.survey_feet_transformer = pyproj.Transformer.from_crs(self.geo_crs, self.survey_feet_crs, always_xy=True)
        self.mercator_transformer = pyproj.Transformer.from_crs(self.mercator_crs, self.geo_crs, always_xy=True)
        self.survey_feet_to_mercator_transformer = pyproj.Transformer.from_crs(self.survey_feet_crs, self.mercator_crs, always_xy=True)

        # Convert latitude and longitude to Web Mercator (EPSG:3857) coordinates
        mercator_coords = [self.transformer.transform(lon, lat) for lat, lon in zip(latitudes, longitudes)]
        self.min_merc_x, self.max_merc_x = min(coord[0] for coord in mercator_coords), max(coord[0] for coord in mercator_coords)
        self.min_merc_y, self.max_merc_y = min(coord[1] for coord in mercator_coords), max(coord[1] for coord in mercator_coords)

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
        longitude, latitude = self.transformer.transform(x_merc, y_merc, direction=pyproj.enums.TransformDirection.INVERSE)
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

    # Convert Lat/Lon to US Survey Feet
    def latlon_to_us_survey_feet(self, latitude, longitude):
        try:
            x_survey_feet, y_survey_feet = self.survey_feet_transformer.transform(longitude, latitude)
            return x_survey_feet, y_survey_feet
        except Exception as e:
            print(f"Error converting lat/lon to US Survey Feet: {e}")
            return None, None

    # Convert US Survey Feet to Pixels
    def us_survey_feet_to_pixel(self, x_survey_feet, y_survey_feet):
        try:
            # Convert US Survey Feet to Mercator
            x_merc, y_merc = self.survey_feet_to_mercator_transformer.transform(x_survey_feet, y_survey_feet)

            # Convert Mercator coordinates to pixel coordinates
            x_pixel, y_pixel = self.mercator_to_pixel(x_merc, y_merc)
            return x_pixel, y_pixel
        except Exception as e:
            print(f"Error converting US Survey Feet to Pixels: {e}")
            return None, None


# Visualization Function: Overlay Points on Image
def overlay_points_on_image(transformer, img, pixel_points, latlon_points, us_survey_feet_points=None, color_pixel=(0, 255, 0), color_latlon=(0, 0, 255), color_survey_feet=(255, 0, 0)):
    overlay_image = img.copy()

    # Plot pixel points
    for x_pixel, y_pixel in pixel_points:
        print(x_pixel, y_pixel)
        cv2.circle(overlay_image, (x_pixel, y_pixel), radius=5, color=color_pixel, thickness=-1)  # Green for pixel points

    # Convert lat/lon to pixel and plot
    for lat, lon in latlon_points:
        x_pixel, y_pixel = transformer.latlon_to_pixel(lat, lon)
        if x_pixel is not None and y_pixel is not None:
            print(f"Lat/Lon: ({lat}, {lon}) -> Pixel: ({x_pixel}, {y_pixel})")
            cv2.circle(overlay_image, (x_pixel, y_pixel), radius=5, color=color_latlon, thickness=-1)  # Red for lat/lon points

    # Convert US Survey Feet to pixel and plot
    if us_survey_feet_points is not None:
        for x_survey_feet, y_survey_feet in us_survey_feet_points:
            x_pixel, y_pixel = transformer.us_survey_feet_to_pixel(x_survey_feet, y_survey_feet)
            print(f"US Survey Feet: ({x_survey_feet}, {y_survey_feet} -> Pixel: ({x_pixel}, {y_pixel})")
            if x_pixel is not None and y_pixel is not None:
                cv2.circle(overlay_image, (x_pixel, y_pixel), radius=5, color=color_survey_feet, thickness=-1)  # Blue for US Survey Feet points

    return overlay_image


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
    data = pd.read_excel(file_path)
    latitudes = data['latitude'].values
    longitudes = data['longitude'].values

    # Initialize the transformer
    transformer = GeoPixelTransformer(img_shape, latitudes, longitudes)

    # Example pixel points
    pixel_points = [(1070, 6320)]

    print(f"Pixel Points: {pixel_points}")
    # Convert pixel points to latitude/longitude for visualization
    latlon_points = [transformer.pixel_to_latlon(x, y) for x, y in pixel_points]

    print(f"Lat/Lon Points: {latlon_points}")

    # Convert lat/lon to US Survey Feet
    us_survey_feet_points = [transformer.latlon_to_us_survey_feet(lat, lon) for lat, lon in latlon_points]

    print(f"US Survey Feet Points: {us_survey_feet_points}")
    # Overlay points on the image
    overlay_image = overlay_points_on_image(transformer, fe_frame, pixel_points, latlon_points, us_survey_feet_points)

    # Display and save the image
    cv2.imshow("Overlay Points", overlay_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('overlay_with_points_and_survey_feet.jpg', overlay_image)
