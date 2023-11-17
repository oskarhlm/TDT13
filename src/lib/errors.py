class EastingOutOfRangeError(Exception):
    def __init__(self, message="Easting value out of valid UTM range"):
        super().__init__(message)

def check_utm_easting_range(array):
    min_easting = 100000  # Minimum valid easting value in meters
    max_easting = 999999  # Maximum valid easting value in meters

    if not ((min_easting <= array[:, 0]).all() and (array[:, 0] <= max_easting).all()):
        raise EastingOutOfRangeError("Easting values are out of the valid UTM range (100,000 m to 999,999 m)")