import numpy as np

from .geo import transform_to_latlon


def haversine_distance(c1, c2):
    """Calculate the Haversine distance.
    
    c1 and c2 can be single [lat, lon] pairs or arrays of [lat, lon] pairs.
    """

    R = 6371  # Radius of Earth in km

    # Convert coordinates to radians
    c1 = np.atleast_2d(np.radians(c1))
    c2 = np.atleast_2d(np.radians(c2))

    # Differences in coordinates
    dlat = c2[:, 0] - c1[:, 0]
    dlon = c2[:, 1] - c1[:, 1]

    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(c1[:, 0]) * np.cos(c2[:, 0]) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c  # Distance in km

def median_distance(preds, labels, scaler=None, config=None):
    if scaler:
        preds = scaler.inverse_transform(preds)
        labels = scaler.inverse_transform(labels)

    if config and 'projection' in config:
        preds = transform_to_latlon(preds, config)
        labels = transform_to_latlon(labels, config)
    return np.median(haversine_distance(preds, labels))

def mean_distance(preds, labels, scaler=None, config=None):    
    if scaler:
        preds = scaler.inverse_transform(preds)
        labels = scaler.inverse_transform(labels)
    
    if config and 'projection' in config:
        preds = transform_to_latlon(preds, config)
        labels = transform_to_latlon(labels, config)
    return np.mean(haversine_distance(preds, labels))