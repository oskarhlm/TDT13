import numpy as np
from .errors import check_utm_easting_range
import utm
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class GeolocationDataset(Dataset):
    def __init__(self, texts, coordinates, config):
        self.texts = texts
        self.coordinates = coordinates
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx], padding='max_length', truncation=True, return_tensors='pt')
        inputs = {key: val.squeeze() for key, val in inputs.items()
                  }  # Remove batch dimension
        coords = torch.tensor(self.coordinates[idx], dtype=torch.float)
        return inputs, coords


def haversine_distance2(c1, c2):
    """c1 and c2 are arrays containing lat/lon in degrees"""
    R = 6371

    c1 = np.radians(c1)
    c2 = np.radians(c2)
    dlat = c2[:, 0] - c1[:, 0]
    dlon = c2[:, 1] - c1[:, 1]

    a = np.sin(dlat / 2)**2 + np.cos(c1[:, 0]) * \
        np.cos(c2[:, 0]) * np.sin(dlon / 2)**2
    d = 2 * R * np.arcsin(np.sqrt(a))

    return d


def transform_to_latlon(latlon, config):
    latlon_trans = None

    if 'projection' not in config:
        return latlon

    match config['projection']:
        case 'utm':
            check_utm_easting_range(latlon)

            latlon_trans = np.array(utm.to_latlon(
                latlon[:, 0], latlon[:, 1], zone_number=config['zone_number'], zone_letter=config['zone_letter']))

            check_utm_easting_range(latlon)
        case _:
            return latlon

    return latlon_trans


def median_distance(preds, labels, scaler, config):
    if scaler:
        preds = scaler.inverse_transform(preds)
        labels = scaler.inverse_transform(labels)

    if 'projection' in config:
        preds = transform_to_latlon(preds, config)
        labels = transform_to_latlon(labels, config)
    return np.median(haversine_distance(preds, labels))


def mean_distance(preds, labels, scaler, config):
    if scaler:
        preds = scaler.inverse_transform(preds)
        labels = scaler.inverse_transform(labels)

    if 'projection' in config:
        preds = transform_to_latlon(preds, config)
        labels = transform_to_latlon(labels, config)
    return np.mean(haversine_distance(preds, labels))


def to_projection(df, config):
    default_col_names = ['lat', 'lon', 'text']
    new_col_names = None
    if 'projection' in config:
        match config['projection']:
            case 'utm':
                new_col_names = ['easting', 'northing', 'text']

                df[['lat', 'lon']] = df.apply(
                    lambda row: utm.from_latlon(
                        row['lat'], row['lon'],
                        force_zone_number=config['zone_number'],
                        force_zone_letter=config['zone_letter'])[:2],
                    axis=1, result_type='expand')
                df.columns = new_col_names

    return df, new_col_names if new_col_names else default_col_names
