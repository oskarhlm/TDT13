import numpy as np
import utm 
import torch 
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pyproj import Proj, Transformer

from .errors import check_utm_easting_range

class GeolocationDataset(Dataset):
    def __init__(self, texts, coordinates, config):
        self.texts = texts
        self.coordinates = coordinates
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, return_tensors='pt')
        inputs = {key: val.squeeze() for key, val in inputs.items()}  # Remove batch dimension
        coords = torch.tensor(self.coordinates[idx], dtype=torch.float)
        return inputs, coords



def lv95_converter(lat, lon, inverse=False):
    wgs84 = 'epsg:4326'  
    lv95 = 'epsg:2056'  

    if inverse: 
        transformer = Transformer.from_crs(lv95, wgs84)
    else:
        transformer = Transformer.from_crs(wgs84, lv95)

    easting, northing = transformer.transform(lat, lon)
    return easting, northing


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
            
            case 'lv95':
                new_col_names = ['easting', 'northing', 'text']

                df[['lat', 'lon']] = df.apply(
                    lambda row: lv95_converter(row['lat'], row['lon']),
                    axis=1, result_type='expand')
                df.columns = new_col_names

    return df, new_col_names if new_col_names else default_col_names


def transform_to_latlon(latlon, config): 
    latlon_trans = None
    
    if 'projection' not in config: 
        return latlon
    
    match config['projection']: 
        case 'utm': 
            check_utm_easting_range(latlon)
            latlon_trans = np.array(utm.to_latlon(
                 latlon[:, 0], latlon[:, 1], zone_number=config['zone_number'], zone_letter=config['zone_letter'])).T
            
            check_utm_easting_range(latlon)
            return latlon_trans
        
        case 'lv95': 
            latlon_trans = np.array(lv95_converter(
                 latlon[:, 0], latlon[:, 1], inverse=True)).T
            
            return latlon_trans

        case _: 
            return latlon

