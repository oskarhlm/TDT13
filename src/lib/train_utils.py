import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import csv
import os
from transformers import XmodForSequenceClassification, AutoModelForSequenceClassification
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torch.nn import L1Loss, MSELoss
from .preprocessing import scalers
import pandas as pd
from .geo import to_projection, GeolocationDataset, transform_to_latlon
from tqdm import tqdm
from .metrics import median_distance, mean_distance
from torch.utils.data import DataLoader
import numpy as np


class TensorBoardCheckpoint:
    def __init__(self, log_dir, checkpoint_path, run_name=None, best_only=True):
        self.checkpoint_path = checkpoint_path
        self.best_metric = float('inf')
        self.best_only = best_only
        self.date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_name = run_name
        self.log_dir = f'{log_dir}/{self.run_name}'
        self.writer = SummaryWriter(self.log_dir)
        self.metric_path = f'{self.log_dir}/metrics.csv'

    def log_metrics(self, metrics, step):
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(metric_name, metric_value, step)

        with open(self.metric_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not os.path.isfile(self.metric_path):
                writer.writerow(metrics.keys())

            writer.writerow(metrics.values())

    def save_checkpoint(self, model, optimizer, epoch, metrics, scaler):
        checkpoint_path = f'{self.checkpoint_path}/{self.run_name}_best_model.pth'
        if metrics['Median_Distance/dev'] < self.best_metric:
            self.best_metric = metrics['Median_Distance/dev']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'scaler': scaler
            }, checkpoint_path)
            self.writer.add_text('New_Best_Checkpoint',
                                 self.checkpoint_path, epoch)
            print(f"New best checkpoint saved at {self.checkpoint_path}")
        elif not self.best_only:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    def close(self):
        self.writer.close()


def get_model(config):
    match config['model_type']:
        case 'x-mod':
            model = XmodForSequenceClassification.from_pretrained(
                config['model_name'], num_labels=2)
            model.set_default_language('de_CH')
        case _:
            model = AutoModelForSequenceClassification.from_pretrained(
                config['model_name'], num_labels=2)

    return model


def get_scheduler(optimizer, config, **kwargs):
    if "scheduler" not in config:
        raise "No scheduler in config"

    match config["scheduler"]:
        case "reduce_lr_on_plateau":
            return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        case 'one_cycle_lr':
            return OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=kwargs['steps_per_epoch'], epochs=config['epochs'])
        case _:
            raise "Invalid scheduler name"


def get_lossfn(config):
    match config['lossfn']:
        case 'MAELoss' | 'L1':
            return L1Loss(reduction='mean')
        case 'MSELoss' | 'L2':
            return MSELoss(reduction='mean')


def evaluate_geolocation_model_by_checkpoint(checkpoint_dir, checkpoint_file, vardial_path, config):
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config)
    model.to(device)

    checkpoint_path = f'{checkpoint_dir}/{checkpoint_file}'

    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])

    train_data = pd.read_table(
        f'{vardial_path}/ch/train.txt', header=None, names=['lat', 'lon', 'text'])
    train_data, col_names = to_projection(train_data, config)

    scaler_name = config['scaler']
    checkpoint_scaler = scalers[scaler_name]()
    checkpoint_scaler.fit(train_data[col_names[:2]].values)

    test_gold_data = pd.read_table(
        f'{vardial_path}/ch/test_gold.txt', header=None, names=['lat', 'lon', 'text'])
    test_gold_data, _ = to_projection(test_gold_data, config)
    test_gold_coords = checkpoint_scaler.transform(
        test_gold_data[col_names[:2]].values)
    test_gold_dataset = GeolocationDataset(
        test_gold_data['text'].tolist(), test_gold_coords, config)
    test_gold_loader = DataLoader(
        test_gold_dataset, batch_size=config['train_batch_size'], shuffle=False)

    model.eval()

    with torch.no_grad():
        test_preds = []
        for batch in tqdm(test_gold_loader):
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            test_preds.append(logits.cpu().numpy())

    test_preds = np.concatenate(test_preds, axis=0)

    results = {
        'median_distance': median_distance(test_gold_coords, test_preds, checkpoint_scaler, config),
        'mean_distance': mean_distance(test_gold_coords, test_preds, checkpoint_scaler, config),
    }

    print(f'{checkpoint_file} test results: {results}\n')

    return results, transform_to_latlon(checkpoint_scaler.inverse_transform(test_preds), config)
