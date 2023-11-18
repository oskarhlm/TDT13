import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import csv
import os
from transformers import XmodForSequenceClassification, AutoModelForSequenceClassification
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import L1Loss, MSELoss


class TensorBoardCheckpoint:
    def __init__(self, log_dir, checkpoint_path, run_name=None, best_only=True):
        self.checkpoint_path = checkpoint_path
        self.best_metric = float('inf')
        self.best_only = best_only
        self.date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_name =  run_name
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
            self.writer.add_text('New_Best_Checkpoint', self.checkpoint_path, epoch)
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
            model = XmodForSequenceClassification.from_pretrained(config['model_name'], num_labels=2)
            model.set_default_language('de_CH')
        case _: 
            model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], num_labels=2)

    return model

def get_scheduler(optimizer, config): 
    if "scheduler" not in config:
        raise "No scheduler in config"

    match config["scheduler"]:
        case "reduce_lr_on_plateau":  
            return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        case _:
            raise "Invalid scheduler name"
        
def get_lossfn(config):
    match config['lossfn']:
        case 'MAELoss' | 'L1': 
            return L1Loss(reduction='mean')
        case 'MSELoss' | 'L2': 
            return MSELoss(reduction='mean')

        
