# -*- coding: utf-8 -*-
print("MSE + P1,P2,P4 Physics Loss Training with P6 KKT Constraints")

# ========== Configuration Parameters ==========
VERSION = "MSE_Physics"  # Version number, includes KKT constraints

RANDOM_SEED = 123  # Fixed random seed
# Training Hyperparameters
EPOCHS = 150  # Maximum number of training epochs
LEARNING_RATE = 3e-4  # Initial learning rate
WEIGHT_DECAY = 1e-4  # L2 regularization coefficient
BATCH_SIZE = 256  # Batch size

# Learning Rate Scheduler Parameters
LR_SCHEDULER_FACTOR = 0.5  # Learning rate decay factor
LR_SCHEDULER_PATIENCE = 5  # Learning rate scheduler patience
MIN_LR = 1e-6  # Minimum learning rate

# Early Stopping Parameters
EARLY_STOP_PATIENCE = 30  # Early stopping patience
MIN_IMPROVEMENT = 3e-4  # Minimum improvement threshold

# Model Saving Parameters
SAVE_EVERY_N_EPOCHS = 10  # Save a checkpoint every N epochs

# Physics Loss Weight
PHYSICS_WEIGHT = 1  # Physics loss weight
# =============================

print("-1.in")
import numpy as np
import os
import sys
import logging
from datetime import datetime
import torch
import random


# Create output directory
output_dir = f"/home/fjl/ML4PSC/output/{VERSION}"
os.makedirs(output_dir, exist_ok=True)

# Create data saving directory
data_dir = os.path.join(output_dir, "data")
os.makedirs(data_dir, exist_ok=True)

model_name = f"{VERSION}.model"
save_img_path1 = f"{output_dir}/convergence_plot.png"
log_file = f"{output_dir}/training.log"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set random seed
def set_random_seed(seed):
    """Set all random seeds to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to: {seed}")

# Initialize random seed
set_random_seed(RANDOM_SEED)

# Log hyperparameter configuration
logger.info("="*50)
logger.info("HYPERPARAMETER CONFIGURATION")
logger.info("="*50)
logger.info(f"VERSION: {VERSION}")
logger.info(f"RANDOM_SEED: {RANDOM_SEED}")
logger.info(f"EPOCHS: {EPOCHS}")
logger.info(f"LEARNING_RATE: {LEARNING_RATE}")
logger.info(f"WEIGHT_DECAY: {WEIGHT_DECAY}")
logger.info(f"BATCH_SIZE: {BATCH_SIZE}")
logger.info(f"LR_SCHEDULER_FACTOR: {LR_SCHEDULER_FACTOR}")
logger.info(f"LR_SCHEDULER_PATIENCE: {LR_SCHEDULER_PATIENCE}")
logger.info(f"MIN_LR: {MIN_LR}")
logger.info(f"EARLY_STOP_PATIENCE: {EARLY_STOP_PATIENCE}")
logger.info(f"MIN_IMPROVEMENT: {MIN_IMPROVEMENT}")
logger.info(f"SAVE_EVERY_N_EPOCHS: {SAVE_EVERY_N_EPOCHS}")
logger.info(f"PHYSICS_WEIGHT: {PHYSICS_WEIGHT}")
logger.info("CONSTRAINTS: P1, P2, P4 (Physics Loss) + P6 (KKT Projection)")
logger.info("="*50)

# Check and switch to the correct working directory
if os.path.exists("ml4physim_startingkit_powergrid"):
    os.chdir("ml4physim_startingkit_powergrid")
    sys.path.append('../ml4physim_startingkit_powergrid')
else:
    logger.warning("ml4physim_startingkit_powergrid directory not found, staying in current directory")
print(sys.path)
import warnings
warnings.filterwarnings('ignore')
import numpy as np
print("-1.1.numpy")
import torch
import pathlib
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import copy
import math
import time
from lips.dataset.scaler.powergrid_scaler import PowerGridScaler
from lips.dataset.scaler.scaler import Scaler
from lips.dataset.dataSet import DataSet
from lips.dataset.powergridDataSet import PowerGridDataSet
from typing import Union
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
from matplotlib import pyplot as plt
from lips.benchmark.powergridBenchmark import PowerGridBenchmark
from lips.evaluation.powergrid_evaluation import PowerGridEvaluation
from pprint import pprint
from utils.compute_score import compute_global_score
import warnings
print("-1.2.all")

logger.info(f"Starting training with VERSION: {VERSION}")
logger.info(f"Output directory: {output_dir}")
logger.info(f"Model will be saved as: {model_name}")

# Import network architecture from network.py
from network_physics import (
    ProcessingInput, ResNetLayer, LtauNoAdd, DecoderLayer, 
    UnscalingLayer, LEAPNet
)

# Custom Loss Function: MSE + P1 + P2  + P4
class CustomLoss(nn.Module):
    """
    Custom loss function: MSE + P1 + P2  + P4 physics constraints
    """
    
    def __init__(self, physics_weight=0.1):
        super(CustomLoss, self).__init__()
        self.physics_weight = physics_weight

    def forward(self, predict, target, data=None, YBus=None, obs=None, env=None, ifPrint=False):
        """
        Forward pass to compute the total loss
        
        Args:
            predict: Model predictions
            target: True labels
            data: Input data (optional)
            YBus: YBus matrix (optional)
            obs: Observation data (optional)
            env: Environment object (optional)
            ifPrint: Whether to print debug information (optional)
            
        Returns:
            total_loss: Total loss
            loss_dict: Dictionary containing various loss components
        """
        # Calculate MSE loss component
        mse_components = self.compute_mse_components(predict, target)
        
        # Calculate physics constraint loss, only for print.
        # physics_loss is always zero
        physics_loss, physics_details = self.compute_physics_loss(predict, target, data)
        
        # Total loss = MSE loss + physics_weight * physics_loss
        total_loss = mse_components['MSE_total'] + self.physics_weight * physics_loss
        
        # Build loss dictionary
        loss_dict = {
            'MSE_total': mse_components['MSE_total'],
            'Physics': physics_loss,
            'Total': total_loss,
            'P1': physics_details['P1'],
            'P2': physics_details['P2'],
            'P4': physics_details['P4']
        }
        
        return total_loss, loss_dict
    
    def compute_mse_components(self, predict, target):
        """
        Directly compute the MSE loss between the entire predict and target
        
        Args:
            predict: Model predictions
            target: True labels
            
        Returns:
            mse_components: Dictionary containing MSE loss
        """
        # Directly compute the MSE loss between the entire predict and target
        mse_total = torch.mean(torch.pow((predict - target), 2))
        
        return {
            'MSE_total': mse_total
        }
    
    def compute_physics_loss(self, predict, target, data):
        """
        Calculate physics constraint loss - includes P1, P2, P4
        
        Args:
            predict: Model predictions
            target: True labels
            data: Input data
            
        Returns:
            physics_loss: Total physics loss
            physics_details: Dictionary containing P1, P2, P4 losses
        """
        # Partition the predict data
        a_or_pred = predict[:,:186]  # 186 lines
        a_ex_pred = predict[:,186:372]
        p_or_pred = predict[:,372:558]
        p_ex_pred = predict[:,558:744]
        v_or_pred = predict[:,744:930]
        v_ex_pred = predict[:,930:]
        
        # Calculate physics loss terms
        physics_loss = torch.tensor(0.0, device=predict.device)
        
        # P1: Penalize parts of a_or_pred and a_ex_pred that are less than 0
        p1_loss = self.compute_p1_loss(a_or_pred, a_ex_pred) * 10
        physics_loss += p1_loss 
        
        # P2: Penalize parts of v_or_pred and v_ex_pred that are less than 0
        p2_loss = self.compute_p2_loss(v_or_pred, v_ex_pred) * 100
        physics_loss += p2_loss 
        
        
        # P4: Ensure power ratio is within the range [0.005, 0.04]
        p4_loss = self.compute_p4_loss(p_or_pred, p_ex_pred, data) * 1000
        physics_loss += p4_loss

        physics_loss = torch.tensor(0.0, device=predict.device)

        
        return physics_loss, {
            'P1': p1_loss,
            'P2': p2_loss,
            'P4': p4_loss
        }
    
    def compute_p1_loss(self, a_or_pred, a_ex_pred):
        """
        Calculate P1 loss: Penalize parts of a_or_pred and a_ex_pred that are less than 0
        Use ReLU function to penalize negative values
        
        Args:
            a_or_pred: Predicted a_or values
            a_ex_pred: Predicted a_ex values
            
        Returns:
            p1_loss: P1 loss value
        """
        # Penalize parts of a_or_pred that are less than 0
        a_or_negative_penalty = torch.relu(-a_or_pred)
        # Penalize parts of a_ex_pred that are less than 0
        a_ex_negative_penalty = torch.relu(-a_ex_pred)
        
        # Calculate the average loss
        p1_loss = torch.mean(a_or_negative_penalty) + torch.mean(a_ex_negative_penalty)
        
        return p1_loss
    
    def compute_p2_loss(self, v_or_pred, v_ex_pred):
        """
        Calculate P2 loss: Penalize parts of v_or_pred and v_ex_pred that are less than 0
        Use ReLU function to penalize negative values
        
        Args:
            v_or_pred: Predicted v_or values
            v_ex_pred: Predicted v_ex values
            
        Returns:
            p2_loss: P2 loss value
        """
        # Penalize parts of v_or_pred that are less than 0
        v_or_negative_penalty = torch.relu(-v_or_pred)
        # Penalize parts of v_ex_pred that are less than 0
        v_ex_negative_penalty = torch.relu(-v_ex_pred)
        
        # Calculate the average loss
        p2_loss = torch.mean(v_or_negative_penalty) + torch.mean(v_ex_negative_penalty)
        
        return p2_loss
    
    
    def compute_p4_loss(self, p_or_pred, p_ex_pred, data):
        """
        Calculate P4 loss: Ensure power ratio is within the range [0.005, 0.04]
        Use ReLU function to penalize values outside the range
        
        Args:
            p_or_pred: Predicted p_or values
            p_ex_pred: Predicted p_ex values
            data: Input data, containing prod_p
            
        Returns:
            p4_loss: P4 loss value
        """
        # Extract prod_p data from data
        if data is None:
            # If data is not provided, return 0 loss
            return torch.tensor(0.0, device=p_or_pred.device)
        
        try:
            # Extract prod_p based on data structure (first 62 elements are generator powers)
            if isinstance(data, torch.Tensor):
                prod_p_data = data[:, :62]  # First 62 elements are generator powers
            else:
                # If prod_p cannot be extracted, return 0 loss
                return torch.tensor(0.0, device=p_or_pred.device)
            
            # Calculate power ratio: (p_ex_pred + p_or_pred).sum(dim=1) / prod_p_data.sum(dim=1)
            p_sum = p_or_pred + p_ex_pred
            p_sum_total = p_sum.sum(dim=1)  # Sum over each line
            prod_p_total = prod_p_data.sum(dim=1)  # Sum over prod_p
            
            # Avoid division by zero
            prod_p_total = torch.clamp(prod_p_total, min=1e-8)
            
            # Calculate power ratio
            power_ratio = p_sum_total / prod_p_total
            
            # Use ReLU to penalize values outside the [0.005, 0.04] range
            # Penalize values less than 0.005
            lower_bound_penalty = torch.relu(0.005 - power_ratio)
            # Penalize values greater than 0.04
            upper_bound_penalty = torch.relu(power_ratio - 0.04)
            
            # Calculate the average loss
            p4_loss = torch.mean(lower_bound_penalty) + torch.mean(upper_bound_penalty)
            
            return p4_loss
            
        except Exception as e:
            # If any error occurs, return 0 loss
            print(f"Warning: Could not compute P4 loss: {e}")
            return torch.tensor(0.0, device=p_or_pred.device)

# Normalize the data
def process_dataset(dataset: DataSet, 
                    scaler: Union[Scaler, None] = None,
                    training: bool=False,
                    ) -> tuple:
        if training:
            inputs, outputs = dataset.extract_data(concat=True)
            
            if scaler is not None:
                inputs, outputs = scaler.fit_transform(dataset)
        else:
            inputs, outputs = dataset.extract_data(concat=True)
            if scaler is not None:
                inputs, outputs = scaler.transform(dataset)
        
        return inputs, outputs

# Generate dataloader with batch processing
def process_dataloader(inputs,outputs,
                       batch_size: int=BATCH_SIZE,
                       shuffle: bool=False,
                       YBus=None,
                       dtype1=torch.float32,
                       dtype2=torch.complex128):
    inputs = np.concatenate([inputs[0][0],inputs[0][1],inputs[0][2],
                      inputs[0][3],inputs[1][0],inputs[1][1]],axis=1)
    
    outputs = np.concatenate([outputs[0],outputs[1],outputs[2],
                      outputs[3],outputs[4],outputs[5]],axis=1)
    
    torch_dataset = TensorDataset(torch.tensor(inputs, dtype=dtype1), 
                                  torch.tensor(outputs, dtype=dtype1))
    data_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle,pin_memory=True)
    return data_loader

def post_process(predictions, scaler: Union[Scaler, None] = None):
        if scaler is not None:
            predictions = scaler.inverse_transform(predictions)
        return predictions

def save_predictions_and_targets(predictions, targets, dataset_name, data_dir):
    """
    Save predictions and true values as numpy files
    
    Args:
        predictions: Dictionary of predictions, containing 6 physical quantities
        targets: Dictionary of true values, containing 6 physical quantities  
        dataset_name: Dataset name ('test' or 'ood_test')
        data_dir: Data saving directory
    """
    # List of physical quantities
    physics_vars = ['a_or', 'a_ex', 'p_or', 'p_ex', 'v_or', 'v_ex']
    
    # List of data types
    data_types = ['predict', 'target']
    
    logger.info(f"Saving {dataset_name} dataset predictions and targets...")
    
    for var in physics_vars:
        for data_type in data_types:
            # Select data source
            if data_type == 'predict':
                data = predictions[var]
            else:
                data = targets[var]
            
            # Construct filename
            filename = f"{dataset_name}_{data_type}_{var}.npy"
            filepath = os.path.join(data_dir, filename)
            
            # Save as numpy file
            np.save(filepath, data)
            logger.info(f"Saved {filename}: shape {data.shape}")
    
    logger.info(f"Successfully saved {len(physics_vars) * len(data_types)} files for {dataset_name} dataset")

def verify_saved_data(data_dir):
    """
    Verify the saved data files
    
    Args:
        data_dir: Data saving directory
    """
    logger.info("Verifying saved data files...")
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory does not exist: {data_dir}")
        return
    
    # List of expected files
    expected_files = []
    datasets = ['test', 'ood_test']
    data_types = ['predict', 'target']
    physics_vars = ['a_or', 'a_ex', 'p_or', 'p_ex', 'v_or', 'v_ex']
    
    for dataset in datasets:
        for data_type in data_types:
            for var in physics_vars:
                expected_files.append(f"{dataset}_{data_type}_{var}.npy")
    
    # Check if files exist
    actual_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    actual_files.sort()
    
    logger.info(f"Expected files: {len(expected_files)}")
    logger.info(f"Actual files: {len(actual_files)}")
    
    missing_files = set(expected_files) - set(actual_files)
    if missing_files:
        logger.warning(f"Missing files: {sorted(missing_files)}")
    else:
        logger.info("All expected files are present!")
    
    # Verify file content
    for file in actual_files:
        file_path = os.path.join(data_dir, file)
        try:
            data = np.load(file_path)
            logger.info(f"  {file}: shape {data.shape}, dtype {data.dtype}")
        except Exception as e:
            logger.error(f"  {file}: Error loading - {e}")

# KKT constraint projection function
def KKTP6ABb(obs, device):
    """
    Calculate the coefficient matrices A*, B*, b* for KKT constraint projection
    Used to project line powers onto the subspace that satisfies the power balance constraint
    """
    sub_number = 118
    line_number = 186

    line_or_to_subid = torch.tensor(obs.line_or_to_subid).to(device)
    line_ex_to_subid = torch.tensor(obs.line_ex_to_subid).to(device)

    # Construct B1 and B2 matrices, representing the connection relationship between lines and substations
    B1 = torch.zeros((line_number, sub_number))
    B2 = torch.zeros((line_number, sub_number))
    B1[torch.arange(line_number), line_or_to_subid] = 1
    B2[torch.arange(line_number), line_ex_to_subid] = 1
    B1 = B1.T
    B2 = B2.T
    
    # Construct constraint matrix
    A = torch.eye(sub_number)
    B = -torch.concatenate((B1, B2), axis=1)
    b = torch.zeros(sub_number)

    # Calculate projection matrix
    BBTinversed = torch.linalg.inv(B @ B.T)
    A_star = - B.T @ BBTinversed @ A
    B_star = torch.eye(line_number*2) - B.T @ BBTinversed @ B
    b_star = B.T @ BBTinversed @ b

    A_star = A_star.to(device)
    B_star = B_star.to(device)
    b_star = b_star.to(device).unsqueeze(1)

    return A_star, B_star, b_star

def KKTP6Xp(obs, device, data):
    """
    Calculate the net power injection for each substation
    """
    batch_size = data.shape[0]
    sub_number = 118
    line_number = 186
    prod_p_data = data[:, :62]
    load_p_data = data[:, 124:223]
    gen_to_subid = torch.tensor(obs.gen_to_subid).to(device)
    load_to_subid = torch.tensor(obs.load_to_subid).to(device)
    
    # Use one-hot encoding to calculate the generation and load power for each substation
    gen_to_subid_one_hot = (gen_to_subid.unsqueeze(0) == torch.arange(sub_number).unsqueeze(1).to(device)).float()
    gen_power_sums = torch.matmul(gen_to_subid_one_hot, prod_p_data.T.float())
    load_to_subid_one_hot = (load_to_subid.unsqueeze(0) == torch.arange(sub_number).unsqueeze(1).to(device)).float()
    load_power_sums = torch.matmul(load_to_subid_one_hot, load_p_data.T.float())
    
    # Calculate net power injection
    X_p = (gen_power_sums - load_power_sums)
    return X_p

# P6 KKT projection function - linearly transform line powers to satisfy power balance constraints
def apply_p6_kkt_projection(predict, data, obs, device):
    """
    Apply P6 KKT projection, linearly transforming line powers to satisfy power balance constraints
    
    Args:
        predict: Model predictions
        data: Input data (un-normalized)
        obs: Observation data
        device: Computation device
        
    Returns:
        predict_projected: Predictions after applying P6 KKT projection
    """
    # Calculate KKT projection coefficient matrices (if not already calculated)
    if not hasattr(apply_p6_kkt_projection, 'P6A_star') or apply_p6_kkt_projection.P6A_star is None:
        apply_p6_kkt_projection.P6A_star, apply_p6_kkt_projection.P6B_star, apply_p6_kkt_projection.P6b_star = KKTP6ABb(obs, device)
    
    # Calculate the net power injection for each substation
    P6X = KKTP6Xp(obs, device, data)
    
    # Extract the line power part
    p_tmp = predict[:, 372:744].T  # p_or_pred + p_ex_pred
    
    # Apply P6 KKT projection: y_tilde = A* @ X + B* @ p + b*
    p_projected = (apply_p6_kkt_projection.P6A_star @ P6X + 
                   apply_p6_kkt_projection.P6B_star @ p_tmp + 
                   apply_p6_kkt_projection.P6b_star).T
    
    # Reassign the projected powers back into the predictions
    predict_projected = predict.clone()
    predict_projected[:, 372:558] = p_projected[:, :186]  # p_or_pred
    predict_projected[:, 558:744] = p_projected[:, 186:]  # p_ex_pred
    
    return predict_projected

from lips.benchmark.powergridBenchmark import get_env, get_kwargs_simulator_scenario

if __name__ == "__main__":
    print("0.device")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    logger.info(f"Using device: {device}")
    
    # Use some required pathes
    DATA_PATH = pathlib.Path().resolve() / "input_data_local" / "lips_idf_2023"
    BENCH_CONFIG_PATH = pathlib.Path().resolve() / "configs" / "benchmarks" / "lips_idf_2023.ini"
    SIM_CONFIG_PATH = pathlib.Path().resolve() / "configs" / "simulators"
    TRAINED_MODELS = pathlib.Path().resolve() / "input_data_local" / "trained_models"
    
    # Create model save directory
    model_save_dir = os.path.join(output_dir, "models")
    os.makedirs(model_save_dir, exist_ok=True)
    
    benchmark_kwargs = {"attr_x": ("prod_p", "prod_v", "load_p", "load_q"),
                        "attr_y": ("a_or", "a_ex", "p_or", "p_ex", "v_or", "v_ex"),
                        "attr_tau": ("line_status", "topo_vect"),
                        "attr_physics": ()}
    print("1.benchmark")
    logger.info("Loading benchmark data...")
    benchmark = PowerGridBenchmark(benchmark_name="Benchmark_competition",
                                benchmark_path=DATA_PATH,
                                load_data_set=True,
                                log_path=None,
                                config_path=BENCH_CONFIG_PATH,
                                load_ybus_as_sparse=True,
                                **benchmark_kwargs
                                )
    
    env_kwargs = get_kwargs_simulator_scenario(benchmark.config)
    env = get_env(env_kwargs)
    obs = env.reset()
    print("2.Data Processing")
    logger.info("Processing dataset...")
    
    # Process data
    powerGridScaler = PowerGridScaler()
    inputs_train,outputs_train = process_dataset(dataset = benchmark.train_dataset,
                                                scaler = powerGridScaler,
                                                training = True)
    dataloader_train = process_dataloader(inputs = inputs_train,
                                        outputs = outputs_train,
                                        shuffle=False)
        
    inputs_val,outputs_val = process_dataset(dataset = benchmark.val_dataset,
                                                scaler = powerGridScaler,
                                                training = False)
    dataloader_val = process_dataloader(inputs = inputs_val,
                                        outputs = outputs_val,
                                        shuffle = True)
    
    print("3.Model Initialization")
    logger.info("Initializing LEAPNet model...")
    processingInputInstance = ProcessingInput(322,1024)
    resNetLayerInstance1 = ResNetLayer(1024)
    resNetLayerInstance2 = ResNetLayer(1024)
    ltauNoAddInstance = LtauNoAdd(1024,726)
    decoderLayerInstance1 = DecoderLayer(1024,186)
    decoderLayerInstance2 = DecoderLayer(1024,186)
    decoderLayerInstance3 = DecoderLayer(1024,186)
    decoderLayerInstance4 = DecoderLayer(1024,186)
    decoderLayerInstance5 = DecoderLayer(1024,186)
    decoderLayerInstance6 = DecoderLayer(1024,186)
    unscalingLayerInstance = UnscalingLayer(powerGridScaler,device)
    
    leapNetInstance = LEAPNet(processingInput=processingInputInstance,
                          resNetLayer1=resNetLayerInstance1,
                          resNetLayer2=resNetLayerInstance2,
                          ltauNoAdd=ltauNoAddInstance,
                          decoderLayer1=decoderLayerInstance1,
                          decoderLayer2=decoderLayerInstance2,
                          decoderLayer3=decoderLayerInstance3,
                          decoderLayer4=decoderLayerInstance4,
                          decoderLayer5=decoderLayerInstance5,
                          decoderLayer6=decoderLayerInstance6,
                          unscalingLayer=unscalingLayerInstance,
                          powerGridScaler=powerGridScaler,
                          device=device,
                          obs=obs
                          )
    
    print("4.Training")
    logger.info("Starting training...")
    
    # First build the model, then count the number of parameters
    leapNetInstance.build_model()
    logger.info("Counting model parameters...")
    total_params = leapNetInstance.count_parameters()
    logger.info(f"Total NN Parameters: {total_params:,} ({total_params/1e3:.1f}K)")
    
    # Save the initial model
    initial_model_path = os.path.join(model_save_dir, f"{VERSION}_initial.model")
    torch.save(leapNetInstance, initial_model_path)
    logger.info(f"Initial model saved to: {initial_model_path}")
    
    # Train using the custom loss function
    custom_loss = CustomLoss(physics_weight=PHYSICS_WEIGHT)
    leapNetInstance.loss_function = custom_loss
    leapNetInstance.obs = obs  # Pass the obs parameter to the model
    
    train_losses,val_losses = leapNetInstance.train(
        train_loader=dataloader_train,
        val_loader=dataloader_val,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_factor=LR_SCHEDULER_FACTOR,
        lr_scheduler_patience=LR_SCHEDULER_PATIENCE,
        min_lr=MIN_LR,
        early_stop_patience=EARLY_STOP_PATIENCE,
        min_improvement=MIN_IMPROVEMENT,
        save_every_n_epochs=SAVE_EVERY_N_EPOCHS,
        start=False
    )
    
    # Save the model immediately after training is complete
    final_model_path = os.path.join(model_save_dir, f"{VERSION}_final.model")
    torch.save(leapNetInstance, final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")
    
    print("5.Visualization")
    logger.info("Generating convergence plots...")
    logger.info(f"Saving convergence plot to: {save_img_path1}")
    
    # Check if the path exists and is a directory, if so, delete it
    if os.path.exists(save_img_path1) and os.path.isdir(save_img_path1):
        logger.warning(f"Removing existing directory: {save_img_path1}")
        os.rmdir(save_img_path1)
    
    try:
        leapNetInstance.visualize_convergence(save_path=save_img_path1)
        logger.info("Convergence plot saved successfully")
    except Exception as e:
        logger.error(f"Failed to save convergence plot: {e}")
        # Continue execution, do not interrupt the program due to failure to save the image
    
    print("6.Prediction and Evaluation")
    logger.info("Making predictions on test dataset...")
    predictions,observations = leapNetInstance.predict(benchmark._test_dataset, 
                                                      process_dataset=process_dataset, 
                                                      process_dataloader=process_dataloader)
    
    # Save test set predictions and true values
    logger.info("Saving test dataset predictions and targets...")
    save_predictions_and_targets(predictions, observations, "test", data_dir)
    
    # Use the path of the saved final model
    SAVE_PATH = final_model_path
    logger.info(f"Using final model: {SAVE_PATH}")

    # Evaluate the model
    logger.info("Evaluating model performance...")
    env = get_env(get_kwargs_simulator_scenario(benchmark.config))
    evaluator = PowerGridEvaluation(benchmark.config)
    metrics_test = evaluator.evaluate(observations=benchmark._test_dataset.data,
                                    predictions=predictions,
                                    dataset=benchmark._test_dataset,
                                    augmented_simulator=leapNetInstance,
                                    env=env)
    print("Test metrics:")
    pprint(metrics_test)
    logger.info(f"Test metrics: {metrics_test}")
    
    # Display evaluation results with levels
    from evaluation_utils import evaluate_metrics_with_levels, print_overall_summary
    print("\n" + "="*80)
    print("🎯 Test Dataset Evaluation Results")
    print("="*80)
    evaluate_metrics_with_levels(metrics_test)
    print_overall_summary(metrics_test)

    metrics_all = dict()
    metrics_all["test"] = metrics_test
    predictions, observations = leapNetInstance.predict(benchmark._test_ood_topo_dataset,
                                                       process_dataset=process_dataset, 
                                                       process_dataloader=process_dataloader)
    
    # Save OOD test set predictions and true values
    logger.info("Saving OOD test dataset predictions and targets...")
    save_predictions_and_targets(predictions, observations, "ood_test", data_dir)
    
    evaluator = PowerGridEvaluation(benchmark.config)
    metrics_ood = evaluator.evaluate(observations=benchmark._test_ood_topo_dataset.data,
                                    predictions=predictions,
                                    env=env)
    print("Test OOD metrics:")
    pprint(metrics_ood)
    logger.info(f"Test OOD metrics: {metrics_ood}")
    metrics_all["test_ood_topo"] = metrics_ood
    
    # Display OOD test set evaluation results with levels
    print("\n" + "="*80)
    print("🌐 OOD Test Dataset Evaluation Results")
    print("="*80)
    evaluate_metrics_with_levels(metrics_ood)
    print_overall_summary(metrics_ood)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        score = compute_global_score(metrics_all, benchmark.config)
        print(f"Global Score: {score}")
        logger.info(f"Global Score: {score}")
    
    # Save evaluation results
    results_file = os.path.join(output_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"VERSION: {VERSION}\n")
        f.write(f"Training completed at: {datetime.now()}\n")
        f.write(f"Model saved to: {SAVE_PATH}\n\n")
        f.write("MODEL PARAMETERS:\n")
        f.write(f"Total NN Parameters: {total_params:,} ({total_params/1e3:.1f}K)\n\n")
        f.write("Test Metrics:\n")
        f.write(str(metrics_test))
        f.write("\n\nTest OOD Metrics:\n")
        f.write(str(metrics_ood))
        f.write(f"\n\nGlobal Score: {score}\n")
    
    logger.info(f"Evaluation results saved to: {results_file}")
    
    # Display saved data file information
    logger.info("="*50)
    logger.info("SAVED DATA FILES SUMMARY")
    logger.info("="*50)
    logger.info(f"Data directory: {data_dir}")
    
    # List the saved files
    if os.path.exists(data_dir):
        saved_files = os.listdir(data_dir)
        saved_files.sort()
        logger.info(f"Total files saved: {len(saved_files)}")
        for file in saved_files:
            if file.endswith('.npy'):
                file_path = os.path.join(data_dir, file)
                file_size = os.path.getsize(file_path)
                logger.info(f"  {file} ({file_size} bytes)")
    
    # Verify the saved data
    verify_saved_data(data_dir)
    
    logger.info("Training completed successfully!")