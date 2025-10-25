# -*- coding: utf-8 -*-
print("MSE Only Training - No Physics Loss")

# ========== Configuration Parameters ==========
VERSION = "MSE_only"  # Version number, can be modified

RANDOM_SEED = 123  # Fixed random seed
# Training Hyperparameters
EPOCHS = 150  # Maximum number of training epochs
LEARNING_RATE = 3e-4  # Initial learning rate
WEIGHT_DECAY = 1e-4  # L2 regularization coefficient
BATCH_SIZE = 256  # Batch size

# Learning Rate Scheduler Parameters
LR_SCHEDULER_FACTOR = 0.5  # Learning rate decay factor
LR_SCHEDULER_PATIENCE = 5  # Learning rate scheduler patience
MIN_LR = 1e-8  # Minimum learning rate

# Early Stopping Parameters
EARLY_STOP_PATIENCE = 20  # Early stopping patience
MIN_IMPROVEMENT = 3e-4  # Minimum improvement threshold

# Model Saving Parameters
SAVE_EVERY_N_EPOCHS = 10  # Save a checkpoint every N epochs
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

# Create data folder to save predicted and true values
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

# Log data directory creation
logger.info(f"Created data directory: {data_dir}")

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
from network_mse import (
    ProcessingInput, ResNetLayer, LtauNoAdd, DecoderLayer, 
    UnscalingLayer, LEAPNet
)

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
        data_dir: Save directory
    """
    logger.info(f"Saving predictions and targets for {dataset_name} dataset...")
    
    # Names of the 6 physical quantities
    physics_quantities = ['a_or', 'a_ex', 'p_or', 'p_ex', 'v_or', 'v_ex']
    
    # Save predictions and true values
    for quantity in physics_quantities:
        # Save predictions
        pred_file = os.path.join(data_dir, f"{dataset_name}_predict_{quantity}.npy")
        np.save(pred_file, predictions[quantity])
        logger.info(f"Saved {quantity} predictions to: {pred_file}")
        
        # Save true values
        target_file = os.path.join(data_dir, f"{dataset_name}_target_{quantity}.npy")
        np.save(target_file, targets[quantity])
        logger.info(f"Saved {quantity} targets to: {target_file}")
    
    logger.info(f"Successfully saved all predictions and targets for {dataset_name} dataset")

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
    
    # Save test dataset predictions and true values
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
    
    # Save OOD test dataset predictions and true values
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
    logger.info("Training completed successfully!")