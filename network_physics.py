# -*- coding: utf-8 -*-
"""
Neural Network Architecture for LEAPNet Model
Contains all the model components and architecture definitions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pathlib
import matplotlib.pyplot as plt
from lips.augmented_simulators import AugmentedSimulator
from torch.utils.data import TensorDataset, DataLoader

# 导入KKT约束函数
def KKTP6ABb(obs, device):
    """计算KKT约束投影的系数矩阵A*, B*, b*"""
    sub_number = 118
    line_number = 186

    line_or_to_subid = torch.tensor(obs.line_or_to_subid).to(device)
    line_ex_to_subid = torch.tensor(obs.line_ex_to_subid).to(device)

    B1 = torch.zeros((line_number, sub_number))
    B2 = torch.zeros((line_number, sub_number))
    B1[torch.arange(line_number), line_or_to_subid] = 1
    B2[torch.arange(line_number), line_ex_to_subid] = 1
    B1 = B1.T
    B2 = B2.T
    
    A = torch.eye(sub_number)
    B = -torch.concatenate((B1, B2), axis=1)
    b = torch.zeros(sub_number)

    BBTinversed = torch.linalg.inv(B @ B.T)
    A_star = - B.T @ BBTinversed @ A
    B_star = torch.eye(line_number*2) - B.T @ BBTinversed @ B
    b_star = B.T @ BBTinversed @ b

    A_star = A_star.to(device)
    B_star = B_star.to(device)
    b_star = b_star.to(device).unsqueeze(1)

    return A_star, B_star, b_star

def KKTP6Xp(obs, device, data):
    """计算每个变电站的净功率注入"""
    batch_size = data.shape[0]
    sub_number = 118
    line_number = 186
    prod_p_data = data[:, :62]
    load_p_data = data[:, 124:223]
    gen_to_subid = torch.tensor(obs.gen_to_subid).to(device)
    load_to_subid = torch.tensor(obs.load_to_subid).to(device)
    
    gen_to_subid_one_hot = (gen_to_subid.unsqueeze(0) == torch.arange(sub_number).unsqueeze(1).to(device)).float()
    gen_power_sums = torch.matmul(gen_to_subid_one_hot, prod_p_data.T.float())
    load_to_subid_one_hot = (load_to_subid.unsqueeze(0) == torch.arange(sub_number).unsqueeze(1).to(device)).float()
    load_power_sums = torch.matmul(load_to_subid_one_hot, load_p_data.T.float())
    
    X_p = (gen_power_sums - load_power_sums)
    return X_p


class ProcessingInput(nn.Module):
    """Input processing layer"""
    def __init__(self, input_size, middle_size) -> None:
        super(ProcessingInput,self).__init__()
        self.middle_size = middle_size
        self.input_size = input_size
        
    def build_model(self):
        self.linear = nn.Linear(self.input_size, self.middle_size)
        self.leakyRelu = nn.LeakyReLU()
        
    def forward(self, data):
        data = self.linear(data)
        data = self.leakyRelu(data)
        return data

class ResNetLayer(nn.Module):
    """Residual Network Layer"""
    def __init__(self,
                 middle_size,
                 activation = nn.LeakyReLU):
        super(ResNetLayer,self).__init__()
        self.middle_size = middle_size
        self.d = None
        self.e = None
        self.activation = activation()
    def build_model(self):
        self.e = nn.Linear(self.middle_size, self.middle_size)
        self.d = nn.Linear(self.middle_size, self.middle_size)
    
    def forward(self, x):
        res = self.e(x)
        res = self.activation(res)
        res = self.d(res)
        res = self.activation(res)
        res = x+res
        
        return res
        
class LtauNoAdd(nn.Module):
    """L-tau layer without addition"""
    def __init__(self,middle_size,tau_size):
        super(LtauNoAdd,self).__init__()
        self.middle_size = middle_size
        self.tau_size = tau_size
        self.e = None
        self.d = None
    
    def build_model(self):
        self.e = nn.Linear(self.middle_size,self.tau_size)
        self.d = nn.Linear(self.tau_size,self.middle_size)

    def forward(self,x,tau):
        tmp = self.e(x)
        tmp = torch.mul(tmp,tau)
        res = self.d(tmp)
        return res
    
class DecoderLayer(nn.Module):
    """Decoder layer for output generation"""
    def __init__(self,middle_size,output_size,activation=nn.LeakyReLU,dropout_rate=0.0):
        super(DecoderLayer,self).__init__()
        self.middle_size = middle_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        self.d1 = None
        self.d2 = None
        self.activation = activation()
        self.dropout = None
    def build_model(self):
        self.d1 = nn.Linear(self.middle_size,self.middle_size)
        self.d2 = nn.Linear(self.middle_size,self.output_size)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self,x):
        res = self.d1(x)
        res = self.activation(res)
        if self.dropout is not None:
            res = self.dropout(res)
        res = self.d2(res)
        
        return res
    
class OutputLayer(nn.Module):
    """Output layer"""
    def __init__(self,middle_size,output_size):
        super(OutputLayer,self).__init__()
        self.middle_size = middle_size
        self.output_size = output_size
        
        self.model = None
        
    def build_model(self):
        self.model = nn.Linear(self.middle_size,self.output_size)
        
    def forward(self,x):
        return self.model(x)
    
class UnscalingLayer(nn.Module):
    """Unscaling layer for data denormalization"""
    def __init__(self,powerGridScaler,device):
        super(UnscalingLayer, self).__init__()
        self.powerGridScaler = powerGridScaler
        self.device = device
        self._m_x,self._sd_x, self._m_y,self._sd_y, self._m_tau,self._sd_tau = powerGridScaler.get_all_m_sd()

    def build_model(self):
        self._m_x = np.concatenate(self._m_x, axis=0)
        if(type(self._sd_x[1])== float):
            self._sd_x[1] = np.full(62, self._sd_x[1])
        
        self._m_y_tmp = []
        self._sd_x = np.concatenate(self._sd_x, axis=0)
        if(len(self._m_y) == 6):
            self._m_y_tmp.append(np.full(186, self._m_y[0]))
            self._m_y_tmp.append(np.full(186, self._m_y[1])) 
            self._m_y_tmp.append(np.full(186, self._m_y[2]))
            self._m_y_tmp.append(np.full(186, self._m_y[3]))
            self._m_y_tmp.append(np.full(186, self._m_y[4]))
            self._m_y_tmp.append(np.full(186, self._m_y[5]))
            self._m_y = np.concatenate(self._m_y_tmp, axis=0)
        else:
            self._m_y = np.concatenate(self._m_y, axis=0)
        self._sd_y = np.concatenate(self._sd_y, axis=0)

        self._m_x = torch.tensor(self._m_x).to(self.device)
        self._sd_x = torch.tensor(self._sd_x).to(self.device)
        self._m_y = torch.tensor(self._m_y).to(self.device)
        self._sd_y = torch.tensor(self._sd_y).to(self.device)
        self._m_tau = torch.tensor(self._m_tau).to(self.device)
        self._sd_tau = torch.tensor(self._sd_tau).to(self.device)

    def forward(self,x,y,tau):
        x = x*self._sd_x + self._m_x
        y = y*self._sd_y + self._m_y
        return x,y,tau
        
    def show(self):
        print(f"m_x:{self._m_x}")
        print(f"sd_x:{self._sd_x}")
        print(f"m_y:{self._m_y}")
        print(f"sd_y:{self._sd_y}")
        print(f"m_tau:{self._m_tau}")
        print(f"sd_tau:{self._sd_tau}")
    
    def show_shape(self):
        print(f"m_x:{self._m_x.shape}")
        print(f"sd_x:{self._sd_x.shape}")
        print(f"m_y:{self._m_y.shape}")
        print(f"sd_y:{self._sd_y.shape}")
        print(f"m_tau:{self._m_tau}")
        print(f"sd_tau:{self._sd_tau}")

class MSELoss(nn.Module):
    """Simple MSE Loss Function"""
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, predict, target, data=None, YBus=None, obs=None, env=None, ifPrint=False):
        # Only compute MSE loss
        MSE = torch.mean(torch.pow((predict - target), 2))
        
        return MSE, {'MSE': MSE}

class LEAPNet(AugmentedSimulator):
    """LEAPNet Model with MSE Loss Only"""
    def __init__(self,
             processingInput,
             resNetLayer1,
             resNetLayer2,
             ltauNoAdd,
             decoderLayer1,
             decoderLayer2,
             decoderLayer3,
             decoderLayer4,
             decoderLayer5,
             decoderLayer6,
             unscalingLayer,
             powerGridScaler,
             device,
             obs
             ):
        
        super(LEAPNet, self).__init__()
        self.mse_list = []
        self.val_losses = []
        self.train_losses = []
        self.processingInput = processingInput
        self.resNetLayer1 = resNetLayer1
        self.resNetLayer2 = resNetLayer2
        self.ltauNoAdd = ltauNoAdd
        self.decoderLayer1 = decoderLayer1
        self.decoderLayer2 = decoderLayer2
        self.decoderLayer3 = decoderLayer3
        self.decoderLayer4 = decoderLayer4
        self.decoderLayer5 = decoderLayer5
        self.decoderLayer6 = decoderLayer6
        self.unscalingLayer = unscalingLayer
        self.powerGridScaler = powerGridScaler
        self.device = device
        self.obs = obs

    def build_model(self):
        self.processingInput.build_model()
        self.resNetLayer1.build_model()
        self.resNetLayer2.build_model()
        self.ltauNoAdd.build_model()
        self.decoderLayer1.build_model()
        self.decoderLayer2.build_model()
        self.decoderLayer3.build_model()
        self.decoderLayer4.build_model()
        self.decoderLayer5.build_model()
        self.decoderLayer6.build_model()
        self.unscalingLayer.build_model()

    def apply_constraints(self, a_or_pred, a_ex_pred, v_or_pred, v_ex_pred, data_tau):
        """
        Apply the softplus function and the line_status constraint to the specific output variable
        """
        # 提取line_status (data_tau的第一部分，前186个元素)
        line_status = data_tau[:, :186]  # line_status shape: [batch_size, 186]
        
        # 对v_or, v_ex, a_or, a_ex应用softplus函数
        v_or_constrained = F.softplus(v_or_pred)
        v_ex_constrained = F.softplus(v_ex_pred)
        a_or_constrained = F.softplus(a_or_pred)
        a_ex_constrained = F.softplus(a_ex_pred)
        
        # 用line_status逐元素相乘
        v_or_final = v_or_constrained * (1-line_status)
        v_ex_final = v_ex_constrained * (1-line_status)
        a_or_final = a_or_constrained * (1-line_status)
        a_ex_final = a_ex_constrained * (1-line_status)
        
        return a_or_final, a_ex_final, v_or_final, v_ex_final

    def train(self,train_loader,val_loader,
              epochs=100,lr=3e-4,weight_decay=1e-4,
              lr_scheduler_factor=0.5,lr_scheduler_patience=5,min_lr=1e-7,
              early_stop_patience=20,min_improvement=3e-4,
              save_every_n_epochs=10,start=True):
        if start:
            self.build_model()
        self.trained = True
        
        self.processingInput.to(self.device)
        self.resNetLayer1.to(self.device)
        self.resNetLayer2.to(self.device)
        self.ltauNoAdd.to(self.device)
        self.decoderLayer1.to(self.device)
        self.decoderLayer2.to(self.device)
        self.decoderLayer3.to(self.device)
        self.decoderLayer4.to(self.device)
        self.decoderLayer5.to(self.device)
        self.decoderLayer6.to(self.device)
        self.unscalingLayer.to(self.device)
        
        # Adam optimizer with L2 regularization (weight_decay)
        optimizer = torch.optim.Adam([
            {'params':self.processingInput.parameters()},
            {'params':self.resNetLayer1.parameters()},
            {'params':self.resNetLayer2.parameters()},
            {'params':self.ltauNoAdd.parameters()},
            {'params':self.decoderLayer1.parameters()},
            {'params':self.decoderLayer2.parameters()},
            {'params':self.decoderLayer3.parameters()},
            {'params':self.decoderLayer4.parameters()},
            {'params':self.decoderLayer5.parameters()},
            {'params':self.decoderLayer6.parameters()},
            {'params':self.unscalingLayer.parameters()}],
            lr=lr,
            weight_decay=weight_decay  # L2 regularization coefficient
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=lr_scheduler_factor, 
            patience=lr_scheduler_patience, 
            min_lr=min_lr,
            verbose=True
        )
        
        # Use a custom loss function (if provided) or the default MSELoss
        if hasattr(self, 'loss_function') and self.loss_function is not None:
            loss_function = self.loss_function
        else:
            loss_function = MSELoss()
        
        # Calculate the KKT constraint coefficients (if using obs)
        if hasattr(self, 'obs') and self.obs is not None:
            self.P6A_star, self.P6B_star, self.P6b_star = KKTP6ABb(self.obs, self.device)
        
        # Early stopping parameters
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            self.processingInput.train()
            self.resNetLayer1.train()
            self.resNetLayer2.train()
            self.ltauNoAdd.train()
            self.decoderLayer1.train()
            self.decoderLayer2.train()
            self.decoderLayer3.train()
            self.decoderLayer4.train()
            self.decoderLayer5.train()
            self.decoderLayer6.train()
            self.unscalingLayer.train()
            total_loss = 0
            for batch in train_loader:
                data,target = batch
                data_tau = data[:,322:]
                data = data[:,0:322]
                data_tau = data_tau.to(self.device)
                data = data.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                output = self.processingInput(data)
                output = self.resNetLayer1(output)
                output = self.resNetLayer2(output)
                output_mid = self.ltauNoAdd(output,data_tau)
                output = output + output_mid
                a_or_pred = self.decoderLayer1(output)
                a_ex_pred = self.decoderLayer2(output)
                p_or_pred = self.decoderLayer3(output)
                p_ex_pred = self.decoderLayer4(output)
                v_or_pred = self.decoderLayer5(output)
                v_ex_pred = self.decoderLayer6(output)
                output = torch.cat((a_or_pred,a_ex_pred,p_or_pred,p_ex_pred,v_or_pred,v_ex_pred),dim=1)
                
                data_inversed,output_inversed,data_tau = self.unscalingLayer(data,output,data_tau)
                _,target_inversed,_ = self.unscalingLayer(data,target,data_tau)
                
                # Apply constraints to specific output variables
                a_or_final, a_ex_final, v_or_final, v_ex_final = self.apply_constraints(
                    output_inversed[:, 0:186],    # a_or
                    output_inversed[:, 186:372],  # a_ex  
                    output_inversed[:, 744:930],  # v_or
                    output_inversed[:, 930:1116], # v_ex
                    data_tau
                )

                # Apply KKT constraint projection (if using a custom loss function)
                if hasattr(self, 'obs') and self.obs is not None:
                    # 应用P6约束：功率平衡约束
                    P6X = KKTP6Xp(self.obs, self.device, data_inversed)
                    p_tmp = output_inversed[:, 372:744].T
                    p_final = (self.P6A_star @ P6X + self.P6B_star @ p_tmp + self.P6b_star).T
                
                # Recombine and output
                output_inversed = torch.cat((a_or_final, a_ex_final, p_final, v_or_final, v_ex_final), dim=1)
                
                # Prepare the parameters of the loss function
                loss_kwargs = {
                    'predict': output_inversed,
                    'target': target_inversed,
                    'data': torch.cat((data_inversed, data_tau), dim=1)
                }
                
                # If a custom loss function is used, additional parameters need to be passed.
                if hasattr(self, 'obs') and self.obs is not None:
                    loss_kwargs['obs'] = self.obs
                
                # Calculate the loss
                if hasattr(loss_function, 'forward') and len(loss_function.forward.__code__.co_varnames) > 3:
                    # Define the custom loss function and pass all parameters
                    loss, extra_vars = loss_function(**loss_kwargs)
                else:
                    # Default loss function
                    loss, extra_vars = loss_function(output_inversed, target_inversed)

                loss.backward()
                optimizer.step()
                total_loss += loss.item() *len(data)
            
            mean_loss = total_loss / len(train_loader.dataset)
            self.train_losses.append(mean_loss)
            self.mse_list.append(extra_vars['MSE_total'].item())
            
            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                # Print validation loss information
                print(f"Epoch {epoch} - New Validation Loss: {val_loss:.5f}")
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss - min_improvement:
                    best_val_loss = val_loss
                    patience_counter = 0
                    print(f"✓ Validation loss improved to {val_loss:.5f} (Best: {best_val_loss:.5f})")
                else:
                    patience_counter += 1
                    print(f"✗ No improvement for {patience_counter} epochs (Current: {val_loss:.5f}, Best: {best_val_loss:.5f})")
                
                # Early stopping
                if patience_counter >= early_stop_patience:
                    print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                    break
                    
                print(f'Train Epoch: {epoch}, Train_Loss: {mean_loss:.5f}, Val_Loss: {val_loss:.5f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
            else:
                print(f'Train Epoch: {epoch}, Avg_Loss: {mean_loss:.5f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
            
            # Print out each loss component
            print(f'MSE_total: {extra_vars["MSE_total"].item():.5f}')
            print(f'P1: {extra_vars["P1"].item():.5f}')
            print(f'P2: {extra_vars["P2"].item():.5f}')
            print(f'P3: {extra_vars["P3"].item():.5f}')
            print(f'P4: {extra_vars["P4"].item():.5f}')
            print(f'Physics: {extra_vars["Physics"].item():.5f}')
            print(f'Total: {extra_vars["Total"].item():.5f}')
            
            # Regularly save the model
            if (epoch + 1) % save_every_n_epochs == 0:
                checkpoint_path = f"./checkpoint_epoch_{epoch+1}.model"
                torch.save(self, checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
            
        return self.train_losses,self.val_losses
    
    def validate(self,val_loader):
        self.processingInput.eval()
        self.resNetLayer1.eval()
        self.resNetLayer2.eval()
        self.ltauNoAdd.eval()
        self.decoderLayer1.eval()
        self.decoderLayer2.eval()
        self.decoderLayer3.eval()
        self.decoderLayer4.eval()
        self.decoderLayer5.eval()
        self.decoderLayer6.eval()
        self.unscalingLayer.eval()
        total_loss = 0
        # Use the same loss function as during training
        if hasattr(self, 'loss_function') and self.loss_function is not None:
            loss_function = self.loss_function
        else:
            loss_function = MSELoss()
        
        with torch.no_grad():
            for batch in val_loader:
                data,target = batch
                data_tau = data[:,322:]
                data = data[:,0:322]
                data_tau = data_tau.to(self.device)
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.processingInput(data)
                output = self.resNetLayer1(output)
                output = self.resNetLayer2(output)
                output_mid = self.ltauNoAdd(output,data_tau)
                output = output + output_mid
                a_or_pred = self.decoderLayer1(output)
                a_ex_pred = self.decoderLayer2(output)
                p_or_pred = self.decoderLayer3(output)
                p_ex_pred = self.decoderLayer4(output)
                v_or_pred = self.decoderLayer5(output)
                v_ex_pred = self.decoderLayer6(output)
                output = torch.cat((a_or_pred,a_ex_pred,p_or_pred,p_ex_pred,v_or_pred,v_ex_pred),dim=1)
                data_inversed,output_inversed,data_tau = self.unscalingLayer(data,output,data_tau)
                _,target_inversed,_ = self.unscalingLayer(data,target,data_tau)
                
                # Apply the same loss function used during training to impose constraints on the specific output variable.
                a_or_final, a_ex_final, v_or_final, v_ex_final = self.apply_constraints(
                    output_inversed[:, 0:186],    # a_or
                    output_inversed[:, 186:372],  # a_ex  
                    output_inversed[:, 744:930],  # v_or
                    output_inversed[:, 930:1116], # v_ex
                    data_tau
                )

                # Apply KKT constraint projection (if using a custom loss function)
                if hasattr(self, 'obs') and self.obs is not None:
                    # 应用P6约束：功率平衡约束
                    P6X = KKTP6Xp(self.obs, self.device, data_inversed)
                    p_tmp = output_inversed[:, 372:744].T
                    p_final = (self.P6A_star @ P6X + self.P6B_star @ p_tmp + self.P6b_star).T
                
                # Recombine and output
                output_inversed = torch.cat((a_or_final, a_ex_final, p_final, v_or_final, v_ex_final), dim=1)
                
                loss,_ = loss_function(output_inversed,target_inversed)
                total_loss += loss.item() *len(data)
            mean_loss = total_loss / len(val_loader.dataset)
            print(f"Eval:   Avg_Loss: {mean_loss:.5f}")
        return mean_loss
    
    def predict(self,dataset,eval_batch_size=128,process_dataset=None,process_dataloader=None):
        self.processingInput.eval()
        self.resNetLayer1.eval()
        self.resNetLayer2.eval()
        self.ltauNoAdd.eval()
        self.decoderLayer1.eval()
        self.decoderLayer2.eval()
        self.decoderLayer3.eval()
        self.decoderLayer4.eval()
        self.decoderLayer5.eval()
        self.decoderLayer6.eval()
        self.unscalingLayer.eval()
        outputs = []
        targets = []
        
        # If process_dataset is None, use the default data processing method
        if process_dataset is None:
            inputs_test, outputs_test = dataset.extract_data(concat=True)
            if self.powerGridScaler is not None:
                inputs_test, outputs_test = self.powerGridScaler.transform(dataset)
        else:
            inputs_test,outputs_test = process_dataset(dataset = dataset,
                                                       scaler=self.powerGridScaler,
                                                       training=False)
        
        # If process_dataloader is None, use the default method for creating the dataloader
        if process_dataloader is None:
            inputs_test = np.concatenate([inputs_test[0][0],inputs_test[0][1],inputs_test[0][2],
                                  inputs_test[0][3],inputs_test[1][0],inputs_test[1][1]],axis=1)
            outputs_test = np.concatenate([outputs_test[0],outputs_test[1],outputs_test[2],
                                  outputs_test[3],outputs_test[4],outputs_test[5]],axis=1)
            torch_dataset = TensorDataset(torch.tensor(inputs_test, dtype=torch.float32), 
                                          torch.tensor(outputs_test, dtype=torch.float32))
            dataloader_test = DataLoader(torch_dataset, batch_size=eval_batch_size, shuffle=False, pin_memory=True)
        else:
            dataloader_test = process_dataloader(inputs = inputs_test,
                                                 outputs = outputs_test,
                                                 shuffle=False,
                                                 batch_size=eval_batch_size,
                                                 )
        
        with torch.no_grad():
            for batch in dataloader_test:
                data,target = batch
                data_tau = data[:,322:]
                data = data[:,0:322]
                data_tau = data_tau.to(self.device)
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.processingInput(data)
                output = self.resNetLayer1(output)
                output = self.resNetLayer2(output)
                output_mid = self.ltauNoAdd(output,data_tau)
                output = output + output_mid
                a_or_pred = self.decoderLayer1(output)
                a_ex_pred = self.decoderLayer2(output)
                p_or_pred = self.decoderLayer3(output)
                p_ex_pred = self.decoderLayer4(output)
                v_or_pred = self.decoderLayer5(output)
                v_ex_pred = self.decoderLayer6(output)
                output = torch.cat((a_or_pred,a_ex_pred,p_or_pred,p_ex_pred,v_or_pred,v_ex_pred),dim=1)
                data_inversed,output_inversed,data_tau = self.unscalingLayer(data,output,data_tau)
                _,target_inversed,_ = self.unscalingLayer(data,target,data_tau)
                
                # Apply constraints to specific output variables
                a_or_final, a_ex_final, v_or_final, v_ex_final = self.apply_constraints(
                    output_inversed[:, 0:186],    # a_or
                    output_inversed[:, 186:372],  # a_ex  
                    output_inversed[:, 744:930],  # v_or
                    output_inversed[:, 930:1116], # v_ex
                    data_tau
                )

                # Apply KKT constraint projection (if using obs)
                if hasattr(self, 'obs') and self.obs is not None:
                    # Calculate the KKT constraint coefficients (if they have not been calculated yet)
                    if not hasattr(self, 'P6A_star') or self.P6A_star is None:
                        self.P6A_star, self.P6B_star, self.P6b_star = KKTP6ABb(self.obs, self.device)
                    
                    # Apply P6 constraint: Power balance constraint
                    P6X = KKTP6Xp(self.obs, self.device, data_inversed)
                    p_tmp = output_inversed[:, 372:744].T
                    p_final = (self.P6A_star @ P6X + self.P6B_star @ p_tmp + self.P6b_star).T
                else:
                    # If the KKT constraints are not used, the original power prediction can be directly employed.
                    p_final = output_inversed[:, 372:744]
                
                # Recombine and output
                output_inversed = torch.cat((a_or_final, a_ex_final, p_final, v_or_final, v_ex_final), dim=1)
                
                if self.device == torch.device('cpu'):
                    outputs.append(output_inversed.numpy())
                    targets.append(target_inversed.numpy())
                else:
                    outputs.append(output_inversed.cpu().numpy())
                    targets.append(target_inversed.cpu().numpy())
        outputs = np.concatenate(outputs)
        outputs = dataset.reconstruct_output(outputs)
        targets = np.concatenate(targets)
        targets = dataset.reconstruct_output(targets)

        return outputs,targets
                
    def count_parameters(self):
        """Count the total number of parameters in the model (in thousands)"""
        # Get all model components
        components = [
            ('ProcessingInput', self.processingInput),
            ('ResNetLayer1', self.resNetLayer1),
            ('ResNetLayer2', self.resNetLayer2),
            ('LtauNoAdd', self.ltauNoAdd),
            ('DecoderLayer1', self.decoderLayer1),
            ('DecoderLayer2', self.decoderLayer2),
            ('DecoderLayer3', self.decoderLayer3),
            ('DecoderLayer4', self.decoderLayer4),
            ('DecoderLayer5', self.decoderLayer5),
            ('DecoderLayer6', self.decoderLayer6),
            ('UnscalingLayer', self.unscalingLayer)
        ]
        
        # Count total parameters
        total_params = 0
        for name, component in components:
            if hasattr(component, 'parameters'):
                component_params = sum(p.numel() for p in component.parameters())
                total_params += component_params
                print(f"{name}: {component_params:,} parameters")
            else:
                print(f"{name}: No parameters (not initialized)")
        
        print(f"Total NN Parameters: {total_params:,} ({total_params/1e3:.1f}K)")
        
        return total_params

    def visualize_convergence(self, figsize=(15,5), save_path: str=None):
        """Visualizing the convergence of the model"""
        if len(self.train_losses) == 0:
            raise RuntimeError("The model should be trained before visualizing the convergence")
        
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        ax[0].set_title("Training and Validation Loss")
        ax[0].plot(self.train_losses, label='Train Loss')
        if len(self.val_losses) > 0:
            ax[0].plot(self.val_losses, label='Validation Loss')
        ax[0].grid()
        ax[0].legend()

        ax[1].set_title("MSE Loss")
        ax[1].plot(self.mse_list, label='MSE')
        # Add horizontal line at minimum MSE
        min_mse = min(self.mse_list)
        min_mse_idx = self.mse_list.index(min_mse)
        ax[1].axhline(y=min_mse, color='r', linestyle='--', alpha=0.5)
        # Add text annotation for minimum MSE value
        ax[1].text(len(self.mse_list) * 0.02, min_mse,
                f'Min MSE: {min_mse:.4f}', 
                verticalalignment='bottom')
        ax[1].grid()
        ax[1].legend()
        
        plt.tight_layout()
        if save_path is not None:
            # Make sure that the directory of the saving path exists.
            save_dir = pathlib.Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
