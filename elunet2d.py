import time
import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def rescale_normalize(x):
    '''
    Rescale the image between min and max
    
    Parameters:
    - x (tensor): torch tensor. 
    
    Returns:
    torch tensor: Normalized tensor
    '''
    
    return (x - torch.min(x))/(torch.max(x)-torch.min(x))


def stack_inputs_from_vars_noisy(input_filenames, input_path, percentage=5, random_seed_number=1, rescale=False):
    
    '''
    Stack 2D tensors in the 0 dimension and adds desired amount of Gaussian noise.
    
    Parameters:
    - input_filenames (list): List of filenames to 2D files to be read.
    - input_path (str): Path to the directory that contains files to be read.
    - percentage (float): Percentage (between 0 and 100) of Gaussian noise to add to each data channel.
    - random_seed_number (int): Seed number for random noise generation.
    - rescale (bool): Set to true if each channel needs to be normalized (Defaulted to False).
    
    Returns:
    torch tensor: An l-by-n-by-m tensor where l is the number of data channels, and n and m are dimensions of each channel
    '''
    
    np.random.seed(random_seed_number)
    percentage = percentage/100
    # Reads files and stacks them in the first dimension
    input_torch = []
    for i in range(len(input_filenames)):
        input_filename = os.path.join(input_path, input_filenames[i])
        arr = torch.tensor(np.loadtxt(input_filename, delimiter=','))
        noise = np.random.normal(
            loc=0, scale=arr.std(), size=arr.shape)*percentage
        arr += noise
        input_torch.append(arr)

    variable_stack = torch.stack(
        tuple(input_torch[i] for i in range(len(input_torch))), axis=0)
    if rescale == True:
        for i in range(len(input_torch)):
            variable_stack[i, :, :] = rescale_normalize(
                variable_stack[i, :, :])
    return variable_stack


def stack_inputs_from_vars(input_filenames, input_path, rescale=False):
    '''
    Stack 2D tensors in the 0 dimension.
    
    Parameters:
    - input_filenames (list): List of filenames to 2D files to be read.
    - input_path (str): Path to the directory that contains files to be read.
    - rescale (bool): Set to true if each channel needs to be normalized (Defaulted to False).
    
    Returns:
    torch tensor: An l-by-n-by-m tensor where l is the number of data channels, and n and m are dimensions of each channel.
    '''
    
    # Reads files and stacks them in the first dimension
    input_torch = []
    for i in range(len(input_filenames)):
        input_filename = os.path.join(input_path, input_filenames[i])
        input_torch.append(torch.tensor(
            np.loadtxt(input_filename, delimiter=',')))

    variable_stack = torch.stack(
        tuple(input_torch[i] for i in range(len(input_torch))), axis=0)
    if rescale == True:
        for i in range(len(input_torch)):
            variable_stack[i, :, :] = rescale_normalize(
                variable_stack[i, :, :])
    return variable_stack


class DoubleConv(nn.Module):
    # class that contains the double convolution at each step of the UNet
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=False),
            # Because we are using batch normalization, bias might not be necessary
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False),
            # Because we are using batch normalization, bias might not be necessary
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    # class that contains the upsampling at each step
    def __init__(self, feature, scale_factor):
        super(UpSample, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(feature*2, feature, 2, 1, padding=1, bias=False)
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    '''Customized UNET class
        credits to Aladdin Persson's tutorial on UNet from scratch in PyTorch:
        https://youtu.be/IHq1t7NxS8k?si=K3JwDYH0LHY0S7z4
    '''
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], convtrans_upsampling=False):
        super(UNET, self).__init__()
        # The convolutions lists should be stored in the ModuleList
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        # Pooling layer downsamples the image by half
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            # As we start the up path, we need to upsample each image
            # and also divide the number of channels by half so that we can concat the output with the skip_connections later
            if convtrans_upsampling:
                self.ups.append(nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2))
            else:
                self.ups.append(UpSample(feature=feature, scale_factor=2))
            # the result of the concatenation of previous output with the skip connections (hence the input size of feature*2) goes throught the DoubleConv filters
            self.ups.append(DoubleConv(feature*2, feature))

        # At the bottleneck level the convolution filters are double the last item in features
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # the last layer simply brings the multi-channel space back to the image space with a conv kerenel of 1
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # a forward pass of the UNet
        skip_connections = []
        for down in self.downs:
            x = down(x)
            # store the result of each doubleconv from the down path in skip_connections
            skip_connections.append(x)
            x = self.pool(x)
        # When we get down, run the bottleneck block
        x = self.bottleneck(x)

        # For ease of use, let's reverse the order in skip_connections for the path up
        skip_connections = skip_connections[::-1]

        # we march in range(len(self.ups)) by steps of 2 since we have instances of up_conv and double_conv stored there
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)

            skip_connection = skip_connections[idx//2]

            # in case the upsampling from ConvTranspose2d results in a shape not consistent with the skip_connection (for input images that are not divisible by 16)
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            # concatenate the upsampled output with the skip_connection from the down_path
            # order of channels are batch, channel, height, width --> so the channel is dim = 1
            concat_skip = torch.cat((skip_connection, x), dim=1)
            # then run the double_conv layer
            x = self.ups[idx+1](concat_skip)
        return self.final_conv(x)


def compute_loss(y, parameter_and_stress_out, criterion,
                 exx_gt, eyy_gt, exy_gt,
                 sxx_bound_gt, syy_bound_gt):
    '''
    Compute loss associated with network output.
    
    Parameters:
    - y (tensor): Network output tensor.
    - parameter_and_stress_out (bool): Set to True if the network outputs material parameters
    and stress and False if it outputs only material parameters.
    - criterion (torch.nn.MSELoss): In default mode, this is a MSELoss object.
    - exx_gt (tensor): Ground truth 2D xx strain distribution.
    - eyy_gt (tensor): Ground truth 2D yy strain distribution.
    - exy_gt (tensor): Ground truth 2D xy strain distribution.
    - sxx_bound_gt (tensor): xx stress values at the lateral boundaries.
    - syy_bound_gt (tensor): yy stress values at the axial boundaries.
    
    Returns:
    tensor: Loss value.
    '''
    zeros_tensor = torch.zeros_like(exx_gt).to(device)
    if parameter_and_stress_out:
        sxx_out = y[0, 2, :, :]
        syy_out = y[0, 3, :, :]
        sxy_out = y[0, 4, :, :]
        lame1_out = y[0, 0, :, :]
        lame2_out = y[0, 1, :, :]
        c11_out = 2*lame2_out + lame1_out
        c12_out = lame1_out
        c33_out = 2*lame2_out
        loss_c1 = criterion(c11_out*exx_gt +
                            c12_out*eyy_gt, sxx_out)
        loss_c2 = criterion(c11_out*eyy_gt +
                            c12_out*exx_gt, syy_out)
        loss_c3 = criterion(c33_out*exy_gt, sxy_out)
    else:
        lame1_out = y[0, 0, :, :]
        lame2_out = y[0, 1, :, :]
        c11_out = 2*lame2_out + lame1_out
        c12_out = lame1_out
        c33_out = 2*lame2_out
        sxx_out = c11_out*exx_gt + c12_out*eyy_gt
        syy_out = c11_out*eyy_gt + c12_out*exx_gt
        sxy_out = c33_out*exy_gt

    sxx_bound_out = torch.stack((sxx_out[:, 0], sxx_out[:, -1]), dim=0)
    syy_bound_out = torch.stack((syy_out[0, :], syy_out[-1, :]), dim=0)

    loss_bound_sxx = criterion(sxx_bound_out, sxx_bound_gt)
    loss_bound_syy = criterion(syy_bound_out, syy_bound_gt)
    spacing = 1/(((zeros_tensor.shape[0]-1)+(zeros_tensor.shape[1]-1))/2)

    _, sxx_x_out = torch.gradient(sxx_out, spacing=spacing)
    syy_y_out, _ = torch.gradient(syy_out, spacing=spacing)
    sxy_y_out, sxy_x_out = torch.gradient(sxy_out, spacing=spacing)

    loss_equib_x = criterion(sxx_x_out+sxy_y_out, zeros_tensor)
    loss_equib_y = criterion(sxy_x_out+syy_y_out, zeros_tensor)

    if parameter_and_stress_out:
        loss = loss_c1 + loss_c2 + loss_c3 + loss_bound_sxx + \
            loss_bound_syy + loss_equib_x + loss_equib_y
    else:
        loss = loss_bound_sxx + loss_bound_syy + loss_equib_x + loss_equib_y
    return loss


def compute_loss_weighted_paramstress(y, criterion,
                                      exx_gt, eyy_gt, exy_gt, sxx_bound_gt, syy_bound_gt,
                                      loss_weight_constit, loss_weight_sides, loss_weight_tb, loss_weight_res):
    '''
    Compute loss associated with network output in the self-adaptive loss weighting scenario.
    
    Parameters:
    - y (tensor): Network output tensor.
    - criterion (torch.nn.MSELoss): In default mode, this is a MSELoss object.
    - exx_gt (tensor): Ground truth 2D xx strain distribution.
    - eyy_gt (tensor): Ground truth 2D yy strain distribution.
    - exy_gt (tensor): Ground truth 2D xy strain distribution.
    - sxx_bound_gt (tensor): xx stress values at the lateral boundaries.
    - syy_bound_gt (tensor): yy stress values at the axial boundaries.
    - loss_weight_constit (tensor): Adaptive spatial weights for the constitutive equations term.
    - loss_weight_sides (tensor): Adaptive spatial weights on the lateral boundaries.
    - loss_weight_tb (tensor): Adaptive spatial weights on the top and bottom boundaries.
    - loss_weight_res (tensor): Adaptive spatial weights for the static equilibrium term.
    
    Returns:
    tensor: Loss value.
    '''
    zeros_tensor = torch.zeros_like(exx_gt).to(device)

    sxx_out = y[0, 2, :, :]
    syy_out = y[0, 3, :, :]
    sxy_out = y[0, 4, :, :]
    lame1_out = y[0, 0, :, :]
    lame2_out = y[0, 1, :, :]
    c11_out = 2*lame2_out + lame1_out
    c12_out = lame1_out
    c33_out = 2*lame2_out
    loss_c1 = criterion(loss_weight_constit*(c11_out*exx_gt +
                        c12_out*eyy_gt), loss_weight_constit*sxx_out)
    loss_c2 = criterion(loss_weight_constit*(c11_out*eyy_gt +
                        c12_out*exx_gt), loss_weight_constit*syy_out)
    loss_c3 = criterion(loss_weight_constit*(c33_out*exy_gt),
                        loss_weight_constit*sxy_out)

    sxx_bound_out = loss_weight_sides * \
        torch.stack((sxx_out[:, 0], sxx_out[:, -1]), dim=0)
    syy_bound_out = loss_weight_tb * \
        torch.stack((syy_out[0, :], syy_out[-1, :]), dim=0)

    sxx_bound_gt = loss_weight_sides*sxx_bound_gt
    syy_bound_gt = loss_weight_tb*syy_bound_gt

    loss_bound_sxx = criterion(sxx_bound_out, sxx_bound_gt)
    loss_bound_syy = criterion(syy_bound_out, syy_bound_gt)

    spacing = 1/(((zeros_tensor.shape[0]-1)+(zeros_tensor.shape[1]-1))/2)

    _, sxx_x_out = torch.gradient(sxx_out, spacing=spacing)
    syy_y_out, _ = torch.gradient(syy_out, spacing=spacing)
    sxy_y_out, sxy_x_out = torch.gradient(sxy_out, spacing=spacing)

    sxx_x_out = loss_weight_res*sxx_x_out
    syy_y_out = loss_weight_res*syy_y_out
    sxy_y_out = loss_weight_res*sxy_y_out
    sxy_x_out = loss_weight_res*sxy_x_out

    loss_equib_x = criterion(sxx_x_out+sxy_y_out, zeros_tensor)
    loss_equib_y = criterion(sxy_x_out+syy_y_out, zeros_tensor)

    loss = loss_c1 + loss_c2 + loss_c3 + loss_bound_sxx + \
        loss_bound_syy + loss_equib_x + loss_equib_y

    return loss


def train_running_time_weighted_loss(model, strain_stack, strain_stack_normalized, sxx_bound_gt, syy_bound_gt,
                                     youngs_gt, poissons_gt, sigma_0, criterion, optimizer, scheduler,
                                     loss_weight_constit, loss_weight_sides, loss_weight_tb, loss_weight_res,
                                     training_duration_max=60,
                                     loss_report_freq=500, early_stopping_threshold=1e-8):
    
    '''
    Run training in a given amount of time using the self-adaptive spatial weighting approach.
    
    Parameters:
    - model (UNet object): An instance of the UNet class.
    - strain_stack (tensor): Strain tensor where 
    channels are stacked in the order of xx, yy, and xy.
    - strain_stack_normalized (tensor): Normalized strain tensor where 
    channels are stacked in the order of xx, yy, and xy.
    - sxx_bound_gt (tensor): xx stress values at the lateral boundaries.
    - syy_bound_gt (tensor): yy stress values at the axial boundaries.
    - youngs_gt (tensor): Young's modulus ground truth distribution provided for real-time accuracy report.
    - poissons_gt (tensor): Poisson's ratio ground truth distribution provided for real-time accuracy report.
    - sigma_0 (float): reference characteristic stress value for dimensionless implementation.
    - criterion (torch.nn.MSELoss): In default mode, this is a MSELoss object.
    - optimizer: Pytorch optimizer.
    - scheduler: Pytorch scheduler for learning rate decay control.
    - loss_weight_constit (tensor): Adaptive spatial weights for the constitutive equations term.
    - loss_weight_sides (tensor): Adaptive spatial weights on the lateral boundaries.
    - loss_weight_tb (tensor): Adaptive spatial weights on the top and bottom boundaries.
    - loss_weight_res (tensor): Adaptive spatial weights for the static equilibrium term.
    - training_duration_max (float): Training time in minutes.
    - loss_report_freq (int): Frequency of loss reporting.
    - early_stopping_threshold (float): Threshold for early stopping of training.
    
    Returns:
    model (UNet object): Instance of the UNet class after training.
    loss_histories (dict): Dictionary containing various loss and accuracy values across training.
    '''
    
    running_loss_history_weighted = []
    running_loss_history = []
    youngs_mae_history = []
    poissons_mae_history = []
    mem_allec_history = []

    training_duration = 0
    e = 0
    start = time.time()

    exx_gt = strain_stack[0, 0, :, :]
    eyy_gt = strain_stack[0, 1, :, :]
    exy_gt = strain_stack[0, 2, :, :]

    while training_duration < training_duration_max:

        y = model.forward(strain_stack_normalized)

        loss = compute_loss_weighted_paramstress(y, criterion,
                                                 exx_gt, eyy_gt, exy_gt,
                                                 sxx_bound_gt, syy_bound_gt,
                                                 loss_weight_constit, loss_weight_sides, loss_weight_tb, loss_weight_res)
        loss_unweighted = compute_loss(y, True, criterion,
                                       exx_gt, eyy_gt, exy_gt,
                                       sxx_bound_gt, syy_bound_gt)

        with torch.no_grad():
            lame1_pred_dim = sigma_0*y[0, 0, :, :]
            lame2_pred_dim = sigma_0*y[0, 1, :, :]
            Youngs_pred = lame2_pred_dim * \
                (3*lame1_pred_dim + 2*lame2_pred_dim) / \
                (lame1_pred_dim+lame2_pred_dim)
            Poissons_pred = lame1_pred_dim/(2*(lame1_pred_dim+lame2_pred_dim))
            Youngs_pred = Youngs_pred/(1-Poissons_pred**2)
            Poissons_pred = Poissons_pred/(1-Poissons_pred)

            MAE_Youngs = 100 * \
                torch.abs(Youngs_pred-youngs_gt)/torch.abs(youngs_gt)
            MAE_Youngs = torch.mean(MAE_Youngs)

            MAE_Poissons = 100 * \
                torch.abs(Poissons_pred-poissons_gt)/torch.abs(poissons_gt)
            MAE_Poissons = torch.mean(MAE_Poissons)

        mem_allec = torch.cuda.memory_allocated(0)/1024/1024/1024

        if e % loss_report_freq == 0:
            print(
                f"epoch # {e}, loss: {loss.item():e}, elapsedtime: {training_duration:.2f} min, memory: {mem_allec:.2f} GB")
            print('Youngs_MAE: ', "{:.2f}".format(
                MAE_Youngs.item()), 'Poissons_MAE: ', "{:.2f}".format(MAE_Poissons.item()))
        e += 1

        running_loss_history_weighted.append(loss.item())
        running_loss_history.append(loss_unweighted.item())
        youngs_mae_history.append(MAE_Youngs.item())
        poissons_mae_history.append(MAE_Poissons.item())
        mem_allec_history.append(mem_allec)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        end = time.time()
        training_duration = (end - start)/60

        if loss < early_stopping_threshold:
            print('Early stopping. Threshold reached.')
            loss_histories = {}
            loss_histories['running_loss_history_weighted'] = running_loss_history_weighted
            loss_histories['running_loss_history'] = running_loss_history
            loss_histories['youngs_mae_history'] = youngs_mae_history
            loss_histories['poissons_mae_history'] = poissons_mae_history
            print('Training took {:.2f} minutes'.format(training_duration))
            return model, loss_histories
    loss_histories = {}
    loss_histories['running_loss_history_weighted'] = running_loss_history_weighted
    loss_histories['running_loss_history'] = running_loss_history
    loss_histories['youngs_mae_history'] = youngs_mae_history
    loss_histories['poissons_mae_history'] = poissons_mae_history
    print('Training took {:.2f} minutes'.format(training_duration))
    return model, loss_histories


def train_running_time(model, strain_stack, strain_stack_normalized, sxx_bound_gt, syy_bound_gt,
                       youngs_gt, poissons_gt, sigma_0, criterion, optimizer, scheduler,
                       parameter_and_stress_out=True, training_duration_max=60,
                       loss_report_freq=500, early_stopping_threshold=1e-8):
    
    '''
    Run training in a given amount of time.
    
    Parameters:
    - model (UNet object): An instance of the UNet class.
    - strain_stack (tensor): Strain tensor where 
    channels are stacked in the order of xx, yy, and xy.
    - strain_stack_normalized (tensor): Normalized strain tensor where 
    channels are stacked in the order of xx, yy, and xy.
    - sxx_bound_gt (tensor): xx stress values at the lateral boundaries.
    - syy_bound_gt (tensor): yy stress values at the axial boundaries.
    - youngs_gt (tensor): Young's modulus ground truth distribution provided for real-time accuracy report.
    - poissons_gt (tensor): Poisson's ratio ground truth distribution provided for real-time accuracy report.
    - sigma_0 (float): reference characteristic stress value for dimensionless implementation.
    - criterion (torch.nn.MSELoss): In default mode, this is a MSELoss object.
    - optimizer: Pytorch optimizer.
    - scheduler: Pytorch scheduler for learning rate decay control.
    - parameter_and_stress_out (bool): Set to True if the network outputs material parameters
    and stress and False if it outputs only material parameters.
    - training_duration_max (float): Training time in minutes.
    - loss_report_freq (int): Frequency of loss reporting.
    - early_stopping_threshold (float): Threshold for early stopping of training.
    
    Returns:
    model (UNet object): Instance of the UNet class after training.
    loss_histories (dict): Dictionary containing various loss and accuracy values across training.
    '''
    
    running_loss_history = []
    youngs_mae_history = []
    poissons_mae_history = []
    mem_allec_history = []

    training_duration = 0
    e = 0
    start = time.time()

    exx_gt = strain_stack[0, 0, :, :]
    eyy_gt = strain_stack[0, 1, :, :]
    exy_gt = strain_stack[0, 2, :, :]

    while training_duration < training_duration_max:

        y = model.forward(strain_stack_normalized)

        loss = compute_loss(y, parameter_and_stress_out, criterion,
                            exx_gt, eyy_gt, exy_gt,
                            sxx_bound_gt, syy_bound_gt)

        with torch.no_grad():
            lame1_pred_dim = sigma_0*y[0, 0, :, :]
            lame2_pred_dim = sigma_0*y[0, 1, :, :]
            Youngs_pred = lame2_pred_dim * \
                (3*lame1_pred_dim + 2*lame2_pred_dim) / \
                (lame1_pred_dim+lame2_pred_dim)
            Poissons_pred = lame1_pred_dim/(2*(lame1_pred_dim+lame2_pred_dim))
            Youngs_pred = Youngs_pred/(1-Poissons_pred**2)
            Poissons_pred = Poissons_pred/(1-Poissons_pred)

            MAE_Youngs = 100 * \
                torch.abs(Youngs_pred-youngs_gt)/torch.abs(youngs_gt)
            MAE_Youngs = torch.mean(MAE_Youngs)

            MAE_Poissons = 100 * \
                torch.abs(Poissons_pred-poissons_gt)/torch.abs(poissons_gt)
            MAE_Poissons = torch.mean(MAE_Poissons)

        mem_allec = torch.cuda.memory_allocated(0)/1024/1024/1024

        if e % loss_report_freq == 0:
            print(
                f"epoch # {e}, loss: {loss.item():e}, elapsedtime: {training_duration:.2f} min, memory: {mem_allec:.2f} GB")
            print('Youngs_MAE: ', "{:.2f}".format(
                MAE_Youngs.item()), 'Poissons_MAE: ', "{:.2f}".format(MAE_Poissons.item()))
        e += 1

        running_loss_history.append(loss.item())
        youngs_mae_history.append(MAE_Youngs.item())
        poissons_mae_history.append(MAE_Poissons.item())
        mem_allec_history.append(mem_allec)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        end = time.time()
        training_duration = (end - start)/60

        if loss < early_stopping_threshold:
            print('Early stopping. Threshold reached.')
            loss_histories = {}
            loss_histories['running_loss_history'] = running_loss_history
            loss_histories['youngs_mae_history'] = youngs_mae_history
            loss_histories['poissons_mae_history'] = poissons_mae_history
            print('Training took {:.2f} minutes'.format(training_duration))
            return model, loss_histories
    loss_histories = {}
    loss_histories['running_loss_history'] = running_loss_history
    loss_histories['youngs_mae_history'] = youngs_mae_history
    loss_histories['poissons_mae_history'] = poissons_mae_history
    print('Training took {:.2f} minutes'.format(training_duration))
    return model, loss_histories


def train_epoch_number_weighted_loss(model, strain_stack, strain_stack_normalized, sxx_bound_gt, syy_bound_gt,
                                     youngs_gt, poissons_gt, sigma_0, criterion, optimizer, scheduler,
                                     loss_weight_constit, loss_weight_sides, loss_weight_tb, loss_weight_res,
                                     max_epochs=1000,
                                     loss_report_freq=500, early_stopping_threshold=1e-8):
    '''
    Run training for a given number of epochs using the self-adaptive spatial weighting approach.
    
    Parameters:
    - model (UNet object): An instance of the UNet class.
    - strain_stack (tensor): Strain tensor where 
    channels are stacked in the order of xx, yy, and xy.
    - strain_stack_normalized (tensor): Normalized strain tensor where 
    channels are stacked in the order of xx, yy, and xy.
    - sxx_bound_gt (tensor): xx stress values at the lateral boundaries.
    - syy_bound_gt (tensor): yy stress values at the axial boundaries.
    - youngs_gt (tensor): Young's modulus ground truth distribution provided for real-time accuracy report.
    - poissons_gt (tensor): Poisson's ratio ground truth distribution provided for real-time accuracy report.
    - sigma_0 (float): reference characteristic stress value for dimensionless implementation.
    - criterion (torch.nn.MSELoss): In default mode, this is a MSELoss object.
    - optimizer: Pytorch optimizer.
    - scheduler: Pytorch scheduler for learning rate decay control.
    - loss_weight_constit (tensor): Adaptive spatial weights for the constitutive equations term.
    - loss_weight_sides (tensor): Adaptive spatial weights on the lateral boundaries.
    - loss_weight_tb (tensor): Adaptive spatial weights on the top and bottom boundaries.
    - loss_weight_res (tensor): Adaptive spatial weights for the static equilibrium term.
    - max_epochs (int): Number of training epochs.
    - loss_report_freq (int): Frequency of loss reporting.
    - early_stopping_threshold (float): Threshold for early stopping of training.
    
    Returns:
    model (UNet object): Instance of the UNet class after training.
    loss_histories (dict): Dictionary containing various loss and accuracy values across training.
    '''    
    
    running_loss_history_weighted = []
    running_loss_history = []
    youngs_mae_history = []
    poissons_mae_history = []
    mem_allec_history = []

    e = 0
    training_duration = 0
    start = time.time()

    exx_gt = strain_stack[0, 0, :, :]
    eyy_gt = strain_stack[0, 1, :, :]
    exy_gt = strain_stack[0, 2, :, :]

    for e in range(max_epochs):

        y = model.forward(strain_stack_normalized)

        loss = compute_loss_weighted_paramstress(y, criterion,
                                                 exx_gt, eyy_gt, exy_gt,
                                                 sxx_bound_gt, syy_bound_gt,
                                                 loss_weight_constit, loss_weight_sides, loss_weight_tb, loss_weight_res)
        loss_unweighted = compute_loss(y, True, criterion,
                                       exx_gt, eyy_gt, exy_gt,
                                       sxx_bound_gt, syy_bound_gt)

        with torch.no_grad():
            lame1_pred_dim = sigma_0*y[0, 0, :, :]
            lame2_pred_dim = sigma_0*y[0, 1, :, :]
            Youngs_pred = lame2_pred_dim *\
                (3*lame1_pred_dim + 2*lame2_pred_dim) /\
                (lame1_pred_dim+lame2_pred_dim)
            Poissons_pred = lame1_pred_dim/(2*(lame1_pred_dim+lame2_pred_dim))
            Youngs_pred = Youngs_pred/(1-Poissons_pred**2)
            Poissons_pred = Poissons_pred/(1-Poissons_pred)

            MAE_Youngs = 100 * \
                torch.abs(Youngs_pred-youngs_gt)/torch.abs(youngs_gt)
            MAE_Youngs = torch.mean(MAE_Youngs)

            MAE_Poissons = 100 * \
                torch.abs(Poissons_pred-poissons_gt)/torch.abs(poissons_gt)
            MAE_Poissons = torch.mean(MAE_Poissons)

        mem_allec = torch.cuda.memory_allocated(0)/1024/1024/1024

        if e % loss_report_freq == 0:
            print(
                f"epoch # {e}, loss: {loss.item():e}, elapsedtime: {training_duration:.2f} min, memory: {mem_allec:.2f} GB")
            print('Youngs_MAE: ', "{:.2f}".format(
                MAE_Youngs.item()), 'Poissons_MAE: ', "{:.2f}".format(MAE_Poissons.item()))
        running_loss_history_weighted.append(loss.item())
        running_loss_history.append(loss_unweighted.item())
        youngs_mae_history.append(MAE_Youngs.item())
        poissons_mae_history.append(MAE_Poissons.item())
        mem_allec_history.append(mem_allec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        end = time.time()
        training_duration = (end-start)/60
        if loss < early_stopping_threshold:
            print('Early stopping. Threshold reached.')
            print('Training took {:.2f} minutes'.format(training_duration))
            loss_histories = {}
            loss_histories['running_loss_history_weighted'] = running_loss_history_weighted
            loss_histories['running_loss_history'] = running_loss_history
            loss_histories['youngs_mae_history'] = youngs_mae_history
            loss_histories['poissons_mae_history'] = poissons_mae_history
            return model, loss_histories
    loss_histories = {}
    loss_histories['running_loss_history_weighted'] = running_loss_history_weighted
    loss_histories['running_loss_history'] = running_loss_history
    loss_histories['youngs_mae_history'] = youngs_mae_history
    loss_histories['poissons_mae_history'] = poissons_mae_history
    print('Training took {:.2f} minutes'.format(training_duration))
    return model, loss_histories


def train_epoch_number(model, strain_stack, strain_stack_normalized, sxx_bound_gt, syy_bound_gt,
                       youngs_gt, poissons_gt, sigma_0, criterion, optimizer, scheduler,
                       parameter_and_stress_out=True, max_epochs=1000,
                       loss_report_freq=500, early_stopping_threshold=1e-8):
    '''
    Run training for a given number of epochs.
    
    Parameters:
    - model (UNet object): An instance of the UNet class.
    - strain_stack (tensor): Strain tensor where 
    channels are stacked in the order of xx, yy, and xy.
    - strain_stack_normalized (tensor): Normalized strain tensor where 
    channels are stacked in the order of xx, yy, and xy.
    - sxx_bound_gt (tensor): xx stress values at the lateral boundaries.
    - syy_bound_gt (tensor): yy stress values at the axial boundaries.
    - youngs_gt (tensor): Young's modulus ground truth distribution provided for real-time accuracy report.
    - poissons_gt (tensor): Poisson's ratio ground truth distribution provided for real-time accuracy report.
    - sigma_0 (float): reference characteristic stress value for dimensionless implementation.
    - criterion (torch.nn.MSELoss): In default mode, this is a MSELoss object.
    - optimizer: Pytorch optimizer.
    - scheduler: Pytorch scheduler for learning rate decay control.
    - parameter_and_stress_out (bool): Set to True if the network outputs material parameters
    and stress and False if it outputs only material parameters.
    - max_epochs (int): Number of training epochs.
    - loss_report_freq (int): Frequency of loss reporting.
    - early_stopping_threshold (float): Threshold for early stopping of training.
    
    Returns:
    model (UNet object): Instance of the UNet class after training.
    loss_histories (dict): Dictionary containing various loss and accuracy values across training.
    '''
    
    running_loss_history = []
    youngs_mae_history = []
    poissons_mae_history = []
    mem_allec_history = []

    e = 0
    training_duration = 0
    start = time.time()

    exx_gt = strain_stack[0, 0, :, :]
    eyy_gt = strain_stack[0, 1, :, :]
    exy_gt = strain_stack[0, 2, :, :]

    for e in range(max_epochs):

        y = model.forward(strain_stack_normalized)

        loss = compute_loss(y, parameter_and_stress_out, criterion,
                            exx_gt, eyy_gt, exy_gt,
                            sxx_bound_gt, syy_bound_gt)

        with torch.no_grad():
            lame1_pred_dim = sigma_0*y[0, 0, :, :]
            lame2_pred_dim = sigma_0*y[0, 1, :, :]
            Youngs_pred = lame2_pred_dim *\
                (3*lame1_pred_dim + 2*lame2_pred_dim) /\
                (lame1_pred_dim+lame2_pred_dim)
            Poissons_pred = lame1_pred_dim/(2*(lame1_pred_dim+lame2_pred_dim))
            Youngs_pred = Youngs_pred/(1-Poissons_pred**2)
            Poissons_pred = Poissons_pred/(1-Poissons_pred)

            MAE_Youngs = 100 * \
                torch.abs(Youngs_pred-youngs_gt)/torch.abs(youngs_gt)
            MAE_Youngs = torch.mean(MAE_Youngs)

            MAE_Poissons = 100 * \
                torch.abs(Poissons_pred-poissons_gt)/torch.abs(poissons_gt)
            MAE_Poissons = torch.mean(MAE_Poissons)

        mem_allec = torch.cuda.memory_allocated(0)/1024/1024/1024

        if e % loss_report_freq == 0:
            print(
                f"epoch # {e}, loss: {loss.item():e}, elapsedtime: {training_duration:.2f} min, memory: {mem_allec:.2f} GB")
            print('Youngs_MAE: ', "{:.2f}".format(
                MAE_Youngs.item()), 'Poissons_MAE: ', "{:.2f}".format(MAE_Poissons.item()))
        running_loss_history.append(loss.item())
        youngs_mae_history.append(MAE_Youngs.item())
        poissons_mae_history.append(MAE_Poissons.item())
        mem_allec_history.append(mem_allec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        end = time.time()
        training_duration = (end-start)/60
        if loss < early_stopping_threshold:
            print('Early stopping. Threshold reached.')
            print('Training took {:.2f} minutes'.format(training_duration))
            loss_histories = {}
            loss_histories['running_loss_history'] = running_loss_history
            loss_histories['youngs_mae_history'] = youngs_mae_history
            loss_histories['poissons_mae_history'] = poissons_mae_history
            return model, loss_histories
    loss_histories = {}
    loss_histories['running_loss_history'] = running_loss_history
    loss_histories['youngs_mae_history'] = youngs_mae_history
    loss_histories['poissons_mae_history'] = poissons_mae_history
    print('Training took {:.2f} minutes'.format(training_duration))
    return model, loss_histories


def save_checkpoint(model, optimizer, filename="my_checkpoint_UNet.pth.tar"):
    # Save model state
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    print("=> Saving checkpoint")
    torch.save(checkpoint, filename)


def gt_estimation_plotter(gt_img, estimated_img, vmin, vmax, cmap='jet'):
    ''' Plot the estimated and ground truth images side-by-side along with relative error 
    
    Parameters:
    - gt_img (tensor): Ground truth image.
    - estimated_img (tensor): Estimated image from the model.
    - vmin (float): minimum value for the colormap.
    - vmax (float): maximum value for the colormap.
    - cmap (str): colormap (matplotlib standards).
    
    Returns:
    None
 
    '''


    if isinstance(gt_img, torch.Tensor):
        gt_img = gt_img.detach().cpu().numpy()

    if isinstance(estimated_img, torch.Tensor):
        estimated_img = estimated_img.detach().cpu().numpy()

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(10, 10))
    ax1_sub = ax1.pcolor(gt_img, cmap=cmap, vmin=vmin,
                         vmax=vmax)
    ax1.set_aspect('equal', 'box')
    ax1.set_title('Ground truth $E$')
    ax2_sub = ax2.pcolor(estimated_img, cmap=cmap, vmin=vmin,
                         vmax=vmax)
    ax2.set_aspect('equal', 'box')
    ax2.set_title('Estimated $E$')

    ax3_sub = ax3.pcolor(100*np.abs(estimated_img-gt_img)/gt_img, cmap=cmap, vmin=0,
                         vmax=20)
    ax3.set_aspect('equal', 'box')
    ax3.set_title('Error')

    f.colorbar(ax1_sub, ax=ax1, fraction=0.046, pad=0.04)
    f.colorbar(ax2_sub, ax=ax2, fraction=0.046, pad=0.04)
    f.colorbar(ax3_sub, ax=ax3, fraction=0.046, pad=0.04)

    ax1.xaxis.set_visible(False)
    ax2.xaxis.set_visible(False)
    ax3.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax3.yaxis.set_visible(False)

    plt.show()


def weight_loss_dist(filename, loss_weight_sides, loss_weight_tb, loss_weight_res):

    '''
    Plot the spatial distribution of self-adaptive loss weights
    
    Parameters:
    - filename (str): Filename to save the file.
    - loss_weight_constit (tensor): Adaptive spatial weights for the constitutive equations term.
    - loss_weight_sides (tensor): Adaptive spatial weights on the lateral boundaries.
    - loss_weight_tb (tensor): Adaptive spatial weights on the top and bottom boundaries.
    - loss_weight_res (tensor): Adaptive spatial weights for the static equilibrium term.
    
    Returns:
    None
    '''
    p2d = (2, 2, 2, 2)
    weight_image = torch.nn.functional.pad(
        loss_weight_res, p2d, "constant", torch.nan)  # effectively zero padding

    weight_image[0, 2:-2] = loss_weight_tb[0, :]
    weight_image[-1, 2:-2] = loss_weight_tb[1, :]
    weight_image[2:-2, 0] = loss_weight_sides[0, :]
    weight_image[2:-2, -1] = loss_weight_sides[1, :]
    weight_image = weight_image.detach().cpu().numpy()

    f, ax1 = plt.subplots(1, 1)
    ax1_sub = ax1.pcolor(weight_image, cmap='jet')
    ax1.set_aspect('equal', 'box')
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    f.colorbar(ax1_sub, ax=ax1)
    
    plt.savefig(filename)
    plt.show()


def weight_loss_dist_single(filename, loss_weight):
    '''
    Plot the spatial distribution of one self-adaptive loss weight.
    
    Parameters:
    - filename (str): Filename to save the file.
    - loss_weight (tensor): Adaptive spatial weights.

    
    Returns:
    None
    '''
    loss_weight = loss_weight.detach().cpu().numpy()
    f, ax1 = plt.subplots(1, 1)
    ax1_sub = ax1.pcolor(loss_weight, cmap='jet')
    ax1.set_aspect('equal', 'box')
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    f.colorbar(ax1_sub, ax=ax1)
    
    plt.savefig(filename)
    plt.show()


def plot_sbs_all(filename, sxx_gt, syy_gt, sxy_gt, youngs_gt, poissons_gt,
                 sxx_pred, syy_pred, sxy_pred, youngs_pred, poissons_pred):

    '''
    Plot the spatial distribution of stress and material parameter ground truths and model estimations.
    
    Parameters:
    - filename (str): Filename to save the file.
    - sxx_gt (tensor): Ground truth xx stress distribution.
    - syy_gt (tensor): Ground truth yy stress distribution.
    - sxy_gt (tensor): Ground truth xy stress distribution.
    - youngs_gt (tensor): Ground truth Young's Modulus stress distribution.
    - poissons_gt (tensor): Ground truth Poisson's ratio distribution.
    - sxx_pred (tensor): Estimated xx stress distribution.
    - syy_pred (tensor): Estimated yy stress distribution.
    - sxy_pred (tensor): Estimated xy stress distribution.
    - youngs_pred (tensor): Estimated Young's Modulus stress distribution.
    - poissons_pred (tensor): Estimated Poisson's ratio distribution.

    Returns:
    None
    '''
    f, axes = plt.subplots(2, 5, sharey=True, figsize=(20, 10))
    gt_img = sxx_gt
    output_img = sxx_pred

    if isinstance(gt_img, torch.Tensor):
        gt_img = gt_img.detach().cpu().numpy()

    if isinstance(output_img, torch.Tensor):
        output_img = output_img.detach().cpu().numpy()

    ax1_sub = axes[0, 0].pcolor(gt_img, cmap='jet', vmin=np.min(gt_img) - 0.1*(np.max(gt_img)-np.min(gt_img)),
                                vmax=np.max(gt_img) + 0.1*(np.max(gt_img)-np.min(gt_img)))
    axes[0, 0].set_aspect('equal', 'box')
    # axes[0, 0].set_title('Ground Truth')
    axes[0, 0].yaxis.set_visible(False)
    axes[0, 0].xaxis.set_visible(False)

    ax2_sub = axes[1, 0].pcolor(output_img, cmap='jet', vmin=np.min(gt_img) - 0.1*(np.max(gt_img)-np.min(gt_img)),
                                vmax=np.max(gt_img) + 0.1*(np.max(gt_img)-np.min(gt_img)))
    axes[1, 0].set_aspect('equal', 'box')
    axes[1, 0].yaxis.set_visible(False)
    axes[1, 0].xaxis.set_visible(False)
    f.colorbar(ax1_sub, ax=axes[0, 0], fraction=0.046, pad=0.04)
    f.colorbar(ax2_sub, ax=axes[1, 0], fraction=0.046, pad=0.04)

    gt_img = syy_gt
    output_img = syy_pred
    if isinstance(gt_img, torch.Tensor):
        gt_img = gt_img.detach().cpu().numpy()

    if isinstance(output_img, torch.Tensor):
        output_img = output_img.detach().cpu().numpy()

    ax1_sub = axes[0, 1].pcolor(gt_img, cmap='jet', vmin=np.min(gt_img) - 0.1*(np.max(gt_img)-np.min(gt_img)),
                                vmax=np.max(gt_img) + 0.1*(np.max(gt_img)-np.min(gt_img)))
    axes[0, 1].set_aspect('equal', 'box')
    axes[0, 1].yaxis.set_visible(False)
    axes[0, 1].xaxis.set_visible(False)

    ax2_sub = axes[1, 1].pcolor(output_img, cmap='jet', vmin=np.min(gt_img) - 0.1*(np.max(gt_img)-np.min(gt_img)),
                                vmax=np.max(gt_img) + 0.1*(np.max(gt_img)-np.min(gt_img)))
    axes[1, 1].set_aspect('equal', 'box')
    axes[1, 1].yaxis.set_visible(False)
    axes[1, 1].xaxis.set_visible(False)
    f.colorbar(ax1_sub, ax=axes[0, 1], fraction=0.046, pad=0.04)
    f.colorbar(ax2_sub, ax=axes[1, 1], fraction=0.046, pad=0.04)

    gt_img = sxy_gt
    output_img = sxy_pred

    if isinstance(gt_img, torch.Tensor):
        gt_img = gt_img.detach().cpu().numpy()

    if isinstance(output_img, torch.Tensor):
        output_img = output_img.detach().cpu().numpy()

    ax1_sub = axes[0, 2].pcolor(gt_img, cmap='jet', vmin=np.min(gt_img) - 0.1*(np.max(gt_img)-np.min(gt_img)),
                                vmax=np.max(gt_img) + 0.1*(np.max(gt_img)-np.min(gt_img)))
    axes[0, 2].set_aspect('equal', 'box')
    axes[0, 2].yaxis.set_visible(False)
    axes[0, 2].xaxis.set_visible(False)

    ax2_sub = axes[1, 2].pcolor(output_img, cmap='jet', vmin=np.min(gt_img) - 0.1*(np.max(gt_img)-np.min(gt_img)),
                                vmax=np.max(gt_img) + 0.1*(np.max(gt_img)-np.min(gt_img)))
    axes[1, 2].set_aspect('equal', 'box')
    axes[1, 2].yaxis.set_visible(False)
    axes[1, 2].xaxis.set_visible(False)
    f.colorbar(ax1_sub, ax=axes[0, 2], fraction=0.046, pad=0.04)
    f.colorbar(ax2_sub, ax=axes[1, 2], fraction=0.046, pad=0.04)

    gt_img = poissons_gt
    output_img = poissons_pred

    if isinstance(gt_img, torch.Tensor):
        gt_img = gt_img.detach().cpu().numpy()

    if isinstance(output_img, torch.Tensor):
        output_img = output_img.detach().cpu().numpy()

    ax1_sub = axes[0, 3].pcolor(gt_img, cmap='jet', vmin=np.min(gt_img)-0.05,
                                vmax=np.max(gt_img)+0.05)
    axes[0, 3].set_aspect('equal', 'box')
    axes[0, 3].yaxis.set_visible(False)
    axes[0, 3].xaxis.set_visible(False)

    ax2_sub = axes[1, 3].pcolor(output_img, cmap='jet', vmin=np.min(gt_img)-0.05,
                                vmax=np.max(gt_img)+0.05)
    axes[1, 3].set_aspect('equal', 'box')
    axes[1, 3].yaxis.set_visible(False)
    axes[1, 3].xaxis.set_visible(False)
    f.colorbar(ax1_sub, ax=axes[0, 3], fraction=0.046, pad=0.04)
    f.colorbar(ax2_sub, ax=axes[1, 3], fraction=0.046, pad=0.04)

    gt_img = youngs_gt/1000
    output_img = youngs_pred/1000

    if isinstance(gt_img, torch.Tensor):
        gt_img = gt_img.detach().cpu().numpy()

    if isinstance(output_img, torch.Tensor):
        output_img = output_img.detach().cpu().numpy()

    ax1_sub = axes[0, 4].pcolor(gt_img, cmap='jet', vmin=np.min(gt_img)-0.2,
                                vmax=np.max(gt_img)+0.2)
    axes[0, 4].set_aspect('equal', 'box')
    axes[0, 4].yaxis.set_visible(False)
    axes[0, 4].xaxis.set_visible(False)

    ax2_sub = axes[1, 4].pcolor(output_img, cmap='jet', vmin=np.min(gt_img)-0.2,
                                vmax=np.max(gt_img)+0.2)
    axes[1, 4].set_aspect('equal', 'box')
    axes[1, 4].yaxis.set_visible(False)
    axes[1, 4].xaxis.set_visible(False)
    f.colorbar(ax1_sub, ax=axes[0, 4], fraction=0.046, pad=0.04)
    f.colorbar(ax2_sub, ax=axes[1, 4], fraction=0.046, pad=0.04)
    
    plt.savefig(filename)
    plt.show()
