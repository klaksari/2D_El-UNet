import argparse
import torch
import elunet2d as eu2d
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
import os
import scipy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(
    prog='ElastUNet_Brain_PlaneStress',
    description='Sample code for 2D spatial estimation of elasticity parameter')

parser.add_argument('-ps', '--parameter_and_stress',
                    action="store_true")
parser.add_argument('--unet_num_channels',
                    help='Num channels per unet depth (default=[64, 128, 256, 512], the bottleneck number of channels will be the last element of this multiplied by two)', type=int, nargs='+', default=[64, 128, 256, 512])
parser.add_argument(
    '-e', '--epochs', help='Maximum number of epochs (default 2)', type=int, nargs=1)
parser.add_argument('-wl', '--weighted_loss',
                    help='Assign spatial weights to boundary and residual loss terms for adverserial (min-max) optimization', action="store_true")
parser.add_argument('-lr', '--learning_rate',
                    help='Initial learning rate (default 0.001)', type=float, nargs=1, default=[0.001])
parser.add_argument('--lr_update_threshold', help='Patience argument for scheduler (default 200000)',
                    type=int, nargs=1, default=[200000])

parser.add_argument('--conv_upsampling',
                    help='Turn on convolution transpose upsampling method, otherwise bilinear upsampling is performed in the UNet up path', action="store_true")
parser.add_argument('--loss_report_freq', help='Frequency of loss report updates',
                    type=int, nargs=1, default=[1])
parser.add_argument(
    '-pl', '--plot', help='plot_results, default is false', action="store_true")
parser.add_argument('--training_time', help='Training time in minutes that the model is expected to run',
                    type=float, nargs=1)

parser.add_argument('-ip', '--input_path', help='Input domain data path (default ./input_domain_data)',
                    type=str, nargs=1, default=['input_domain_data'])
parser.add_argument('-of', '--output_path',
                    help='Output path (default res**)', type=str, nargs=1, default='output')

args = parser.parse_args()

# Read data

stress_stack = eu2d.stack_inputs_from_vars(
    ['Sxx_structured.txt', 'Syy_structured.txt', 'Sxy_structured.txt'], args.input_path[0], rescale=False)
strain_stack = eu2d.stack_inputs_from_vars(
    ['Exx_structured.txt', 'Eyy_structured.txt', 'Exy_structured.txt'], args.input_path[0], rescale=False)
strain_stack_normalized = eu2d.stack_inputs_from_vars(
    ['Exx_structured.txt', 'Eyy_structured.txt', 'Exy_structured.txt'], args.input_path[0], rescale=True)

stress_stack = stress_stack.float().unsqueeze(0).to(device)
strain_stack = strain_stack.float().unsqueeze(0).to(device)
strain_stack_normalized = strain_stack_normalized.unsqueeze(
    0).float().to(device)


sigma_0 = torch.max(stress_stack[0, 1, -1, :])

sxx_bound_gt = torch.stack(
    (stress_stack[0, 0, :, 0], stress_stack[0, 0, :, -1]), dim=0)/sigma_0
syy_bound_gt = torch.stack(
    (stress_stack[0, 1, 0, :], stress_stack[0, 1, -1, :]), dim=0)/sigma_0

youngs_gt = np.loadtxt(os.path.join(
    args.input_path[0], 'Youngs_Structured.txt'), delimiter=',')
poissons_gt = np.loadtxt(os.path.join(
    args.input_path[0], 'Poissons_Structured.txt'), delimiter=',')
youngs_gt = torch.tensor(youngs_gt).float().to(device)
poissons_gt = torch.tensor(poissons_gt).float().to(device)

# Construct the model
if args.parameter_and_stress:
    model = eu2d.UNET(in_channels=3, out_channels=5, features=args.unet_num_channels,
                      convtrans_upsampling=args.conv_upsampling)
else:
    model = eu2d.UNET(in_channels=3, out_channels=2, features=args.unet_num_channels,
                      convtrans_upsampling=args.conv_upsampling)
model = model.to(device)
model = model.float()
learning_rate = args.learning_rate[0]
criterion = nn.MSELoss()

if args.weighted_loss:
    loss_weight_constit = torch.ones_like(
        strain_stack[0, 0, :, :], requires_grad=True)
    loss_weight_sides = torch.ones_like(sxx_bound_gt, requires_grad=True)
    loss_weight_tb = torch.ones_like(syy_bound_gt, requires_grad=True)
    loss_weight_res = torch.ones_like(
        strain_stack[0, 0, :, :], requires_grad=True)

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': loss_weight_constit, 'maximize': True},
        {'params': loss_weight_sides, 'maximize': True},
        {'params': loss_weight_tb, 'maximize': True},
        {'params': loss_weight_res, 'maximize': True}
    ], lr=learning_rate)
    
# If loss_weight_res is not needed, simply set its requires_grad to False, and remove it from the optimizer
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if args.training_time and args.lr_update_threshold:
    patience = args.lr_update_threshold[0]
elif args.training_time and not args.lr_update_threshold:
    raise ValueError(
        'Please set a lr_update_threshold for the scheduler')
elif args.epochs and args.lr_update_threshold:
    patience = args.lr_update_threshold[0]
elif args.epochs and not args.lr_update_threshold:
    patience = args.epochs[0]/10

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=patience, threshold=0, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)


if args.training_time and args.epochs:
    raise ValueError(
        'Please set either epochs or training_time argument, not both.')
elif args.training_time:
    if args.weighted_loss:
        if not args.parameter_and_stress:
            raise ValueError(
                'Weighted loss distribution is only available for training where the parameter_and_stress is True')
        model, loss_histories = eu2d.train_running_time_weighted_loss(model, strain_stack, strain_stack_normalized, sxx_bound_gt, syy_bound_gt,
                                                                      youngs_gt, poissons_gt, sigma_0, criterion, optimizer, scheduler,
                                                                      loss_weight_constit, loss_weight_sides, loss_weight_tb, loss_weight_res,
                                                                      training_duration_max=args.training_time[
                                                                          0],
                                                                      loss_report_freq=args.loss_report_freq[0], early_stopping_threshold=1e-8)
    else:
        model, loss_histories = eu2d.train_running_time(model, strain_stack, strain_stack_normalized, sxx_bound_gt, syy_bound_gt,
                                                        youngs_gt, poissons_gt, sigma_0, criterion, optimizer, scheduler,
                                                        parameter_and_stress_out=args.parameter_and_stress, training_duration_max=args.training_time[
                                                            0],
                                                        loss_report_freq=args.loss_report_freq[0], early_stopping_threshold=1e-8)
elif args.epochs:
    if args.weighted_loss:
        if not args.parameter_and_stress:
            raise ValueError(
                'Weighted loss distribution is only available for training where the parameter_and_stress is True')
        model, loss_histories = eu2d.train_epoch_number_weighted_loss(model, strain_stack, strain_stack_normalized, sxx_bound_gt, syy_bound_gt,
                                                                      youngs_gt, poissons_gt, sigma_0, criterion, optimizer, scheduler,
                                                                      loss_weight_constit, loss_weight_sides, loss_weight_tb, loss_weight_res,
                                                                      max_epochs=args.epochs[
                                                                          0],
                                                                      loss_report_freq=args.loss_report_freq[0], early_stopping_threshold=1e-8)
    else:
        model, loss_histories = eu2d.train_epoch_number(model, strain_stack, strain_stack_normalized, sxx_bound_gt, syy_bound_gt,
                                                        youngs_gt, poissons_gt, sigma_0, criterion, optimizer, scheduler,
                                                        parameter_and_stress_out=args.parameter_and_stress, max_epochs=args.epochs[
                                                            0],
                                                        loss_report_freq=args.loss_report_freq[0], early_stopping_threshold=1e-8)


if not os.path.isdir(args.output_path[0]):
    os.makedirs(args.output_path[0])

eu2d.save_checkpoint(
    model, optimizer, filename=os.path.join(args.output_path[0], 'El-UNet_trained.pth.tar'))

if args.plot:
    plt.plot(loss_histories['running_loss_history'], label='training loss')
    plt.yscale('log')
    plt.show()

    plt.plot(loss_histories['youngs_mae_history'], label='E MAE')
    plt.plot(loss_histories['poissons_mae_history'], label='nu MAE')
    plt.legend()
    plt.yscale('log')
    plt.ylim(0.01, 100)
    

    plt.savefig(os.path.join(args.output_path[0], 'loss_history.png'))
    plt.show()
    
    y = model.forward(strain_stack_normalized)
    if args.parameter_and_stress:
        sxx_out = y[0, 2, :, :]
        syy_out = y[0, 3, :, :]
        sxy_out = y[0, 4, :, :]
        lame1_out = y[0, 0, :, :]
        lame2_out = y[0, 1, :, :]
    else:
        lame1_out = y[0, 0, :, :]
        lame2_out = y[0, 1, :, :]
        c11_out = 2*lame2_out + lame1_out
        c12_out = lame1_out
        c33_out = 2*lame2_out
        exx_gt = strain_stack[0, 0, :, :]
        eyy_gt = strain_stack[0, 1, :, :]
        exy_gt = strain_stack[0, 2, :, :]
        sxx_out = c11_out*exx_gt + c12_out*eyy_gt
        syy_out = c11_out*eyy_gt + c12_out*exx_gt
        sxy_out = c33_out*exy_gt

    lame1_out = sigma_0*lame1_out
    lame2_out = sigma_0*lame2_out
    sxx_out = sigma_0*sxx_out
    syy_out = sigma_0*syy_out
    sxy_out = sigma_0*sxy_out

    youngs_pred = lame2_out * \
        (3*lame1_out + 2*lame2_out)/(lame1_out+lame2_out)
    poissons_pred = lame1_out/(2*(lame1_out+lame2_out))
    youngs_pred = youngs_pred/(1-poissons_pred**2)
    poissons_pred = poissons_pred/(1-poissons_pred)

    eu2d.plot_sbs_all(os.path.join(args.output_path[0], 'comparison.png'), stress_stack[0, 0, :, :], stress_stack[0, 1, :, :], stress_stack[0, 2, :, :], youngs_gt, poissons_gt,
                      sxx_out, syy_out, sxy_out, youngs_pred, poissons_pred)
    dict_matlab = {"youngs_pred": youngs_pred.detach().cpu().numpy(),
                   "poissons_pred": poissons_pred.detach().cpu().numpy(),
                   "sxx_pred": sxx_out.detach().cpu().numpy(),
                   "syy_pred": syy_out.detach().cpu().numpy(),
                   "sxy_pred": sxy_out.detach().cpu().numpy()}

    if args.weighted_loss:
        eu2d.weight_loss_dist(
            os.path.join(args.output_path[0], 'weights_const_bound.png'), loss_weight_sides, loss_weight_tb, loss_weight_constit)
        eu2d.weight_loss_dist_single(os.path.join(
            args.output_path[0], 'weights_res.png'), loss_weight_res)
        dict_matlab["loss_weight_sides"] = loss_weight_sides.detach().cpu().numpy()
        dict_matlab["loss_weight_tb"] = loss_weight_tb.detach().cpu().numpy()
        dict_matlab["loss_weight_constit"] = loss_weight_constit.detach(
        ).cpu().numpy()
        dict_matlab["loss_weight_res"] = loss_weight_res.detach().cpu().numpy()

    scipy.io.savemat(
        os.path.join(args.output_path[0], 'results.mat'), dict_matlab)
    np.save(os.path.join(args.output_path[0], 'loss_history'),
            np.array(loss_histories['running_loss_history']))
    np.save(os.path.join(args.output_path[0], 'youngs_mae_hist'),
            np.array(loss_histories['youngs_mae_history']))
    np.save(os.path.join(args.output_path[0], 'poissons_mae_hist'),
            np.array(loss_histories['poissons_mae_history']))
