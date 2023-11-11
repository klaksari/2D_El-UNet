# Physics-informed UNets for Discovering Hidden Elasticity in Heterogeneous Materials
Data and codes from our paper: 
[Physics-informed UNets for Discovering Hidden Elasticity in Heterogeneous Materials](https://doi.org/10.1016/j.jmbbm.2023.106228)


We developed a novel UNet-based neural network model for inversion in elasticity (El-UNet) to infer the spatial distributions of mechanical parameters from strain maps as input images, normal stress boundary conditions, and domain physics information.
This repository currently contains data and sample code from our published paper.


![image](https://github.com/klaksari/2D_El-UNet/assets/60515966/0806bfa6-2222-4f6a-bf08-404b27d2b0c9)


## Running the code
After cloning or downloading this repository, running the inverse script can be done with the desired training parameters. Below is an example:

```
python3 -u inverse_plane_stress_script.py --training_time 30 --loss_report_freq 100 --unet_num_channels 64 128 256 512 --weighted_loss --parameter_and_stress --input_path 'soft_background_example_data' --output_path 'ExampleTraining'
```


Sample data for the examples included in the study can be found in the 'soft_background_example_data', 'stiff_background_example_data', and 'tumor_example_data' directories.


Have a question about implementing the code? contact us at [klaksari@engr.ucr.edu](mailto:klaksari@engr.ucr.edu), [akamali@arizona.edu](mailto:akamali@arizona.edu).
