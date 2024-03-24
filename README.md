# Denoising-Autoencoder

This project was created for the article "Denoising Image-based Experimental Data without Clean Targets based on Deep Autoencoders".

## Dependencies
Suggest installing Anaconda and Pytorch:

https://www.anaconda.com/download
pip install --upgrade torch

## To run the demo

### Cycle cylinder

1. Download the data file according to the /cylinder_data/get_cylinder_data.txt
2. Run the gen_cylinder_data.py
3. Adjust the parameter in train_cylinder.py
4. Run train_cylinder.py
5. Chosse the best model according to the methods in paper to reconstruct the clean output

### Pattern data

1. Run the gen_pattern_data.m to generate the pattern data
2. Run the gen_pattern_data.py
3. Adjust the parameter in train_pattern.py
4. Run train_pattern.py
5. Chosse the best model according to the methods in paper to reconstruct the clean output
