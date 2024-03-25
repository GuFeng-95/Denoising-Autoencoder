# Denoising-Autoencoder

This project was created for the article "Denoising Image-based Experimental Data without Clean Targets based on Deep Autoencoders".

## Cite
    @article{gu2024denoising,
      title={Denoising image-based experimental data without clean targets based on deep autoencoders},
      author={Gu, Feng and Discetti, Stefano and Liu, Yingzheng and Cao, Zhaomin and Peng, Di},
      journal={Experimental Thermal and Fluid Science},
      pages={111195},
      year={2024},
      publisher={Elsevier}
    }

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
5. Choose the best model according to the methods in paper to reconstruct the clean output

### Pattern data

1. Run the gen_pattern_data.m to generate the pattern data
2. Run the gen_pattern_data.py
3. Adjust the parameter in train_pattern.py
4. Run train_pattern.py
5. Choose the best model according to the methods in paper to reconstruct the clean output
