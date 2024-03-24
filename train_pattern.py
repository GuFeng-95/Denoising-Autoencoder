import random
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from data_set import pattern_Dataset
from model import *
import scipy.io as io
from matplotlib import pyplot as plt
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == "__main__":

    model_type = 'AE'
    # AE: use auto-encoder AE_attention: use auto-encoder with self-attention

    print_interval=1
    #print results every print_interval
    plot_interval=1
    # plot denoised results every plot_interval
    save_interval=100
    # save trained model every save_interval

    for vector_number in [2,4,8,16,32,64,128]:
    # define the vectorsize list, for AE_attention you can use only a large vectorsize
        for g in [0.1,0.8,20,80]:
            # define the noise level 0.1 for SNR 1/0.1, 0.8 for SNR 1/0.8, 20 for SNR 1/20, 80 for SNR 1/80
            print(model_type,'   noise:',g)
            seed=2034
            batch_size=64
            n_epoch=5000
            # define the total train epoch
            gauss_value = g

            save_path='output_pattern/{}_{}_noise{}'.format(model_type,vector_number,gauss_value)
            model_save_path='output_pattern/models/{}_{}_noise{}'.format(model_type,vector_number,gauss_value)
            try:
                print(save_path)
                os.makedirs('./'+save_path)
            except:
                pass
            try:
                os.makedirs('./' + model_save_path)
            except:
                pass


            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                print(torch.cuda.get_device_name(torch.cuda.current_device()))
            # define the random seed and device
            pattern_data = io.loadmat('./pattern_data/Patterns.mat')['Q'].T
            pattern_data_noise = np.load('./pattern_data/pattern_noise_{}.npy'.format(gauss_value))
            # load clean and noisy data, clean data is only for computing the error with ground truth

            train_dataset = pattern_Dataset(pattern_data, pattern_data_noise, if_norm=False)
            # get dataset in torch format

            clean_show_data,noisy_show_data=train_dataset[0]
            clean_show_data = torch.Tensor([clean_show_data]).to(device)
            noisy_show_data = torch.Tensor([noisy_show_data]).to(device)
            # prepare the data to plot

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

            if model_type == 'AE':
                model = autoencoder_n(128,320,vector_number)
            elif model_type == 'AE_attention':
                model = autoencoder_attention_n(128,320,vector_number)
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=0)
            l1_loss = nn.L1Loss()
            # define the model loss function


            train_loss_list=[] # train_error in the denoising process
            valid_loss_list=[] # reconstruction_error in the denoising process


            print("start training")
            for epoch in range(n_epoch):
                model.train()
                train_loss = 0.0
                valid_loss= 0.0

                for i, batch_value in enumerate(train_loader):
                    glob_iter = epoch * len(train_loader) + i
                    clean_data,noise_data=batch_value

                    train_inputs = noise_data.float().to(device)
                    clean_data=clean_data.float().to(device)
                    optimizer.zero_grad()

                    vector,train_outputs = model(train_inputs)
                    loss = l1_loss(train_outputs, train_inputs)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()


                    clean_data_np = clean_data.cpu().numpy()
                    Y=train_outputs.cpu().detach().numpy()
                    loss2=np.mean(np.abs(Y-clean_data_np))
                    valid_loss += loss2
                    # compute the error with ground truth


                train_loss /= len(train_loader)
                valid_loss /= len(train_loader)
                train_loss_list.append(train_loss)
                valid_loss_list.append(valid_loss)

                if epoch % print_interval== 0:
                    print(
                        "Training: Epoch[{:0>3}/{:0>3}] Iter[{:0>4}]/[{:0>4}] train loss: {:.6f} valid loss:{:.6f}".format(
                            epoch + 1, n_epoch, i + 1, len(train_loader), train_loss, valid_loss))

                if epoch % plot_interval== 0:
                    model.eval()

                    _, AE_show = model(noisy_show_data)
                    AE_show=AE_show.cpu().detach().numpy()

                    plt.figure(figsize=(6, 3))
                    plt.imshow(AE_show[0][0], cmap='jet', vmin=-2.5, vmax=2.5)

                    x_tt = np.arange(0, 320, 40)
                    x_t = np.around((np.arange(0, 320, 40) / 10))
                    y_tt = np.arange(4, 125, 30)
                    y_t = np.around((np.arange(-60, 61, 30) / 10))

                    plt.xticks(x_tt, x_t)
                    plt.yticks(y_tt, y_t)
                    plt.xlabel('x (mm)')
                    plt.ylabel('y (mm)')

                    cb = plt.colorbar()

                    plt.savefig(os.path.join(save_path, f'AE_epoch_{epoch}.png'), dpi=300)
                    plt.close()
                    model.train()

                    model.train()
                if epoch==0:
                    clean_show_data_np=clean_show_data.cpu().detach().numpy()

                    plt.imshow(clean_show_data_np[0][0], cmap='jet', vmin=-2.5, vmax=2.5)
                    x_tt = np.arange(0, 320, 40)
                    x_t = np.around((np.arange(0, 320, 40) / 10))
                    y_tt = np.arange(4, 125, 30)
                    y_t = np.around((np.arange(-60, 61, 30) / 10))

                    plt.xticks(x_tt, x_t)
                    plt.yticks(y_tt, y_t)
                    plt.xlabel('x (mm)')
                    plt.ylabel('y (mm)')

                    cb = plt.colorbar()
                    plt.savefig(os.path.join(save_path, f'clean.png'), dpi=300)
                    plt.close()


                    noisy_show_data_np = noisy_show_data.cpu().detach().numpy()
                    plt.imshow(noisy_show_data_np[0][0], cmap='jet', vmin=-2.5, vmax=2.5)
                    x_tt = np.arange(0, 320, 40)
                    x_t = np.around((np.arange(0, 320, 40) / 10))
                    y_tt = np.arange(4, 125, 30)
                    y_t = np.around((np.arange(-60, 61, 30) / 10))

                    plt.xticks(x_tt, x_t)
                    plt.yticks(y_tt, y_t)
                    plt.xlabel('x (mm)')
                    plt.ylabel('y (mm)')

                    cb = plt.colorbar()
                    plt.savefig(os.path.join(save_path, f'noisy.png'), dpi=300)
                    plt.close()



                if (epoch + 1) % 100 == 0:
                    filename = '{}_noise_{}_{:0>4}.pth'.format(model_type,gauss_value,epoch + 1)
                    torch.save(model, os.path.join(model_save_path, filename))
                    np.save('./{}/train_loss.npy'.format(save_path), np.array(train_loss_list))
                    np.save('./{}/valid_loss.npy'.format(save_path), np.array(valid_loss_list))
            print("Finished training.")


            train_loss_list=np.array(train_loss_list)
            np.save('./{}/train_loss.npy'.format(save_path),train_loss_list)
            plt.plot(train_loss_list,'-',label='Train')
            plt.xlabel('Epoch')
            plt.ylabel('L1 loss')
            plt.yscale('log')
            plt.savefig('./{}/train_loss.png'.format(save_path),dpi=300)
            plt.show()
            plt.close()
            # save the train loss

            valid_loss_list=np.array(valid_loss_list)
            np.save('./{}/valid_loss.npy'.format(save_path),valid_loss_list)
            plt.plot(valid_loss_list,'-',label='Valid')
            plt.xlabel('Epoch')
            plt.ylabel('L1 loss')
            plt.yscale('log')
            plt.savefig('./{}/valid_loss.png'.format(save_path),dpi=300)
            plt.show()
            plt.close()

            # save the reconstruction error


