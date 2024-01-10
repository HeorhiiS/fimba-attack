'''
This script is used to precalculate SHAP values for the models trained on the TCGA and COVID datasets.
In our attack we compute SHAP values in real time on the target sample, here to analyze the SHAP values in general we compute them for DL models in bulk.
It is highly recommended to run this script on a machine with a GPU, as it is very computationally expensive and use a sample of the dataset.
'''



import shap
import pandas as pd
import os
import pickle
import torch
import argparse
import gc
import sys
from tqdm import tqdm
from models import *

from torch.nn import DataParallel

@contextlib.contextmanager
def change_dir(path):
    _oldcwd = os.getcwd()
    os.chdir(path)

    try:
        yield
    finally:
        os.chdir(_oldcwd)


if __name__ == '__main__':

    print(torch.cuda.memory_summary())
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('model_name', type=str,
                        help='Model to load')
    parser.add_argument('dataset', type=str,
                        help='Dataset to use')
    args = parser.parse_args()

    dataset = args.dataset # covid
    model_name = args.model_name # resnet, transformer

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_name=='cnn' and dataset=='tcga':
        cnn_tcga = FastCNN()
        cnn_tcga.load_state_dict(torch.load('models/cnn_tcga.pth'))
        model = cnn_tcga
    elif model_name=='cnn' and dataset=='covid':
        cnn_covid = FastCNN()
        cnn_covid.load_state_dict(torch.load('models/cnn_covid.pth'))
        model = cnn_covid
    elif model_name=='resnet' and dataset=='tcga':
        resnet_tcga = EfficientResNet18(block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=1)
        resnet_tcga.load_state_dict(torch.load('models/resnet_tcga.pth'))
        model = resnet_tcga
    elif model_name=='resnet' and dataset=='covid':
        resnet_covid = EfficientResNet18(block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=1)
        resnet_covid.load_state_dict(torch.load('models/resnet_covid.pth'))
        model = resnet_covid
    elif model_name=='transformer' and dataset=='tcga':
        transformer_tcga = DeepVisionTransformer()
        transformer_tcga.load_state_dict(torch.load('models/transformer.pth'))
        model = transformer_tcga
    elif model_name=='transformer' and dataset=='covid':
        transformer_covid = DeepVisionTransformer()
        transformer_covid.load_state_dict(torch.load('models/transformer_covid.pth'))
        model = transformer_covid
    else:
        raise ValueError('Invalid model name or dataset name')
    print(f'Loaded model: {model_name} for {dataset} dataset\n')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    model.eval()
    
    # Load both datsets, normalize and split into train and test

    if dataset =='tcga':
        split_val = 0.33
        with change_dir("datasets/tcga"):
            tcga_data = pd.read_csv('tcga_data.csv')
            tcga_data_sl  = tcga_data.drop("Unnamed: 0", axis=1)
            x_train, x_test , y_train, y_test =  train_test_split(tcga_data_sl.drop(['_primary_site','_sample_type'], axis=1),
                                                                  tcga_data_sl['_sample_type'],
                                                                  test_size=split_val, random_state=42)

            num_classes = len(np.unique(y_train))
            num_samples = 700 #1000 for the rest of the models
            label0_indices = np.where(y_train == 0)[0]
            label1_indices = np.where(y_train == 1)[0]

            sample_indices0 = np.random.choice(label0_indices, size=num_samples, replace=False)
            sample_indices1 = np.random.choice(label1_indices, size=num_samples, replace=False)
            sample_indices = np.concatenate((sample_indices0, sample_indices1), axis=0)

            tcga_data_train = TCGA_Dataset(x_train.iloc[sample_indices], y_train.iloc[sample_indices], scale_range=(-1,1))
            tcga_data_test = TCGA_Dataset(x_test, y_test, scale_range=(-1,1))
            train_data, _ = tcga_data_train[:]
            test_data, _ = tcga_data_test[:]
            vis_dataset = tcga_data_train.scaled_data


    else:
        split_val = 0.33
        resamp = True
        ds_size = 50000
        with change_dir("datasets/viral/covid"):
            covid_data = pd.read_csv('covid_data.csv')

            if resamp:
                sampled_data = resample(covid_data, n_samples=ds_size, stratify=covid_data['disease__ontology_label'])
                sm_covid_data_x = sampled_data.drop(['disease__ontology_label', 'Unnamed: 0', 'cell_type_intermediate'], axis=1)
                sm_covid_data_y = sampled_data['disease__ontology_label']
                X_train, X_test, Y_train, Y_test = train_test_split(sm_covid_data_x, sm_covid_data_y,
                                                                    test_size=split_val, random_state=42)

                num_samples = 1000
                label0_indices = np.where(Y_train == 0)[0]
                label1_indices = np.where(Y_train == 1)[0]

                sample_indices0 = np.random.choice(label0_indices, size=num_samples, replace=False)
                sample_indices1 = np.random.choice(label1_indices, size=num_samples, replace=False)
                sample_indices = np.concatenate((sample_indices0, sample_indices1), axis=0)

                covid_data_train = COVID_Dataset(X_train.iloc[sample_indices], Y_train.iloc[sample_indices], scale_range=(-1,1))
                covid_data_test = COVID_Dataset(X_test, Y_test, scale_range=(-1, 1))

                train_data, _ = covid_data_train[:]
                train_data_full, _ = covid_data_train[:]
                test_data, _ = covid_data_test[:]
                vis_dataset = covid_data_train.scaled_data



            else:
                covid_data_x = covid_data.drop(['disease__ontology_label', 'Unnamed: 0', 'cell_type_intermediate'], axis=1)
                covid_data_y = covid_data['disease__ontology_label']
                X_train, X_test, Y_train, Y_test = train_test_split(covid_data_x, covid_data_y,
                                                                    test_size=split_val, random_state=42)

                num_samples = 1000
                label0_indices = np.where(Y_train == 0)[0]
                label1_indices = np.where(Y_train == 1)[0]

                sample_indices0 = np.random.choice(label0_indices, size=num_samples, replace=False)
                sample_indices1 = np.random.choice(label1_indices, size=num_samples, replace=False)
                sample_indices = np.concatenate((sample_indices0, sample_indices1), axis=0)

                covid_data_train = COVID_Dataset(X_train.iloc[sample_indices], Y_train.iloc[sample_indices], scale_range=(-1,1))
                covid_data_test = COVID_Dataset(X_test, Y_test, scale_range=(-1, 1))

                train_data, _ = covid_data_train[:]
                train_data_full, _ = covid_data_train[:]
                test_data, _ = covid_data_test[:]
                vis_dataset = covid_data_train.scaled_data

    print(f'Dataset {dataset} of shape {train_data.shape} loaded successfully!\n')
    print('Starting SHAP analysis...\n')
    print(torch.cuda.memory_summary())
    with torch.no_grad():
        explainer = shap.DeepExplainer(model,train_data.unsqueeze(1).to(device).to(torch.float32))
        del train_data

    print('SHAP explainer created!\n')
    expected_value = explainer.expected_value[0]

    pbar = tqdm(total=test_data.shape[0])


    shap_values = []
    flag = True

    for i in range(test_data.shape[0]):
        # Get the i-th sample
        x = test_data[i:i+1]
        # Calculate SHAP values for the i-th sample
        shap_values_i = explainer.shap_values(x.to(device).unsqueeze(1).to(torch.float32))
        if flag:
            print(f'Shap calculation check, val = {shap_values_i}')
            flag = False
        del x
        gc.collect()
        torch.cuda.empty_cache()
        # Append the SHAP values for the i-th sample to the list of SHAP values
        shap_values.append(shap_values_i[0])
        pbar.set_description('Calculating SHAP values for sample: %d' % (1 + i))
        pbar.update(1)
        pbar.refresh()

    pbar.close()
    shapvalues = np.concatenate(shap_values, axis=0)
    print(f'SHAP values shape: {shapvalues.shape}')
    #shapvalues = explainer.shap_values(test_data.unsqueeze(1).to(device).to(torch.float32))


    if dataset=='tcga':
        base_values = np.full((2881),expected_value)
    else:
        base_values = np.full((2000),expected_value)
    print('SHAP analysis finished\n')

    shap_arr = shapvalues.reshape(shapvalues.shape[0],1,-1)
    shap_arr = shap_arr.transpose(0,2,1).squeeze()

    exp_model = shap.Explanation(shap_arr, 
                    base_values,
                    data=vis_dataset.values,
                    feature_names=vis_dataset.columns)
    print('Saving SHAP values...\n')
    with open('shap_dicts/shap_'+model_name+'_2_'+dataset+'.pkl', 'wb') as f:
        pickle.dump(exp_model, f)