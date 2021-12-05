import pickle
import pandas as pd
import numpy as np


def load_model(model_name):  # function for load the model and scaler
    model_in = open(model_name, 'rb')
    model = pickle.load(model_in)

    return model


def to_dataframe(clump_thickness, uniformity_of_cell_size, uniformity_of_cell_shape, marginal_adhesion, single_epithelial_cell_size, bare_nuclei, bland_chromatin, normal_nucleoli, mitoses):
    # convert data input into a dataframe
    # convert to pandas dataframe
    data = {'clump_thickness': [clump_thickness],
            'uniformity_of_cell_size': [uniformity_of_cell_size],
            'uniformity_of_cell_shape': [uniformity_of_cell_shape],
            'marginal_adhesion': [marginal_adhesion],
            'single_epithelial_cell_size': [single_epithelial_cell_size],
            'bare_nuclei': [bare_nuclei],
            'bland_chromatin': [bland_chromatin],
            'normal_nucleoli': [normal_nucleoli],
            'mitoses': [mitoses]}

    cancer_data = pd.DataFrame(data=data)
    return cancer_data


def data_preprocessing(data, bsm_scaler):  # preprocess the data
    data_prep = data.copy()

    # log transform marginal adhesion and single epithelial cell size data
    data_prep['marginal_adhesion'] = np.log(data_prep['marginal_adhesion'])
    data_prep['single_epithelial_cell_size'] = np.log(
        data_prep['single_epithelial_cell_size'])

    # normalize the data using StandardScaler()
    data_norm = data_prep.copy()
    columns = ['clump_thickness', 'uniformity_of_cell_size',
               'uniformity_of_cell_shape', 'marginal_adhesion',
               'single_epithelial_cell_size', 'bare_nuclei',
               'bland_chromatin', 'normal_nucleoli', 'mitoses']

    bsm_scaled_feature = bsm_scaler.transform(data_norm)

    # put the normalized data into dataframe
    data_balanced = pd.DataFrame(bsm_scaled_feature, columns=columns)

    return data_balanced


# defining the function which will make the prediction using the data which the user inputs
def get_prediction(data_prep, classifier):
    # make the prediction
    pred_result = classifier.predict(data_prep)

    if pred_result == 0:
        result = 'Benign'
    else:
        result = 'Malignant'

    return result


def get_summary(data):
    data_summary = data.copy()
    data_summary = data_summary.rename({'clump_thickness': 'Clump Thickness',
                                        'uniformity_of_cell_size': 'Uniformity of Cell Size',
                                        'uniformity_of_cell_shape': 'Uniformity of Cell Shape',
                                        'marginal_adhesion': 'Marginal Adhesion',
                                        'single_epithelial_cell_size': 'Single Epithelial Cell Size',
                                        'bare_nuclei': 'Bare Nuclei',
                                        'bland_chromatin': 'Bland Chromatin',
                                        'normal_nucleoli': 'Normal Nucleoli',
                                        'mitoses': 'Mitoses'}, axis=1)
    data_summary = data_summary.T
    data_summary.reset_index(inplace=True)
    data_summary = data_summary.rename(
        {'index': 'Features', 0: 'Data Input'}, axis=1)

    return data_summary
