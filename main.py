import pickle
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(layout="wide")

# load the trained model
pickle_in = open('rf_bsm_final_model.pkl', 'rb')
classifier = pickle.load(pickle_in)

# load the scaler
scaler_in = open('bsm_scaler.pkl', 'rb')
bsm_scaler = pickle.load(scaler_in)


@st.cache()
# convert data into dataframe
def to_dataframe(clump_thickness, uniformity_of_cell_size, uniformity_of_cell_shape, marginal_adhesion, single_epithelial_cell_size, bare_nuclei, bland_chromatin, normal_nucleoli, mitoses):
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


# data preprocessing
def data_preprocessing(data):
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
def get_prediction(data_prep):

    # make the prediction
    # prediction = classifier.predict([[clump_thickness, uniformity_of_cell_size,
    #                                 uniformity_of_cell_shape, marginal_adhesion,
    #                                 single_epithelial_cell_size, bare_nuclei,
    #                                 bland_chromatin, normal_nucleoli, mitoses]])

    pred_result = classifier.predict(data_prep)

    if pred_result == 0:
        result = 'Benign'
    else:
        result = 'Malignant'

    return result

# main function


def main():
    st.title('Breast Cancer Prediction Using Random Forest Algorithm')

    # input the data
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        clump_thickness = st.number_input('Clump Thickness')
        uniformity_of_cell_size = st.number_input('Uniformity of Cell Size')
        uniformity_of_cell_shape = st.number_input('Uniformity of Cell Shape')

    with col2:
        marginal_adhesion = st.number_input('Marginal Adhesion')
        single_epithelial_cell_size = st.number_input(
            'Single Epithelial Cell Size')
        bare_nuclei = st.number_input('Bare Nuclei')

    with col3:
        bland_chromatin = st.number_input('Bland Chromatin')
        normal_nucleoli = st.number_input('Normal Nucleoli')
        mitoses = st.number_input('Mitoses')

    # when the 'Predict' is clicked, make the prediction and store it
    if st.button('Predict'):
        # convert data into dataframe
        cancer_data = to_dataframe(clump_thickness, uniformity_of_cell_size,
                                   uniformity_of_cell_shape, marginal_adhesion,
                                   single_epithelial_cell_size, bare_nuclei,
                                   bland_chromatin, normal_nucleoli, mitoses)
        # st.write(result)
        # st.write(cancer_data)

        data_prep = data_preprocessing(cancer_data)
        # st.write(data_prep)
        swap_data = cancer_data.T

        result = get_prediction(data_prep)
        # st.write(swap_data)
        # st.write(result)

        summary, predicted = st.columns([1, 2])
        with summary:
            st.markdown('**Prediction Summary**')
            swap_data.reset_index(inplace=True)
            swap_data = swap_data.rename(
                {'index': 'features', 0: 'data_input'}, axis=1)
            st.write(swap_data)

        with predicted:
            st.markdown('**Predicted Result**')
            st.write(f'Predicted Cancer Type : **{result}**')


if __name__ == '__main__':
    main()
