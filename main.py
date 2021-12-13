from fpdf.fpdf import FPDF
import streamlit as st
from helper import get_summary, load_model, to_dataframe, data_preprocessing, get_prediction

st.set_page_config(layout="wide")


def main():  # main function
    st.title('Breast Cancer Prediction Using KNN-SMOTE Algorithms')

    # load the trained model
    classifier = load_model('KNN_bsm_classifier.pkl')

    # load the scaler
    bsm_scaler = load_model('smote_scaler.pkl')

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

        # preprocess the data
        data_prep = data_preprocessing(cancer_data, bsm_scaler)

        # get prediction result
        result, recommendation = get_prediction(data_prep, classifier)

        # show the data summary and the predicted result
        summary, predicted = st.columns([1, 2])
        with summary:
            st.markdown('**Prediction Summary**')

            # show data summary
            summary_data = get_summary(cancer_data)
            st.write(summary_data)

        with predicted:
            st.markdown('**Prediction Result**')

            # show the predicted result
            st.write(f'Predicted Cancer Type : **{result}**')
            st.write(f'Recommendation        : **{recommendation}**')


if __name__ == '__main__':
    main()
