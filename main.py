import pickle
import streamlit as st

# load the trained model
pickle_in = open('rf_bsm_final_model.pkl', 'rb')
classifier = pickle.load(pickle_in)

# load the scaler
scaler_in = open('bsm_scaler.pkl', 'rb')
bsm_scaler = pickle.load(scaler_in)


@st.cache()
# defining the function which will make the prediction using the data which the user inputs
def prediction(clump_thickness, uniformity_of_cell_size, uniformity_of_cell_shape, marginal_adhesion, single_epithelial_cell_size, bare_nuclei, bland_chromatin, normal_nucleoli, mitoses):

    # make the prediction
    prediction = classifier.predict([[clump_thickness, uniformity_of_cell_size,
                                    uniformity_of_cell_shape, marginal_adhesion,
                                    single_epithelial_cell_size, bare_nuclei,
                                    bland_chromatin, normal_nucleoli, mitoses]])

    if prediction == 0:
        result = 'Benign'
    else:
        result = 'Malignant'

    return result

# main function


def main():
    st.title('Prediksi Kanker Payudara Menggunakan Algoritma Random Forest')

    # input the data
    clump_thickness = st.sidebar.number_input('Clump Thickness')
    uniformity_of_cell_size = st.sidebar.number_input(
        'Uniformity of Cell Size')
    uniformity_of_cell_shape = st.sidebar.number_input(
        'Uniformity of Cell Shape')
    marginal_adhesion = st.sidebar.number_input('Marginal Adhesion')
    single_epithelial_cell_size = st.sidebar.number_input(
        'Single Epithelial Cell Size')
    bare_nuclei = st.sidebar.number_input('Bare Nuclei')
    bland_chromatin = st.sidebar.number_input('Bland Chromatin')
    normal_nucleoli = st.sidebar.number_input('Normal Nucleoli')
    mitoses = st.sidebar.number_input('Mitoses')

    # when the 'Predict' is clicked, make the prediction and store it
    if st.sidebar.button('Predict'):
        result = prediction(clump_thickness, uniformity_of_cell_size,
                            uniformity_of_cell_shape, marginal_adhesion,
                            single_epithelial_cell_size, bare_nuclei,
                            bland_chromatin, normal_nucleoli, mitoses)
        st.success(f'Prediction Result = {result}')
        st.write('Will be filled by the data')


if __name__ == '__main__':
    main()
