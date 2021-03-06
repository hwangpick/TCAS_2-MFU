import streamlit as st
import pandas as pd
import pickle

st.markdown("""
<style>
body {
    color: #FFA500;
    background-color: #D3D3D3;
}
st.image {
    width: 100px;
}
</style>
    """, unsafe_allow_html=True)

st.write(""" 

## My First Web Application 
Let's enjoy **data science** project! 

""")

st.sidebar.header('User Input') 
st.sidebar.subheader('Please enter your data:')

# -- Define function to display widgets and store data
def get_input():
    # Display widgets and store their values in variables
    Sex = st.sidebar.radio('Sex', ['Male','Female'])
    FacultyName = st.sidebar.selectbox('FacultyName', ['School of Agro-industry',
       'School of Cosmetic Science', 'School of Dentistry',
       'School of Health Science', 'School of Information Technology',
       'School of Integrative Medicine', 'School of Law',
       'School of Liberal Arts', 'School of Management', 'School of Medicine',
       'School of Nursing', 'School of Science', 'School of Sinology',
       'School of Social Innovation'])
    GPAX = st.sidebar.slider('GPAX', 0.0, 4.0, 2.0)
    GPA_Eng = st.sidebar.slider('GPA _Eng', 0.0, 4.0, 2.0)
    GPA_Math = st.sidebar.slider('GPA_Math', 0.0, 4.0, 2.0)
    GPA_Sci = st.sidebar.slider('GPA_Sci', 0.0, 4.0, 2.0)
    GPA_Sco = st.sidebar.slider('GPA_Sco',0.0, 4.0, 2.0)
    #v_Viscera_weight = st.sidebar.slider('Viscera Weight', 0.0005, 0.54, 0.17)
    #v_Shell_weight = st.sidebar.slider('Shell Weight', 0.0015, 1.0, 0.24)

    # Change the value of sex to be {'M', 'F', 'I'} as stored in the trained dataset
    #if Sex == 'Male':
        #Sex = 'Male'
    #elif Sex == 'Female':
        #Sex = 'Female'
    #else:
        #Sex ='invalid'

    # Store user input data in a dictionary
    data = {'Sex': Sex,
            'FacultyName':FacultyName,
            'GPAX': GPAX,
            'GPA_Eng': GPA_Eng,
            'GPA_Math': GPA_Math,
            'GPA_Sci': GPA_Sci,
            'GPA_Sco':  GPA_Sco}
           # 'Viscera_weight': v_Viscera_weight,
           # 'Shell_weight': v_Shell_weight}

    # Create a data frame from the above dictionary
    data_df = pd.DataFrame(data, index=[0])
    return data_df

# -- Call function to display widgets and get data from user
df = get_input()

st.header('Application of Abalone\'s Age Prediction:')

# -- Display new data from user inputs:

st.image('./1.jpg')
st.subheader('User Input:')
st.write(df)

# -- Data Pre-processing for New Data:
# Combines user input data with sample dataset
# The sample data contains unique values for each nominal features
# This will be used for the One-hot encoding
data_sample = pd.read_excel('cleaned_tcas_v3.xlsx')
data_sample = data_sample.drop(columns=['Status'])
df = pd.concat([df, data_sample],axis=0)

#One-hot encoding for nominal features
cat_data = pd.get_dummies(df[['Sex','FacultyName']])

#Combine all transformed features together
X_new = pd.concat([cat_data, df], axis=1)
X_new = X_new[:1] # Select only the first row (the user input data)

#Drop un-used feature
X_new = X_new.drop(columns=['Sex','FacultyName'])

# -- Display pre-processed new data:
st.subheader('Pre-Processed Input:')
st.write(X_new)

# -- Reads the saved normalization model
load_nor = pickle.load(open('normalization.pkl', 'rb'))
#Apply the normalization model to new data
X_new = load_nor.transform(X_new)

# -- Display normalized new data:
st.subheader('Normalized Input:')
st.write(X_new)

# -- Reads the saved classification model
load_knn = pickle.load(open('best_knn.pkl', 'rb'))
# Apply model for prediction
prediction = load_knn.predict(X_new)

# -- Display predicted class:
st.subheader('Prediction:')
#penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(prediction)
