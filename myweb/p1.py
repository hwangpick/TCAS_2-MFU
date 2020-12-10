import streamlit as st
import pandas as pd
import pickle

st.markdown(
    """
<style>
.body{
   background-color: gray;
}
.reportview-container .markdown-text-container {
    font-family: monospace;
}
.sidebar .sidebar-content {
    background-image: linear-gradient(aqua,gold,salmon);
    color: black;
    font-size: 40px;
}
.Widget>label {
    color:black;
    font-family: monospace;
}
[class^="st-b"]  {
    color: black;
    font-family: monospace;
}
.st-bb {
    background-color: transparent;
}
.st-at {
    
}
footer {
    font-family: monospace;
}
.reportview-container .main footer, .reportview-container .main footer a {
    color:#fcfafa;
}
header .decoration {
    background-image: none;
}


</style>
""",
    unsafe_allow_html=True,
)
#st.image('./007.png')
st.header('TCAS MFU God7')

st.sidebar.header('User Input')
st.sidebar.subheader('Please enter your data:')


def get_input():
    #widgets
    k_Sex = st.sidebar.radio('Sex', ['Male','Female'])
    k_Fac = st.sidebar.selectbox('FacultyName', ['School of Agro-industry',
       'School of Cosmetic Science', 'School of Dentistry',
       'School of Health Science', 'School of Information Technology',
       'School of Integrative Medicine', 'School of Law',
       'School of Liberal Arts', 'School of Management', 'School of Medicine',
       'School of Nursing', 'School of Science', 'School of Sinology',
       'School of Social Innovation'])
    k_GPX = st.sidebar.slider('GPAX', 0.00, 4.00, 2.00)
    k_Eng = st.sidebar.slider('GPA_Eng', 0.00, 4.00, 2.00)
    k_Mat = st.sidebar.slider('GPA_Math', 0.00, 4.00, 2.00)
    k_Sci = st.sidebar.slider('GPA_Sci', 0.00, 4.00, 2.00)
    k_Sco = st.sidebar.slider('GPA_Sco', 0.00, 4.00, 2.00)

   #  if k_Sex == 'Male': k_Sex = 'M'
   #  else: k_Sex = 'F'
    

    #dictionary
    data = {'Sex': k_Sex,
            'FacultyName':k_Fac,
            'GPAX': k_GPX,
            'GPA_Eng': k_Eng,
            'GPA_Math': k_Mat,
            'GPA_Sci': k_Sci,
            'GPA_Sco': k_Sco
            }

    #create data frame
    data_df = pd.DataFrame(data, index=[0])
    return data_df

df = get_input()
st.write(df)

data_sample = pd.read_excel('cleaned_tcas_v3.xlsx')
data_sample = data_sample.drop(columns=['Status'])
df = pd.concat([df, data_sample],axis=0)
st.write(df)


cat_data = pd.get_dummies(df[['Sex','FacultyName']])
# cat_data = pd.get_dummies(df[[]])
# cat_data = pd.get_dummies(df[[]])
# cat_data = pd.get_dummies(df[[]])

st.write(cat_data)

#Combine all transformed features together
X_new = pd.concat([cat_data, df], axis=1)
X_new = X_new[:1] # Select only the first row (the user input data)

#Drop un-used feature
X_new = X_new.drop(columns=['Sex','FacultyName'])
st.write(X_new)

# -- Reads the saved normalization model
load_nor = pickle.load(open('normalization.pkl', 'rb'))
#Apply the normalization model to new data
X_new = load_nor.transform(X_new)
st.write(X_new)

load_knn = pickle.load(open('best_knn.pkl', 'rb'))
# Apply model for prediction
prediction = load_knn.predict(X_new)
st.write(prediction)
