import streamlit as st
import pandas as pd
import pickle

st.sidebar.image('Img/123.png', width=250)
st.image('Img/222.png', width=500)
st.write(""" 

## My First Web Application 
Let's enjoy **data science** project! 

""")

st.sidebar.header('User Input')
st.sidebar.subheader('Please enter your data:')

FacultyName = ['School of Medicine',
               'School of Management',
               'School of Dentistry',
               'School of Liberal Arts',
               'School of Law',
               'School of Nursing',
               'School of Information Technology',
               'School of Integrative Medicine',
               'School of Cosmetic Science',
               'School of Health Science',
               'School of Sinology',
               'School of Agro-industry',
               'School of Science',
               'School of Social Innovation'
               ]


# -- Define function to display widgets and store data


def get_input():
    # Display widgets and store their values in variables
    v_Sex = st.sidebar.radio('Sex', ['Male', 'Female'])

    v_FacultyName = st.sidebar.selectbox('FacultyName', FacultyName)

    v_GPAX = st.sidebar.slider('GPAX', 0.0, 4.0)

    v_GPA_ENG = st.sidebar.slider('English GPA', 0.0, 4.0)

    v_GPA_MATH = st.sidebar.slider('Math GPA', 0.0, 4.0)

    v_GPA_SCI = st.sidebar.slider('Science GPA', 0.0, 4.0)

    v_GPA_SO = st.sidebar.slider('Social GPA', 0.0, 4.0)

    v_Q1 = st.sidebar.radio('Q1', [0, 1])
    v_Q2 = st.sidebar.radio('Q2', [0, 1])
    v_Q3 = st.sidebar.radio('Q3', [0, 1])
    v_Q4 = st.sidebar.radio('Q4', [0, 1])
    v_Q5 = st.sidebar.radio('Q5', [0, 1])
    v_Q6 = st.sidebar.radio('Q6', [0, 1])
    v_Q7 = st.sidebar.radio('Q7', [0, 1])
    v_Q8 = st.sidebar.radio('Q8', [0, 1])
    v_Q9 = st.sidebar.radio('Q9', [0, 1])
    v_Q10 = st.sidebar.radio('Q10', [0, 1])
    v_Q11 = st.sidebar.radio('Q11', [0, 1])
    v_Q12 = st.sidebar.radio('Q12', [0, 1])
    v_Q13 = st.sidebar.radio('Q13', [0, 1])
    v_Q14 = st.sidebar.radio('Q14', [0, 1])
    v_Q15 = st.sidebar.radio('Q15', [0, 1])
    v_Q16 = st.sidebar.radio('Q16', [0, 1])
    v_Q17 = st.sidebar.radio('Q17', [0, 1])
    v_Q18 = st.sidebar.radio('Q18', [0, 1])
    v_Q19 = st.sidebar.radio('Q19', [0, 1])
    v_Q20 = st.sidebar.radio('Q20', [0, 1])
    v_Q21 = st.sidebar.radio('Q21', [0, 1])
    v_Q22 = st.sidebar.radio('Q22', [0, 1])
    v_Q23 = st.sidebar.radio('Q23', [0, 1])
    v_Q24 = st.sidebar.radio('Q24', [0, 1])
    v_Q25 = st.sidebar.radio('Q25', [0, 1])
    v_Q26 = st.sidebar.radio('Q26', [0, 1])
    v_Q27 = st.sidebar.radio('Q27', [0, 1])
    v_Q28 = st.sidebar.radio('Q28', [0, 1])
    v_Q29 = st.sidebar.radio('Q29', [0, 1])
    v_Q30 = st.sidebar.radio('Q30', [0, 1])
    v_Q31 = st.sidebar.radio('Q31', [0, 1])
    v_Q32 = st.sidebar.radio('Q32', [0, 1])
    v_Q33 = st.sidebar.radio('Q33', [0, 1])
    v_Q34 = st.sidebar.radio('Q34', [0, 1])
    v_Q35 = st.sidebar.radio('Q35', [0, 1])
    v_Q36 = st.sidebar.radio('Q36', [0, 1])
    v_Q37 = st.sidebar.radio('Q37', [0, 1])
    v_Q38 = st.sidebar.radio('Q38', [0, 1])
    v_Q39 = st.sidebar.radio('Q39', [0, 1])
    v_Q40 = st.sidebar.radio('Q40', [0, 1])
    v_Q41 = st.sidebar.radio('Q41', [0, 1])
    v_Q42 = st.sidebar.radio('Q42', [0, 1])

    # Store user input data in a dictionary
    data = {'Sex': v_Sex,
            'FacultyName': v_FacultyName,
            'GPAX': v_GPAX,
            'GPA_Eng': v_GPA_ENG,
            'GPA_Math': v_GPA_MATH,
            'GPA_Sci': v_GPA_SCI,
            'GPA_Sco': v_GPA_SO,
            'Q1': v_Q1,
            'Q2': v_Q2,
            'Q3': v_Q3,
            'Q4': v_Q4,
            'Q5': v_Q5,
            'Q6': v_Q6,
            'Q7': v_Q7,
            'Q8': v_Q8,
            'Q9': v_Q9,
            'Q10': v_Q10,
            'Q11': v_Q11,
            'Q12': v_Q12,
            'Q13': v_Q13,
            'Q14': v_Q14,
            'Q15': v_Q15,
            'Q16': v_Q16,
            'Q17': v_Q17,
            'Q18': v_Q18,
            'Q19': v_Q19,
            'Q20': v_Q20,
            'Q21': v_Q21,
            'Q22': v_Q22,
            'Q23': v_Q23,
            'Q24': v_Q24,
            'Q25': v_Q25,
            'Q26': v_Q26,
            'Q27': v_Q27,
            'Q28': v_Q28,
            'Q29': v_Q29,
            'Q30': v_Q30,
            'Q31': v_Q31,
            'Q32': v_Q32,
            'Q33': v_Q33,
            'Q34': v_Q34,
            'Q35': v_Q35,
            'Q36': v_Q36,
            'Q37': v_Q37,
            'Q38': v_Q38,
            'Q39': v_Q39,
            'Q40': v_Q40,
            'Q41': v_Q41,
            'Q42': v_Q42,

            }

    # Create a data frame from the above dictionary
    data_df = pd.DataFrame(data, index=[0])
    return data_df


# -- Call function to display widgets and get data from user
df = get_input()

st.header('Application of TCAS round 2 MFUStudent Prediction:')

# -- Display new data from user inputs:
st.subheader('User Input:')
st.write(df)
st.write(
    '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
st.markdown(
    """
<style>
.sidebar .sidebar-content{
    background: linear-gradient(#e66465, #9198e5);
    color: white;
    font-family: 'IBM Plex Mono', monospace;
}



</style>
""",
    unsafe_allow_html=True
)


# -- Data Pre-processing for New Data:
# Combines user input data with sample dataset
# The sample data contains unique values for each nominal features
# This will be used for the One-hot encoding
data_sample = pd.read_csv('test_sample_v2.csv')


data_sample = data_sample.drop(columns=['Unnamed: 0', 'TCAS'])

df = pd.concat([df, data_sample], axis=0)


# get all nominal / ordinal / Boolean features
cat_data = df[['Sex', 'FacultyName']]
cat_data = pd.get_dummies(cat_data)

# get all numberic features
num_data = df.select_dtypes('number')

# Combine all transformed features together
X_new = pd.concat([cat_data, num_data], axis=1)
X_new = X_new[:1]

# -- Display pre-processed new data:
st.subheader('Pre-Processed Input:')
st.write(X_new)

# -- Reads the saved normalization model
load_mms = pickle.load(open('normalization_v5.pkl', 'rb'))
# Apply the normalization model to new data
# Rescaling features into a range of [0,1]
X_new = load_mms.fit_transform(X_new)

# -- Display normalized new data:
st.subheader('Normalized Input:')
st.write(X_new)

# -- Reads the saved classification model
load_knn = pickle.load(open('best_knn_v5.pkl', 'rb'))
# Apply model for prediction
prediction = load_knn.predict(X_new)

# -- Display predicted class:
st.subheader('Prediction:')
#penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(prediction)
