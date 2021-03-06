import sklearn
import numpy as np
import pickle
import streamlit as st

PAGE_CONFIG = {"page_title":"Diabetes Risk Prediction","page_icon":"💪","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)

pickle_in = open('svm_diabetes.pkl','rb')
prediction_model = pickle.load(pickle_in)

st.title("Diabetes Risk Prediction")

html = '''
<style>
body {
background-image: url("data:image/jpeg;base64,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");
background-size: cover;
}
</style>
'''

def main():
	st.markdown(html, unsafe_allow_html=True)
	st.markdown("**Please *enter the following details* to know your results**")
  
	Age = st.slider("Select your age", 1, 150)
  
	select_gender = st.selectbox('Select your gender', ['Male', 'Female'])
	if (select_gender == 'Male'):
		Gender = 1
	elif (select_gender == 'Female'):
		Gender = 0
	
	st.markdown("Please ***check*** the symptoms exhibited")
	
	check_polyuria = st.checkbox('Polyuria')
	Polyuria = 1 if check_polyuria else 0
		
	check_polydipsia = st.checkbox('Polydipsia')
	Polydipsia = 1 if check_polydipsia else 0
	    
	check_weight_loss = st.checkbox('Sudden Weight Loss')
	sudden_weight_loss = 1 if check_weight_loss else 0
	
	check_polyphagia = st.checkbox('Polyphagia')
	Polyphagia = 1 if check_polyphagia else 0
		
	check_blur = st.checkbox('Visual Blurring')
	visual_blurring = 1 if check_blur else 0
		
	check_itch = st.checkbox('Itching')
	Itching = 1 if check_itch else 0
			
	check_irr = st.checkbox('Irritability')
	Irritability = 1 if check_irr else 0
	
	check_par = st.checkbox('Partial Paresis')
	partial_paresis = 1 if check_par else 0
	
	check_alopecia = st.checkbox('Alopecia')
	Alopecia = 1 if check_alopecia else 0
	
	x = [Age, Gender, Polyuria, Polydipsia, sudden_weight_loss, Polyphagia, visual_blurring, Itching, Irritability, partial_paresis, Alopecia]
	x = np.array(x)
	if (st.button('Check Results')):
		st.write("You are at ", round(((prediction_model.predict_proba(x.reshape(1, -1))*100)[0][1]),2), "% at risk")

if __name__ == '__main__':
	main()
