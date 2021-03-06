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
background-image: url("https://i.pinimg.com/originals/a0/7e/8f/a07e8f05a7d516ba7fd90519c5126058.jpg");
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
	result = round(((prediction_model.predict_proba(x.reshape(1, -1))*100)[0][1]),2)
	if (st.button('Check Results')):
		if (result > 50.0):
			st.write("You are at ",result, "% at risk")
		elif (result <=50.0):
			st.write("Don't worry! You are safe")

if __name__ == '__main__':
	main()
