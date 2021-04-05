import sklearn
import numpy as np
import pickle
import streamlit as st

PAGE_CONFIG = {"page_title":"Diabetes Risk Prediction","page_icon":"💪","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)

pickle_in = open('model_final_diabetes.pkl','rb')
prediction_model = pickle.load(pickle_in)

st.title("Diabetes Risk Prediction")


hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

def main():
	st.markdown(
		"""
		<style>
		.reportview-container {
		background: url("https://i.pinimg.com/originals/a5/91/17/a59117a046cbc0082afe2ce27622c0c4.jpg")
		}
		</style>
		""", unsafe_allow_html=True) 
	st.markdown(hide_streamlit_style, unsafe_allow_html = True)
	st.markdown("**Please *enter the following details* to know your results**")
  
	Age = st.slider("Select your age", 1, 100)
  
	select_gender = st.selectbox('Select your gender', ['Male', 'Female'])
	if (select_gender == 'Male'):
		Gender = 1
	elif (select_gender == 'Female'):
		Gender = 0
	
	st.markdown("Please ***check*** the symptoms exhibited")
	
	check_polyuria = st.checkbox('Polyuria (Excessive urination)') 
	Polyuria = 1 if check_polyuria else 0
	st.text("    Excessive urination")

	check_polydipsia = st.checkbox('Polydipsia (Excessive thirst)')
	Polydipsia = 1 if check_polydipsia else 0
		
	check_polyphagia = st.checkbox('Polyphagia (Excessive eating)')
	Polyphagia = 1 if check_polyphagia else 0
	
	check_genital_thrush = st.checkbox('Genital Thrush (Irritation in genital area)')
	Genital_thrush = 1 if check_genital_thrush else 0
	
	check_irr = st.checkbox('Irritability (Feeling agitated)')
	Irritability = 1 if check_irr else 0
	
	check_par = st.checkbox('Partial Paresis (Muscle weakness or impairment)')
	partial_paresis = 1 if check_par else 0
	
	check_alopecia = st.checkbox('Alopecia (Baldness)')
	Alopecia = 1 if check_alopecia else 0
	
	check_weight_loss = st.checkbox('Sudden Weight Loss')
	sudden_weight_loss = 1 if check_weight_loss else 0
	
	check_weakness = st.checkbox('Weakness')
	weakness = 1 if check_weakness else 0
	
	check_blur = st.checkbox('Visual Blurring')
	visual_blurring = 1 if check_blur else 0
		
	check_itch = st.checkbox('Itching')
	Itching = 1 if check_itch else 0
	
	check_del = st.checkbox('Delayed healing')
	delayed_healing = 1 if check_del else 0
	
	
	x = [Age, Gender, Polyuria, Polydipsia, sudden_weight_loss,weakness, Polyphagia,Genital_thrush, visual_blurring, Itching, Irritability, partial_paresis,delayed_healing, Alopecia]
	x = np.array(x)
	result = round(((prediction_model.predict_proba(x.reshape(1, -1))*100)[0][1]),2)
	my_expander = st.beta_expander("Check Results")
	with my_expander:
		if (result > 50.0):
			st.write("You are at ",result, "% at risk")
		elif (result <=50.0):
			st.write("Don't worry! You are safe 😀")

if __name__ == '__main__':
	main()
