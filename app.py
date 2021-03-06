import sklearn
import numpy as np
import pickle
import streamlit as st

PAGE_CONFIG = {"page_title":"Diabetes Risk Prediction","page_icon":"💪","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)

pickle_in = open('svm_diabetes.pkl','rb')
model = pickle.load(pickle_in)

st.title("Diabetes Risk Prediction")

html = '''
<style>
body {
background-image: url("https://wallpapercave.com/wp/wp2838450.jpg");
background-size: cover;
}
</style>
'''

def main():
	st.markdown(html, unsafe_allow_html=True)
	st.subheader("*Change it **later***")
  
	age = st.slider("Select your age", 1, 150)
  
	select_gender = st.selectbox('Gender', ['Male', 'Female'])
	if (select_gender == 'Male'):
		gender = 1
	elif (select_gender == 'Female'):
		gender = 0
	
	check_polyuria = st.checkbox('Polyuria')
	if check_polyuria:
		polyuria = 1
	else:
		polyuria = 0
	with st.beta_expander('See info about Polyuria'):
		st.write("""Polyuria is defined as the frequent passage of large volumes of urine – 
			 more than 3 litres a day compared to the normal daily urine output in adults of about 1 to 2 litres.""")
    
if __name__ == '__main__':
	main()
