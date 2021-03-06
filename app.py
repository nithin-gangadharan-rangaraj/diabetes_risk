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
  
	gender = st.selectbox('Gender', ['Male', 'Female'])
	if (gender == 'Male'):
		st.write(0)
	elif (gender == 'Female'):
		st.write(1)
    
if __name__ == '__main__':
	main()
