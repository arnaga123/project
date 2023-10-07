
import streamlit as st
from ultralytics import YOLO
from PIL import Image
st.title('Data Classification App')

@st.cache_resource
def my_model():
	model = YOLO('best.pt')
	return model
img = st.file_uploader('Upload your dogs image to know its breed')

if img is not None:
	im = Image.open(img)	
	st.image(im)
	mod = my_model()
	res = mod.predict(im)
	temp = res[0].probs.top5
	conf = res[0].probs.top5conf
	conf = conf.tolist()
	col = st.columns(2)
	with col[0]:
		for i in temp:
			st.write(res[0].names[i])
	with col[1]:
		for i in conf:
			st.write(str(i))
