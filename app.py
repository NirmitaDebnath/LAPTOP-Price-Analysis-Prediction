import streamlit as st
import pickle
import numpy as np

pipe=pickle.load(open('pipe.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))

st.title("Laptop Price Predictor")

company = st.selectbox('Brand',df['Manufacturer'].unique())

type = st.selectbox('Category',df['Category'].unique())

screen_size = st.number_input('Screen Size')

resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

ram = st.selectbox('RAM in GB',[2,4,6,8,12,16,24,32,64])

os = st.selectbox('Operating System',df['Operating System'].unique())

weight = st.number_input('Weight of the Laptop')

touchscreen = st.selectbox('Touchscreen',['No','Yes'])

ips = st.selectbox('IPS',['No','Yes'])

gpu = st.selectbox('GPU Brand',df['Gpubrand'].unique())

cpu = st.selectbox('CPU Brand',df['Cpubrand'].unique())

hdd = st.selectbox('HDD(GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(GB)',[0,128,256,512,1024])

hybrid = st.selectbox('Hybrid(GB)',[0,128,256,512,1024])

flash = st.selectbox('Flash Storage(GB)',[0,128,256,512,1024])

if st.button('Predict Price'):
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0
    if ips == 1:
        ips = 1
    else:
        ips = 0
    xres = resolution.split('x')[0]
    yres = resolution.split('x')[1]
    query=np.array([company,type,screen_size,xres,yres,ram,os,weight,touchscreen,ips,gpu,cpu,hdd,ssd,hybrid,flash])
    query = query.reshape(1,16)
    st.title("Predicted Price : "+str(int(np.exp(pipe.predict(query)[0]))))