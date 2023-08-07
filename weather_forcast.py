import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

df = pd.read_csv('weather with date.csv')
met = pd.read_csv('meta_data.csv')
df1 = pd.read_csv('data_weather.csv')

def home():
	st.title("WEARTHER ANALYSIS")
	st.divider()
	st.caption(""":red[**_Weather_** forecasting] begins with an analysis of the current state of the temperature, wind speed, humidity, visibility and atmosphere pressure. Reliable observations drawn from many platforms, including satellites, radar, weather balloons, surface stations, and aircraft are crucial for generating accurate analyses
	Human feedback is also required to choose the best possible forecast model on which to base the forecast
	Every day the weather is recorded by the meteorologists and these records are preserved for decades
""")


def data():
	col1, col2, col3 = st.columns(3)
	with col1:
		c1=st.button('Dateset info')
		if c1:
			st.write(df1.head(10))
	with col2:
		c2=st.button("Describe")
		if c2:
			st.write(df1.describe())
	with col3:
		c3=st.button("Meta Data")
		if c3:
			st.caption("Metadata is data that describes or gives information about this data.")
			st.write(met)

def per():
	b,c,d= st.tabs(["Date wise","Weather wise","bar plot"])

	with b:
		options=st.selectbox('select the variable',["Temp_C","Dew Point Temp_C","Rel Hum_%","Wind Speed_km/h","Visibility_km","Press_kPa","Weather"])
		date=st.selectbox('select the variable',["month","day",'year'])
		#ddd=st.slider('select lessthan 75 days range for better view:',1,100,(25,75))

		#ddd=(1,2,3,4,5,6,7,8,9,10,11,12)
		fig = plt.figure()
		if(options=="Weather"):
			plt.plot(df[date],df[options],'s')
		else:
			sns.lineplot(df[date],df[options])
		#plt.xticks(rotation='vertical')
		#plt.xticks(ddd)
		#plt.xlim(ddd)
		plt.xlabel('month')
		plt.ylabel(options)
		st.pyplot(fig)

	with c:
		options1=st.selectbox('select the 1st variable',["Temp_C","Dew Point Temp_C","Rel Hum_%","Wind Speed_km/h","Visibility_km","Press_kPa"])

		fig = plt.figure()
		ax = fig.add_axes([.1,.1,.8,.8])	
		ax1 = ax.plot(df[options1],df['Weather'],'s',color='red')
		plt.xlabel(options1)

				
		st.pyplot(fig)
	with d:
		opt=st.selectbox('select variable',["Temp_C","Dew Point Temp_C","Rel Hum_%","Wind Speed_km/h","Visibility_km","Press_kPa"])
		fig=plt.figure()
		plt.bar(df.Weather,df[opt])
		plt.xticks(rotation='vertical')
		plt.ylabel(opt)
		st.pyplot(fig)
	#with e:

	
			
def fore():		
	x = df1.iloc[:,1:7]
	y = df1.iloc[:,-1]
	x_train1, x_test1, y_train1, y_test1 = train_test_split(x,y,train_size=0.99)

	def user_report():
		Temp_C = st.sidebar.slider('Temp_C', -23.30,33.00, 3.00 )
		Dew_Point_Temp_C = st.sidebar.slider('Dew Point Temp_C', -28.50,24.40, -0.20 )
		Rel_Hum_ = st.sidebar.slider('Rel Hum_%', 18.00,100.00, 82.00 )
		Wind_Speed_kmh = st.sidebar.slider('Wind Speed_km/h', 0.00,83.00, 13.00 )
		Visibility_km = st.sidebar.slider('Visibility_km', 2.40,48.30, 12.90 )
		Press_kPa = st.sidebar.slider('Press_kPa', 97.52,103.65, 99.93 )
		


		user_report_data = {
			'Temp_C':Temp_C,
			'Dew Point Temp_C':Dew_Point_Temp_C,
			'Rel Hum_%':Rel_Hum_,
			'Wind Speed_km/h':Wind_Speed_kmh,
			'Visibility_km':Visibility_km,
			'Press_kPa':Press_kPa,
		
		}
		report_data = pd.DataFrame(user_report_data, index=[0])
		return report_data

	user_data = user_report()
	st.subheader('Weather Attribute')
	st.write(user_data)

	x_train=x_train1.values
	y_train=y_train1.values

	rf  = RandomForestClassifier()
	rf.fit(x_train, y_train)
	user_result = rf.predict(user_data)

	st.subheader('Weather Report: ')
	st.write(user_result[0])
	output=''
	loc='C:/Users/AssassiN/ddd/img/'
	image=Image.open(loc+user_result[0]+'.jpg')
	st.image(image)

	
	st.title(output)
	st.write(str(accuracy_score(y_test1,rf.predict(x_test1))*100)+'%')

def rep():
	st.title("Conclusion")
	st.write("""Generally, all the other weather forecasting applications and sources would give the weather report of that particular area.
	 Using the previously recorded information, weather condition would be given at that place. The types of clouds the author considered, 
	 can deliberately give accurate condition of the weather. For now, the model can give the weather condition at that point of time.""")
	st.title("Future Work")
	st.write("""To get weather forecast for the next few days, we can modify our system by using different algorithms and use that as an 
	extension to our current project.""")
	st.title("Book Reference")
	st.write("Deep Learning by Ian GoodFellow, Yoshuna Bengio and Aaron Courville")
	st.write("Beginner's Guide to Streamlit with Python by Sujay Raghavendra")
	st.title("Web Reference")
	st.write("https://www.geeksforgeeks.org/random-forest-classifier-using-scikit-learn/")
	st.write("https://docs.streamlit.io/")

st.sidebar.title('Navigation')
side = st.sidebar.radio('Select what you want to display:', ['Home','Data set info', "Forecasting", 'graphs', "Conclusion"])

if side == 'Home':
 	home()
elif side == 'Data set info':
 	data()
elif side == 'graphs':
 	per()
elif side == 'Forecasting':  
	fore()
elif side == 'Conclusion':
 	rep()