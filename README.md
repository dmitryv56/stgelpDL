# stgelpDL
This project "Short-Term (Green) Energy Load Prediction with using Deep Learning (DL) and Statistical Time Series Analysis (TSA) methods", hereinafter "stgelpDL", 
is aimed  as short-term forecasting of the load on electrical grig, where 30-40% of energy is generated from renewable energy sources, i.e. Green Energy (GE). 

In the GE environment description we follow by  O.Novykh, I.Sviridenko,J.A.Mendez Perez, B. Gonzalez-Diaz "Mejorando la eficiencia 
energetica de las ciudades intelligentes a traves del amplio uso de dispositivos de almacenamiento de Energia", IV CONGRESSO 
CIUDADES INTELLIGENTES, 2018/5, O. Novykh,J.A.Mendez Perez, B.Gonzalez-Diaz , I. Sviridenko "Performance analysis of hybrid 
hydroelectric Gorona delViento and the basic directions of its perfections", 17 International Conference on Renewable Energiesand Power 
Quality (ICREPQ`19),  Tenerife, Spain, April, 2019. 

In the algorithm implementations we follow by T.W. Anderson "The Statistical Analysis of Time Series",J.Wilej & Sons, 2011, 
J.Brownlee "Predict the Future with MLPs,CNNs and LSTMs in Python", 2018 ,  Google' Tensorflow Guide https://www.tensorflow.org/tutorials/,
Smith, Taylor G., et al. pmdarima: ARIMA estimators for Python, 2017-, http://www.alkaline-ml.com/pmdarima [Online; accessed 2020-09-07].

RED Electrical De Espana (https://www.ree.es/en/apidatos) provides a simple REST service to allow third parties to access the backend data used 
in the REData application. 


The development framework is PyCharm Community 2020.1 (Windows 10 or Ubuntu 18)

The project setup is :
Python3.6
tensorflow-cpu
pmdarima
numpy
matplotlib
dundas

The project is developing now under MIT License.

"stgelpDL" use a conceptual models called "planes". Planes describe a data sources, how to train modeles (DL and STS), how to predict,how to control on dataflow 
in the predictor.""stgelpDL" has Control Plane (CP), Training Plane(TP),Predict Plane(PP), Management Plane(MP).
1. The functions and services of the Control Plane (CP)
• define data source( dataset) and their features (Time Series) than should be predicted .
• connect to datasource (DS). Now statical (SDS) and dynamical datasources (DDS) are partially supported.  
• split the data to the training, evaluation and test sequences.
• set DL and  (TSA) models and their structures.
For DL there are neuron layer types and their combinations with nonlinear . For TSA modeles there is order of autoregression or middle average.
• perform pimary statistical estimation of TS.
• serve the model repository (MR)
• set the normal energy load process limits, upon reaching which an event  is raised
SDS that has being supported now there is csv-file.
DDS that has being supported now there is REData API of https://www.ree.es/en/apidatos
2. The function and services of the Training Plane (TP)
• solve the parametric identification problem for specified foreasting modeles models. The linear identification for TSA and nonlinear for DL .
• store the trained models in MR both structures and nonlinear elements and weight tensors.
3. The Predict Plane (PP) functions
• load the specified models from MR
• load short history of TS Y(t-k), Y(t-k+1), ..., Y(t-1), Y(t)
• predict feature values of TS Yp(t+1), Yp(+2), ..., Yp(t+l)
• raise the event on reaching limits of the normal energy load process
4. The Management Plane (MP) controls the flow of the data and of the signals in the predictor. It implemented as Observer pattern. 

The project is in developing state and some changes are  possible.

NOTES:
stgelpDL v.1.0.0 published at 17/09/2020
stgelpDL v.2.0.1 published at 27/09/2020

RELEASE NOTES

In Development
- Improving structure of logs.

2.0.1.
New Features
- instead of '[Real demand]' time series the delta '[Programmed demand] - [Real demand]' time series is predicting.
- command-line arguments added.
- stand-alone script for run the predictor added.
Bug Fixes
- exception in seasonal ARIMA model fixed.

1.0.0.
New Features
- predictor runs in 'auro'- mode.











References:  dmitryv56@walla.com, alexandr.novykh@gmail.com
