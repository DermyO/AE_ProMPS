# AE_ProMPS

Requirement :
- You need to install tensorflow.

If you want to communicate with the Matlab module you need also:
- YARP
- To install the matlab module: https://github.com/misaki43/Multimodal-prediction-of-intention-with-ProMPs/tree/VTSFE.


DATA :
To try this software, you can use data available there: https://github.com/inria-larsen/activity-recognition-prediction-wearable/tree/master/VTSFE/data/7x10_actions

Put them in a folder "data/7x10_actions".

Information: 
The connector.py is a programm that uses YARP to communicate with a Matlab module. 

By using this Matlab module with the current Python files, you can model, generate and recognize whole-body movements. Moreover you can predict the continuation of an initiate whole-body movement.

But to use this file, you have to read and adapt them, because all the Python module is not present there.

To have more information, you can contact : oriane.dermy@gmail.com.

