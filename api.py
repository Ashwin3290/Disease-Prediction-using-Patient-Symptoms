from flask import Flask
from flask_restful import Api,Resource,reqparse
import pandas as pd 
import numpy as np
import pickle 
import warnings
import os
import time
warnings.filterwarnings("ignore", category=UserWarning)

forest=pickle.load(open("model_data/randomforest.pkl","rb"))
symp=list(pickle.load(open("model_data/symptoms.pkl","rb")))
disease_dict=pickle.load(open("model_data/disease_dict.pkl","rb"))

def classify(sample):
    testx=np.zeros(len(symp),dtype=int)
    for i in sample:
        if type(i)==str:
            if i in symp:
                testx[symp.index(i)]=1

    cls=forest.predict([testx])   
    return disease_dict[cls[0]]

app=Flask((__name__))
api=Api(app)

predict_Arg=reqparse.RequestParser()
predict_Arg.add_argument("symptoms",type=str,help="Send symptoms seperated by '-'")

#test json req {"symptoms":"muscle_weakness-stiff_neck-swelling_joints-movement_stiffness-painful_walking"}
class predict(Resource):
    def post(self):
        args=predict_Arg.parse_args()
        symptoms=args["symptoms"].split("-")
        
        return classify(symptoms)

api.add_resource(predict,"/predict")

if __name__=="__main__":
    app.run(debug=True)