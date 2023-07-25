
from fastapi import FastAPI
#from mangum import Mangum
from fastapi.responses import JSONResponse
import uvicorn
from mlflow.pyfunc import load_model
import mlflow
from databricks.sdk import WorkspaceClient
from mlflow import MlflowClient
import numpy as np
import os


os.environ['DATABRICKS_HOST'] = "DB_URL"
os.environ['DATABRICKS_TOKEN'] = "DB_TOKEN"

w = WorkspaceClient(
  host  = os.environ.get('DATABRICKS_HOST'),
  token = os.environ.get('DATABRICKS_TOKEN')
)

mlflow.set_tracking_uri('databricks')

for mv in w.model_registry.search_model_versions(filter="name='sk-learn-logistic-reg-model'"):
                                          #dic = dict(mv)  
                                          dic = mv.__dict__
                                          #print(dic['run_id'])
                                          if dic['current_stage']=='Production':
                                                   d = w.experiments.get_run(run_id=dic['run_id']).run
                                                   #print(d.data.metrics) ---------->metrics
                                                   print(dic['source'])
                                                   model_uri = dic['source']
                                                   my_model = load_model(model_uri)

app = FastAPI()
#handler = Mangum(app)

@app.get("/")
def read_root():
   return {"Welcome to": "My first FastAPI depolyment using Docker image"}

@app.get("/{text}")
def read_item(text: str):
   return JSONResponse({"result": text})



@app.get("/invocations/{arr}")
def predict(arr: list):
                 ans = my_model.predict(np.array(arr).reshape(1,4))

                 return {"status":200, "prediction":ans}
       

if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8080)

