import bentoml
from bentoml.io import JSON
import mlflow.sklearn

# Load model artifact from MLflow
model = mlflow.sklearn.load_model("runs:/<RUN_ID>/model")

@bentoml.env(auto_pip_dependencies=True)
@bentoml.artifacts([bentoml.SklearnModelArtifact('model')])
class IrisService(bentoml.BentoService):
    @bentoml.api(input=JSON(), output=JSON())
    def predict(self, parsed_json):
        data = parsed_json["data"]
        preds = self.artifacts.model.predict(data)
        return {"predictions": preds.tolist()}

svc = IrisService()