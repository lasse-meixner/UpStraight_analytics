import coremltools
import click
import os
import sys
import joblib

sys.path.append("../")
from UpStraight_Train import training_columns


@click.command()
@click.option("--user", default="cr", help="User to select model for")
def convert_model(user):
    # check if a trained model exists in models directory 
    if os.path.exists(user+"_model.pkl"):
        model = joblib.load(user+"_model.pkl")
        # extract the last estimator from the pipeline
        classifier = model.named_steps["clf"]
        # convert to coreml
        coreml_model = coremltools.converters.sklearn.convert(classifier)
        # save model
        coreml_model.save(user+"_model.mlmodel")
    else:
        raise FileNotFoundError("No model found for user "+user)

if __name__ == "__main__":
    convert_model()