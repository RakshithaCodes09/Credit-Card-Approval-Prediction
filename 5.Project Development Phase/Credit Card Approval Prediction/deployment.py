import os
import sys
import logging
from ibm_watson_machine_learning import APIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deploy_to_watson():
    """
    Demonstrates deploying the trained credit card approval model to IBM Watson Machine Learning.
    Loads credentials safely from environment variables and deploys the model pipeline.
    """
    api_key = os.environ.get("IBM_CLOUD_API_KEY")
    wml_url = os.environ.get("IBM_WML_URL", "https://us-south.ml.cloud.ibm.com")
    space_id = os.environ.get("IBM_WML_SPACE_ID")

    if not api_key or not space_id:
        logger.error("Missing deployment configuration. Please set IBM_CLOUD_API_KEY and IBM_WML_SPACE_ID environment variables.")
        print("\n[!] Setup Guide: To deploy to IBM Cloud, set these variables in your shell:")
        print("    export IBM_CLOUD_API_KEY='your_api_key'")
        print("    export IBM_WML_SPACE_ID='your_deployment_space_id'")
        print("    export IBM_WML_URL='https://us-south.ml.cloud.ibm.com'\n")
        return False

    wml_credentials = {
        "url": wml_url,
        "apikey": api_key
    }

    try:
        logger.info("Initializing Watson Machine Learning client...")
        client = APIClient(wml_credentials)
        client.set.default_space(space_id)
        
        # 1. Define Model Meta Properties
        model_name = "Credit_Card_Approval_Gradient_Boosting"
        sofware_spec_uid = client.software_specifications.get_id_by_name("runtime-22.2-py3.10") # Or appropriate Python runtime
        
        metadata = {
            client.repository.ModelMetaNames.NAME: model_name,
            client.repository.ModelMetaNames.TYPE: "scikit-learn_1.1", # adjust based on library version
            client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sofware_spec_uid
        }

        # 2. Store model pipeline to IBM repository
        logger.info(f"Storing model: {model_name} in deployment space...")
        model_artifact_path = "model/model_pipeline.joblib"
        
        if not os.path.exists(model_artifact_path):
            logger.error("Trained model pipeline file not found! Train model locally first.")
            return False
            
        stored_model_details = client.repository.store_model(
            model=model_artifact_path,
            meta_props=metadata
        )
        
        model_uid = client.repository.get_model_id(stored_model_details)
        logger.info(f"Successfully stored model. Model UID: {model_uid}")

        # 3. Create Web Service Deployment
        logger.info("Creating online deployment (web service)...")
        deploy_meta = {
            client.deployments.ConfigurationMetaNames.NAME: f"{model_name}_Online",
            client.deployments.ConfigurationMetaNames.ONLINE: {}
        }
        
        deployment_details = client.deployments.create(
            model_uid=model_uid,
            meta_props=deploy_meta
        )
        
        deployment_uid = client.deployments.get_id(deployment_details)
        logger.info(f"Successfully created deployment. Deployment UID: {deployment_uid}")
        print(f"\n[+] Watson Machine Learning Deployment successful!")
        print(f"    Deployment ID: {deployment_uid}")
        print(f"    Scoring Endpoint: {client.deployments.get_scoring_href(deployment_details)}\n")
        return True

    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        return False

def score_watson_endpoint(scoring_payload):
    """
    Demonstrates scoring an active Watson Machine Learning endpoint.
    """
    api_key = os.environ.get("IBM_CLOUD_API_KEY")
    wml_url = os.environ.get("IBM_WML_URL", "https://us-south.ml.cloud.ibm.com")
    deployment_id = os.environ.get("IBM_WML_DEPLOYMENT_ID")

    if not all([api_key, deployment_id]):
        logger.error("Missing scoring configuration. Set IBM_CLOUD_API_KEY and IBM_WML_DEPLOYMENT_ID.")
        return None

    wml_credentials = {"url": wml_url, "apikey": api_key}
    client = APIClient(wml_credentials)
    
    # Payload format required by Watson ML:
    # {"input_data": [{"fields": [...], "values": [[...]]}]}
    
    logger.info(f"Sending scoring request to deployment: {deployment_id}")
    predictions = client.deployments.score(deployment_id, scoring_payload)
    return predictions

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "deploy":
        deploy_to_watson()
    else:
        print("Credit Card Approval IBM Watson ML deployment script.")
        print("Usage: python deployment.py deploy")
