# IBM Cloud & Watson Machine Learning Deployment Guide

This guide details how to deploy the Credit Card Approval Application to **IBM Cloud** using **Docker**, **IBM Cloud Foundry**, or **IBM Watson Machine Learning (WML)**.

---

## 1. Deploying the ML Model to IBM Watson WML

The application includes `deployment.py` to automate importing and deploying the trained classification pipeline onto an IBM WML instance.

### Prerequisites
1. An active [IBM Cloud Account](https://cloud.ibm.com).
2. A **Watson Machine Learning** service instance provisioned.
3. A **Deployment Space** created in your IBM Cloud catalog.

### Step-by-Step Model Upload
1. Retrieve your IBM Cloud API Key:
   - Go to **Manage** > **Access (IAM)** > **API Keys**.
   - Click **Create an IBM Cloud API Key** and save it.
2. Retrieve your WML Deployment Space ID:
   - Open your Deployment Space on IBM Cloud.
   - Go to the **Manage** tab and copy the **Space GUID**.
3. Set your environment variables in your command prompt:
   ```bash
   # Linux/macOS
   export IBM_CLOUD_API_KEY="your_api_key_here"
   export IBM_WML_SPACE_ID="your_space_guid_here"
   export IBM_WML_URL="https://us-south.ml.cloud.ibm.com" # Adjust region if needed

   # Windows (PowerShell)
   $env:IBM_CLOUD_API_KEY="your_api_key_here"
   $env:IBM_WML_SPACE_ID="your_space_guid_here"
   $env:IBM_WML_URL="https://us-south.ml.cloud.ibm.com"
   ```
4. Run the upload script:
   ```bash
   python deployment.py deploy
   ```
This stores your Scikit-Learn pipeline in the WML repository and provisions a secure web scoring endpoint.

---

## 2. Containerized Deployment to IBM Cloud Code Engine

IBM Cloud Code Engine is a fully managed, serverless platform that runs containerized workloads.

### Prerequisites
1. Install the [IBM Cloud CLI](https://cloud.ibm.com/docs/cli).
2. Install the Container Registry plugin:
   ```bash
   ibmcloud plugin install container-registry
   ibmcloud plugin install code-engine
   ```

### Deployment Steps
1. **Login to IBM Cloud CLI**:
   ```bash
   ibmcloud login --sso
   ```
2. **Target Container Registry Region** (e.g., US South):
   ```bash
   ibmcloud cr region-set us-south
   ```
3. **Create a Namespace**:
   ```bash
   ibmcloud cr namespace-add credit-card-apps
   ```
4. **Build and Tag the Docker Image**:
   ```bash
   docker build -t us.icr.io/credit-card-apps/approval-app:v1 .
   ```
5. **Log Docker into IBM Container Registry**:
   ```bash
   ibmcloud cr login
   ```
6. **Push Image to Registry**:
   ```bash
   docker push us.icr.io/credit-card-apps/approval-app:v1
   ```
7. **Deploy to Code Engine**:
   ```bash
   ibmcloud ce project create --name CreditAnalytics
   ibmcloud ce project select --name CreditAnalytics
   
   ibmcloud ce application create --name credit-card-predictor \
     --image us.icr.io/credit-card-apps/approval-app:v1 \
     --port 5000 \
     --env FLASK_SECRET_KEY="A_Strong_Random_Secret_Key_For_Session"
   ```
8. Save the generated application URL to access the live web application.

---

## 3. Deployment using IBM Cloud Foundry (Legacy / PaaS)

To deploy as a traditional cloud application using buildpacks:

1. Ensure `Procfile`, `runtime.txt`, and `requirements.txt` are in the project root.
2. Edit or create a `manifest.yml` file in the root:
   ```yaml
   ---
   applications:
   - name: credit-card-approval-predictor
     memory: 512M
     instances: 1
     buildpacks:
     - python_buildpack
     env:
       FLASK_SECRET_KEY: "change_me_to_something_secure"
       FLASK_ENV: "production"
   ```
3. Deploy the application:
   ```bash
   ibmcloud cf push
   ```
