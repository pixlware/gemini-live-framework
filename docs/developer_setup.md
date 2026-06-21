# Developer Setup Guide

This guide describes how to configure your local development environment and verify connection/authentication with Google Cloud Vertex AI to run the Gemini Live Voice Bot.

## Prerequisites

- **Python 3.12+**
- **Google Cloud SDK** installed and configured on your machine.
- **GCP Project** with the Vertex AI API enabled.

## Step 1: Clone and Set Up Python Virtual Environment

1. Navigate to the project root:
   ```bash
   cd gemini-live-framework
   ```

2. Create a virtual environment using Python 3.12+:
   ```bash
   python3 -m venv venv
   ```

3. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```cmd
     venv\Scripts\activate
     ```

4. Install the required dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Step 2: Configure Environment Variables

1. Copy the environment template to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` in your editor and configure `GOOGLE_CLOUD_PROJECT` with your GCP Project ID:
   ```env
   GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
   ```

## Step 3: Configure Google Cloud Authentication

The Google GenAI SDK authenticates with Vertex AI using standard Google Cloud mechanisms. You have two options for local development:

### Option A: Application Default Credentials (ADC) — Recommended
This is the preferred approach for local development.

1. Ensure the Google Cloud SDK is installed and run:
   ```bash
   gcloud auth application-default login
   ```
2. Follow the browser prompts to log in. This stores credentials locally that the Google GenAI library will auto-detect.

### Option B: Service Account Key File
If you have a dedicated service account:

1. Download the service account key in JSON format.
2. Store it securely (do not commit it to git!).
3. Update the path to the key in `.env`:
   ```env
   GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
   ```

## Step 4: Verify the Setup

To verify that the local environment and GCP authentication are correctly configured:

1. Ensure your virtual environment is active and `.env` has your project ID.
2. Run the application locally using Uvicorn:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
3. Check the startup logs to ensure:
   - FastAPI initialized on `http://0.0.0.0:8000`.
   - Logging shows `Starting Gemini Live Framework`.
   - No authentication errors are thrown on startup.
