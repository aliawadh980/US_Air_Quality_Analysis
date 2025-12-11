# US Air Quality Analysis (1980-2025)

This project provides a comprehensive script to aggregate, analyze, and visualize historical and real-time air quality data for the United States. It combines data from a historical Kaggle dataset with more recent bulk data from the EPA and real-time data from the AirNow API.

The script is designed to be a complete end-to-end pipeline: it fetches and cleans the data, merges it into a master dataset, and then performs a detailed analysis, including generating visualizations and clustering cities by their pollution profiles.

## Features

- **Data Aggregation:** Combines multiple data sources into a single, clean dataset.
- **Real-time Data:** Fetches up-to-date daily AQI readings from the AirNow API.
- **Comprehensive Analysis:** Calculates descriptive statistics and identifies trends.
- **Professional Visualizations:** Generates a multi-panel plot with five key charts.
- **Machine Learning:** Uses K-Means clustering to identify cities with similar pollution patterns.
- **Secure & Modular:** Loads API keys securely from a `.env` file and is written in a modular, easy-to-read style.

## Prerequisites

Before you begin, you will need to have **Python 3** installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

## How to Use This Project: A Step-by-Step Guide

This guide will walk you through everything you need to do to run this project, from setup to execution.

### Step 1: Get the Code

First, you need to get a copy of the project on your local machine. You can do this by cloning the repository using Git:

```bash
git clone <repository_url>
cd <repository_name>
```
*(Replace `<repository_url>` and `<repository_name>` with the actual URL and name of this project's repository.)*

### Step 2: Set Up a Virtual Environment

It is highly recommended to use a virtual environment to keep the project's dependencies isolated from your system's Python installation.

**Create the environment:**
```bash
python3 -m venv venv
```

**Activate the environment:**
- On **macOS and Linux**:
  ```bash
  source venv/bin/activate
  ```
- On **Windows**:
  ```bash
  .\\venv\\Scripts\\activate
  ```
Your terminal prompt should now show `(venv)` at the beginning, indicating that the virtual environment is active.

### Step 3: Install the Required Packages

All the necessary Python libraries are listed in the `requirements.txt` file. Install them all with a single command:

```bash
pip install -r requirements.txt
```

### Step 4: Get Your API Key

This project uses the [AirNow API](https://docs.airnowapi.org/) to fetch recent air quality data. You will need a free API key to run the script.

1.  Go to the AirNow API website and register for an account.
2.  Once you receive your API key via email, copy it.
3.  In the project directory, rename the `.env.example` file to `.env`.
4.  Open the `.env` file and paste your API key in place of `"your_api_key_here"`.

The file should look like this:
```
# Rename this file to .env and replace the placeholder with your actual AirNow API key.
AIRNOW_API_KEY="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
```

### Step 5: Download the Data Files

The script requires several historical data files to build the base dataset. You will need to download these and place them in the root of the project directory.

**Required Files:**
- `US_AQI.csv`
- `daily_aqi_by_cbsa_2022.csv`
- `daily_aqi_by_cbsa_2023.csv`
- `daily_aqi_by_cbsa_2024.csv`
- `daily_aqi_by_cbsa_2025.csv`

*(Note: Please ensure you have obtained these files from the appropriate sources, such as Kaggle or the EPA website.)*

### Step 6: Run the Script!

You are now ready to run the analysis. From the root of the project directory (with your virtual environment still active), simply run the following command:

```bash
python aqi_analysis.py
```

The script will print its progress to the console. It will first build the historical dataset, then fetch the real-time data, and finally perform the analysis.

### Step 7: View the Output

Once the script is finished, it will generate two new files in the project directory:

- `Step5_All_Visualizations.png`: A high-resolution image containing five different charts that analyze the air quality data.
- `Step6_Clustering_Map.png`: A map of the US showing cities clustered by their pollution profiles.

You can open these image files to see the results of the analysis. The script also saves two intermediate and final CSV files, but these are primarily for processing and are ignored by Git.

## Project Structure

- `aqi_analysis.py`: The main Python script that contains all the logic for the project.
- `requirements.txt`: A list of all the Python packages required to run the script.
- `.env.example`: A template file for your environment variables. You will rename this to `.env` and add your API key.
- `.gitignore`: A file that tells Git which files to ignore (like `venv` and `.env`).
- `README.md`: This file!
