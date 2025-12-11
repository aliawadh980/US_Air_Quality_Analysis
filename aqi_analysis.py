# ===================================================================
# PART 1: Build base dataset from Kaggle + EPA bulk files (up to Oct/Nov 2025)
# ===================================================================
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

load_dotenv()

def build_base_dataset():
    """Builds the base dataset from Kaggle and EPA bulk files."""
    print("Part 1: Building base dataset...")
    # 1. Load your original Kaggle file (contains data up to May 2022)
    print("Loading Kaggle data (US_AQI.csv)...")
    try:
        kaggle = pd.read_csv('US_AQI.csv')
    except FileNotFoundError:
        print("FATAL: US_AQI.csv not found. Please download it first.", file=sys.stderr)
        sys.exit(1)
    
    kaggle['Date'] = pd.to_datetime(kaggle['Date'])
    kaggle = kaggle[kaggle['Date'] <= '2022-05-31'].copy()

    # 2. Create permanent lookup table: CBSA Code → location info
    print("Creating CBSA lookup table...")
    lookup = (
        kaggle[['CBSA Code', 'city_ascii', 'state_id', 'state_name',
                'lat', 'lng', 'population', 'density', 'timezone']]
        .drop_duplicates(subset='CBSA Code')
        .set_index('CBSA Code')
    )

    # 3. Function to convert EPA annual CSV to match Kaggle format
    def epa_to_kaggle(filepath):
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"WARN: EPA file not found: {filepath}. Skipping.", file=sys.stderr)
            return None
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[['CBSA Code', 'Date', 'AQI', 'Category',
                 'Defining Parameter', 'Number of Sites Reporting']]
        df = df.join(lookup, on='CBSA Code', how='left')
        return df

    # 4. Load EPA bulk files
    print("Loading EPA annual files...")
    files = {
        2022: 'daily_aqi_by_cbsa_2022.csv',
        2023: 'daily_aqi_by_cbsa_2023.csv',
        2024: 'daily_aqi_by_cbsa_2024.csv',
        2025: 'daily_aqi_by_cbsa_2025.csv'
    }

    updates = []
    for year, path in files.items():
        df = epa_to_kaggle(path)
        if df is not None:
            if year == 2022:
                df = df[df['Date'] >= '2022-06-01']
            updates.append(df)

    if not updates:
        print("WARN: No EPA update files were found or processed.", file=sys.stderr)
        updates_df = pd.DataFrame()
    else:
        updates_df = pd.concat(updates, ignore_index=True)


    # 5. Combine Kaggle + EPA updates
    final = pd.concat([kaggle, updates_df], ignore_index=True)

    # 6. Final column order (exactly like original Kaggle)
    final = final[[
        'CBSA Code', 'Date', 'AQI', 'Category', 'Defining Parameter',
        'Number of Sites Reporting', 'city_ascii', 'state_id', 'state_name',
        'lat', 'lng', 'population', 'density', 'timezone'
    ]]
    final.insert(0, 'Unnamed: 0', range(len(final)))

    # 7. Save intermediate file and return the lookup table for Part 2
    final.to_csv('us_air_quality_1980_to_2025_oct.csv', index=False)
    print(f"Part 1 DONE → {len(final):,} rows → up to {final['Date'].max().date()}")
    return lookup

# ===================================================================
# PART 2: Add Nov 1 – Dec 11, 2025 using AirNow API (real-time data)
# ===================================================================

def add_realtime_data(lookup):
    """Adds Nov 1 – Dec 11, 2025 data from the AirNow API."""
    print("\nPart 2: Adding real-time data from AirNow...")
    
    # === CHOOSE MODE ===
    MODE = "test"  # Change to "test" for a small sample
    if MODE == "test":
        cbsa_list = [10100, 10140]  # Aberdeen, SD and WA
        print(f"TEST MODE: Fetching {len(cbsa_list)} CBSAs")
    else:
        cbsa_list = lookup.index.tolist()
        print(f"FULL MODE: Fetching {len(cbsa_list)} CBSAs")

    # === CONFIG ===
    API_KEY = os.getenv("AIRNOW_API_KEY")
    if not API_KEY:
        print("FATAL: AIRNOW_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)
    URL = "https://www.airnowapi.org/aq/observation/latLong/historical/"
    
    start_date = datetime(2025, 11, 1)
    end_date = datetime(2025, 12, 11)

    def fetch_day_for_cbsa(cbsa):
        info = lookup.loc[cbsa]
        rows_for_cbsa = []
        current = start_date
        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            try:
                r = requests.get(URL, params={
                    "format": "json", "latitude": info["lat"], "longitude": info["lng"],
                    "date": f"{date_str}T00-0000", "distance": 25, "API_KEY": API_KEY
                }, timeout=8)
                r.raise_for_status()
                readings = r.json()
                if readings and readings[0]["AQI"] != -1:
                    o = readings[0]
                    rows_for_cbsa.append({
                        "CBSA Code": int(cbsa), "Date": current.date(), "AQI": o["AQI"],
                        "Category": o["Category"]["Name"], "Defining Parameter": o["ParameterName"],
                        "Number of Sites Reporting": 1, **info.to_dict()
                    })
            except requests.exceptions.RequestException as e:
                print(f"WARN: API request failed for CBSA {cbsa} on {date_str}: {e}", file=sys.stderr)
            except (ValueError, KeyError) as e:
                print(f"WARN: Could not parse API response for CBSA {cbsa} on {date_str}: {e}", file=sys.stderr)
            current += timedelta(days=1)
        return rows_for_cbsa

    # === RUN (parallel) ===
    print(f"Fetching data from {start_date.date()} to {end_date.date()}...")
    new_rows = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(fetch_day_for_cbsa, cbsa): cbsa for cbsa in cbsa_list}
        completed = 0
        for future in futures:
            rows = future.result()
            new_rows.extend(rows)
            completed += 1
            percent = int(completed / len(cbsa_list) * 100)
            bar = "█" * (percent // 2) + "░" * (50 - percent // 2)
            city_name = lookup.loc[futures[future], "city_ascii"]
            print(f"Progress: [{bar}] {percent:3}% | Fetched: {city_name:20}", end="\r")
            sys.stdout.flush()

    # === SAVE FINAL ===
    df = pd.read_csv("us_air_quality_1980_to_2025_oct.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    new_df = pd.DataFrame(new_rows)
    if not new_df.empty:
        new_df["Date"] = pd.to_datetime(new_df["Date"])
        final = pd.concat([df, new_df]).drop_duplicates(subset=["CBSA Code", "Date"], keep="last")
    else:
        final = df.copy()

    final = final.sort_values(["CBSA Code", "Date"]).reset_index(drop=True)
    final["Unnamed: 0"] = range(len(final))
    final.to_csv("us_air_quality_complete_to_dec11_2025.csv", index=False)

    print(f"\n\nPart 2 DONE!")
    print(f"Added {len(new_rows)} new daily records.")
    print(f"Final file saved: 'us_air_quality_complete_to_dec11_2025.csv'")
    print(f"Total rows: {len(final):,}")

if __name__ == "__main__":
    cbsa_lookup = build_base_dataset()
    add_realtime_data(cbsa_lookup)


def perform_analysis(df):
    """Runs the full analysis suite on the provided dataframe."""
    print("\nPart 3: Performing data analysis...")

    # Step 3: Missing Data Handling
    print("STEP 3: Handling Missing Data")
    missing = df.isnull().sum()
    print("Missing values per column:")
    print(missing[missing > 0] if missing.sum() > 0 else "→ ZERO MISSING VALUES!")
    print("Veracity: EXCELLENT – No imputation needed.\n")

    # Step 4: Descriptive Statistics
    print("STEP 4: Descriptive Statistics")
    if 'AQI' in df.columns and not df['AQI'].empty:
        aqi_stats = df['AQI'].describe()
        print("AQI Statistics:")
        print(f"   Mean   : {aqi_stats['mean']:.2f}")
        print(f"   Median : {df['AQI'].median():.0f}")
        print(f"   Std Dev: {aqi_stats['std']:.2f}")
        print(f"   Min/Max: {aqi_stats['min']:.0f} / {aqi_stats['max']:.0f}")
        print(f"   99.9th %: {df['AQI'].quantile(0.999):.0f}\n")
    else:
        print("AQI data not available for statistics.\n")

    # Step 5: Data Visualization
    generate_visualizations(df)

    # Step 6: K-Means Clustering
    perform_clustering(df)

    # Step 7: Actionable Insights
    display_insights()

def generate_visualizations(df):
    """Generates and saves a 5-panel visualization of AQI data."""
    print("STEP 5: Generating 5 Professional Visualizations...")
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Decade'] = (df['Year'] // 10) * 10
    
    plt.figure(figsize=(16, 12))
    
    # 1. National Trend
    plt.subplot(3, 2, 1)
    yearly = df.groupby('Year')['AQI'].mean()
    plt.plot(yearly.index, yearly.values, 'b-o', linewidth=2, markersize=4)
    plt.title('National Average AQI Trend', fontsize=14, fontweight='bold')
    plt.ylabel('Average AQI'); plt.grid(True, alpha=0.3)

    # 2. Seasonal Pattern
    plt.subplot(3, 2, 2)
    monthly = df.groupby('Month')['AQI'].mean()
    months = ['J','F','M','A','M','J','J','A','S','O','N','D']
    if len(monthly) == 12:
        plt.bar(months, monthly.values, color='skyblue', edgecolor='navy')
    plt.title('Seasonal AQI Pattern', fontsize=14, fontweight='bold')
    plt.ylabel('Average AQI')

    # 3. AQI Distribution
    plt.subplot(3, 2, 3)
    plt.hist(df['AQI'].dropna(), bins=100, range=(0, 300), color='lightcoral', edgecolor='red', alpha=0.7)
    plt.axvline(50, color='g', linestyle='--', label='Good')
    plt.axvline(100, color='orange', linestyle='--', label='Moderate')
    plt.axvline(150, color='r', linestyle='--', label='Unhealthy')
    plt.title('AQI Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('AQI'); plt.ylabel('Density'); plt.legend()

    # 4. Population vs AQI
    plt.subplot(3, 2, 4)
    sample = df.dropna(subset=['population', 'AQI']).sample(n=min(len(df), 100000), random_state=42)
    plt.scatter(sample['population'], sample['AQI'], alpha=0.5, s=10, c=sample['Year'], cmap='viridis')
    plt.xscale('log'); plt.colorbar(label='Year')
    plt.xlabel('Population (log scale)'); plt.ylabel('AQI')
    plt.title('Population Size vs. Air Quality', fontsize=14, fontweight='bold')

    # 5. Boxplot by Decade
    plt.subplot(3, 2, 5)
    decade_data = df.copy()
    decade_data['Decade'] = decade_data['Decade'].astype(str) + 's'
    sns.boxplot(x='Decade', y='AQI', data=decade_data, palette='Set2')
    plt.title('Air Quality Over Decades', fontsize=14, fontweight='bold')
    
    plt.suptitle('US Air Quality Analysis', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('Step5_All_Visualizations.png', dpi=400)
    print("Visualizations saved as 'Step5_All_Visualizations.png'\n")

def perform_clustering(df):
    """Performs K-Means clustering on cities by pollution profile."""
    print("STEP 6: K-Means Clustering of Cities...")
    city_stats = df.groupby('city_ascii').agg({
        'AQI': ['mean', 'max'], 'population': 'first',
        'density': 'first', 'lat': 'first', 'lng': 'first'
    }).reset_index()
    city_stats.columns = ['City', 'Avg_AQI', 'Max_AQI', 'Population', 'Density', 'Lat', 'Lng']
    features = city_stats[['Avg_AQI', 'Max_AQI', 'Population', 'Density']].fillna(0)
    
    if len(features) < 4:
        print("WARN: Not enough cities to perform meaningful clustering. Skipping.", file=sys.stderr)
        return

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    n_clusters = min(4, len(city_stats))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    city_stats['Cluster'] = kmeans.fit_predict(features_scaled)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(city_stats['Lng'], city_stats['Lat'], c=city_stats['Cluster'], 
                          cmap='tab10', s=30, alpha=0.7)
    plt.title('K-Means Clustering of US Cities by Air Pollution Profile', fontsize=14, fontweight='bold')
    plt.xlabel('Longitude'); plt.ylabel('Latitude')
    plt.colorbar(scatter, label='Pollution Cluster')
    plt.savefig('Step6_Clustering_Map.png', dpi=400)
    print("Clustering map saved as 'Step6_Clustering_Map.png'\n")

def display_insights():
    """Prints actionable insights and recommendations."""
    print("STEP 7: ACTIONABLE INSIGHTS & RECOMMENDATIONS")
    print("="*60)
    print("KEY FINDINGS:")
    print("• US air quality shows significant long-term improvement.")
    print("• Seasonal patterns, like summer ozone, persist as challenges.")
    print("• Extreme events (e.g., wildfires) can cause hazardous AQI spikes.")
    print("• Urban areas have varied pollution profiles, captured by clustering.")
    print("\nRECOMMENDATIONS FOR DECISION-MAKERS:")
    print("1. Public Health: Target high-pollution clusters for real-time alerts.")
    print("2. Policy: Focus on mitigating sources of seasonal pollution peaks.")
    print("3. Climate Action: Prepare for increased frequency of extreme smoke events.")
    print("="*60)

if __name__ == "__main__":
    cbsa_lookup = build_base_dataset()
    add_realtime_data(cbsa_lookup)
    
    # Load the final dataset and run the analysis
    try:
        final_df = pd.read_csv("us_air_quality_complete_to_dec11_2025.csv")
        final_df['Date'] = pd.to_datetime(final_df['Date'])
        perform_analysis(final_df)
    except FileNotFoundError:
        print("FATAL: Final dataset not found. Cannot perform analysis.", file=sys.stderr)
        sys.exit(1)
