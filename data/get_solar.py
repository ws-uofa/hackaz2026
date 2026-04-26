import requests
import pandas as pd
import time

# 1. Setup
API_KEY = "uhBjwCtuSkCYJ0RpADCLuXprAASxdR66m7heTcpn"
EMAIL = "kevinburns6711@gmail.com"

# Moved the API_KEY into the URL string to force authentication
BASE_URL = f"https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-conus-v4-0-0-download.json?api_key={API_KEY}"

# Tucson Metropolitan Area Point IDs
TUCSON_POINTS = ['563602,563603,564805,564806']

def main():
    # Note: we removed 'api_key' from this dictionary because it's now in the URL
    input_data = {
        'attributes': 'air_temperature,ghi',
        'interval': '60',
        'email': EMAIL,
        'utc': 'true',
        'full_name': 'Kevin Burns',
        'reason': 'Academic',
        'affiliation': 'University of Arizona'
    }
    
    for name in ['2023']:
        print(f"Processing Year: {name}")
        for id, location_ids in enumerate(TUCSON_POINTS):
            input_data['names'] = [name]
            input_data['location_ids'] = location_ids
            
            print(f'Sending authenticated request for Tucson points...')
            
            # Using POST as per their original script structure
            response = requests.post(BASE_URL, data=input_data)
            
            # Handle response using their provided error handling logic
            data = get_response_json_and_handle_errors(response)
            
            download_url = data['outputs']['downloadUrl']
            print(f"\n✅ SUCCESS: {data['outputs']['message']}")
            print(f"🔗 DOWNLOAD LINK: {download_url}")

def get_response_json_and_handle_errors(response: requests.Response) -> dict:
    if response.status_code != 200:
        # This will now show us exactly what the server says if it still fails
        print(f"An error has occurred. Status: {response.status_code}")
        print(f"Server Message: {response.text}")
        exit(1)
    try:
        response_json = response.json()
    except:
        print(f"Couldn't parse JSON response: {response.text}")
        exit(1)
    return response_json

if __name__ == "__main__":
    main()