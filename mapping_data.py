import json 
import pandas as pd

def main(): 
    hotelID = pd.read_csv('/home/llogn/workspace/rcm_sys/data/hotel_mapping.csv')

    with open('/home/llogn/workspace/rcm_sys/data/ivivu_hotels.json', 'r') as f:
        hotel_mapping = json.load(f)

    # print(hotel_mapping[0]['name'])
    # print(hotel.head())

    for idx, row in hotel_mapping.iterrows():
        hotel_name = row['name']
        
        if hotelID['hotel_name'][]
        print(hotel_name)
        

if __name__ == "__main__":
    main()