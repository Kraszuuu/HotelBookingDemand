# HotelBookingDemand

## Problem definition (done)
We will be looking for data dependencies, which may suggest better business strategy
## Data collection (done)
Data imported using kaggle. URL: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
## Data understanding (done)
Given data gives us booking information from two hotels (very likely from Madeira, Portugal) for time period of two totally normal years (no COVID, no war, unaffected by migration crisis)
32 columns (17 int, 13 string, 1 decimal), missing data in some of them.
## Data cleaning (done)
- Columns with very high missing data % got dropped (company)
- Rows with missing values in columns children & country got dropped, due to insignificant data loss
- Rows with missing value in column agent (~15%) got filled using RandomForestClassifier with current success rate of >70% (TODO)
- Columns about date got merged into one
- Clearly wrong records (bookings for kids without adults, non-existent dates (eg. 31.11))
## Data analysis (TODO)
## Data visualization (TODO)