# HotelBookingDemand

## Problem definition (done)
### We will be looking for data dependencies, which may suggest better approach for future visitors
## Data collection (done)
### Data imported using kaggle. URL: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
## Data understanding (TODO)
### 32 columns (17 int, 13 string, 1 decimal), missing data in 4 of them.
## Data cleaning (done)
### Columns with very high missing data % got dropped (company)
### Rows with missing values in columns children & country got dropped, due to insignificant data loss
### Rows with missing value in column agent (~15%) got filled using RandomForestClassifier with current success rate of >70% (TODO)
## Data engineering (TODO)
## Data analysis (TODO)