from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: kdd_el_nino-small
====
Examples: 782
====
URL: https://www.openml.org/search?type=data&id=563
====
Description: **Author**:   
**Source**: Unknown - Date unknown  
**Please cite**:   

El Nino Data

Data Type

spatio-temporal

Abstract

The data set contains oceanographic and surface meteorological
readings taken from a series of buoys positioned throughout the
equatorial Pacific. The data is expected to aid in the understanding
and prediction of El Nino/Southern Oscillation (ENSO) cycles.

Sources

Original Owner

[1]Pacific Marine Environmental Laboratory
National Oceanic and Atmospheric Administration
US Department of Commerce

Donor

[2]Dr Di Cook
Department of Statistics
Iowa State University
[3]dicook@iastate.edu

Date Donated: June 30, 1999

Data Characteristics

This data was collected with the Tropical Atmosphere Ocean (TAO) array
which was developed by the international Tropical Ocean Global
Atmosphere (TOGA) program. The TAO array consists of nearly 70 moored
buoys spanning the equatorial Pacific, measuring oceanographic and
surface meteorological variables critical for improved detection,
understanding and prediction of seasonal-to-interannual climate
variations originating in the tropics, most notably those related to
the El Nino/Southern Oscillation (ENSO) cycles.

The moorings were developed by National Oceanic and Atmospheric
Administration's (NOAA) Pacific Marine Environmental Laboratory
(PMEL). Each mooring measures air temperature, relative humidity,
surface winds, sea surface temperatures and subsurface temperatures
down to a depth of 500 meters and a few a of the buoys measure
currents, rainfall and solar radiation. The data from the array, and
current updates, can be viewed on the web at the this address .

The data consists of the following variables: date, latitude,
longitude, zonal winds (west<0, east>0), meridional winds (south<0,
north>0), relative humidity, air temperature, sea surface temperature
and subsurface temperatures down to a depth of 500 meters. Data taken
from the buoys from as early as 1980 for some locations. Other data
that was taken in various locations are rainfall, solar radiation,
current levels, and subsurface temperatures.

Variable Characteristics

The latitude and longitude in the data showed that the bouys moved
around to different locations. The latitude values stayed within a
degree from the approximate location. Yet the longitude values were
sometimes as far as five degrees off of the approximate location.

Looking at the wind data, both the zonal and meridional winds
fluctuated between -10 m/s and 10 m/s. The plot of the two wind
variables showed no linear relationship. Also, the plots of each wind
variable against the other three meteorolgical data showed no linear
relationships.

The relative humidity values in the tropical Pacific were typically
between 70% and 90%.

Both the air temperature and the sea surface temperature fluctuated
between 20 and 30 degrees Celcius. The plot of the two temperatures
variables shows a positive linear relationship existing. The two
temperatures when each plotted against time also have similar plot
designs. Plots of the other meteorological variables against the
temperature variables showed no linear relationship.

There are missing values in the data. As mentioned earlier, not all
buoys are able to measure currents, rainfall, and solar radiation, so
these values are missing dependent on the individual buoy. The amount
of data available is also dependent on the buoy, as certain buoys were
commissioned earlier than others.

All readings were taken at the same time of day.

Other Relevant Information

Background

The El Nino/Southern Oscillation (ENSO) cycle of 1982-1983, the
strongest of the century, created many problems throughout the world.
Parts of the world such as Peru and the Unites States experienced
destructive flooding from increased rainfalls while the western
Pacific areas experienced drought and devastating brush fires. The
ENSO cycle was neither predicted nor detected until it was near its
peak. This highlighted the need for an ocean observing system (i.e.
the TAO array) to support studies of large scale ocean-atmosphere
interactions on seasonal-to-interannual time scales.

The TAO array provides real-time data to climate researchers, weather
prediction centers and scientists around the world. Forcasts for
tropical Pacific Ocean temperatures for one to two years in advance
can be made using the ENSO cycle data. These forcasts are possible
because of the moored buoys, along with drifting buoys, volunteer ship
temperature probes, and sea level measurements.

Research Questions

Research questions of interest include:
* How can the data be used to predict weather conditions throughout
the world?
* How do the variables relate to each other?
* Which variables have a greater effect on the climate variations?
* Does the amount of movement of the buoy effect the reliability of
the data?

When performing an analysis of the data, one should pay attention the
possible affect of autocorrelation. Using a multiple regression
approach to model the data would require a look at autoregression
since the weather statistics of the previous days will affect today's
weather.

Data Format

The data is stored in an ASCII files with one observation per line.
Spaces separate fields and periods (.) denote missing values.

Past Usage

This data was used in the American Statistical Association Statistical
Graphics and Computing Sections 1999 Data Exposition.

References and Further Information

More information and data from the TAO array can be found at the
Pacific Marine Environmental Laboratory [4]TAO data webpage.

Information on storm data is available [5]here. This site contains
data from January 1994 to April 1998 in a chronological listing by
state provided by the National Weather Service. The data includes
hurricanes, tornadoes, thunderstorms, hail, floods, drought
conditions, lightning, high winds, snow, and temperature extremes.

Hurricane tracking data for the Atlantic is available [6]here. The
site contains a map showing the paths of the Atlantic hurricanes and
also includes the storms winds (in knots), pressure (in millibars),
and the category of the storm based on Saffir-Simpson scale.

Another site of interest related to the ENSO cyles is available
[7]here. This site contains information on twelve areas of the world
that have demonstrated ENSO-precipitation relationships. Included in
the site are maps of the areas and time series plots of actual daily
precipitation and accumulated normal precipitation for the areas.
_________________________________________________________________


[8]The UCI KDD Archive
[9]Information and Computer Science
[10]University of California, Irvine
Irvine, CA 92697-3425

Last modified: June 30, 1999

References

1. http://www.pmel.noaa.gov/
2. http://www.public.iastate.edu/~dicook/
3. mailto:dicook@iastate.edu
4. http://www.pmel.noaa.gov/toga-tao/
5. http://www.ncdc.noaa.gov/pdfs/sd/sd.html
6. http://wxp.eas.purdue.edu/hur_atlantic/
7. http://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/current_impacts/precip_accum.html
8. http://kdd.ics.uci.edu/
9. http://www.ics.uci.edu/
10. http://www.uci.edu/


Information about the dataset
CLASSTYPE: numeric
CLASSINDEX: none specific
====
Target Variable: s_s_temp (numeric, 329 distinct): ['28.77', '29.38', '29.34', '29.3', '29.44', '28.19', '29.6', '29.23', '29.58', '28.97']
====
Features:

buoy (nominal, 59 distinct): ['26', '30', '29', '32', '33', '34', '35', '36', '37', '39']
day (nominal, 14 distinct): ['1', '2', '3', '6', '4', '5', '7', '8', '9', '10']
latitude (numeric, 88 distinct): ['0.0', '-5.01', '4.97', '-1.99', '-7.97', '-5.02', '4.98', '2.08', '-1.98', '-5.0']
longitude (numeric, 123 distinct): ['-179.87', '-154.99', '-154.96', '-94.95', '-170.02', '147.0', '-170.01', '165.02', '155.94', '-125.07']
zon_winds (numeric, 114 distinct): ['-4.5', '-6.2', '-4.7', '-3.0', '-3.5', '-4.2', '-3.6', '-3.4', '-3.9', '-5.2']
mer_winds (numeric, 122 distinct): ['0.0', '-1.8', '-2.5', '-1.3', '-1.9', '-1.2', '-2.2', '-0.9', '-2.7', '-3.2']
humidity (numeric, 180 distinct): ['81.3', '87.3', '87.1', '84.4', '79.8', '83.7', '82.8', '84.8', '82.2', '79.6']
air_temp (numeric, 322 distinct): ['27.89', '27.85', '27.97', '27.65', '28.44', '28.64', '27.61', '28.18', '28.33', '28.41']
'''

CONTEXT = "Oceonographic and surface meteorogical readings for ENSO cycles"
TARGET = CuratedTarget(raw_name="s_s_temp", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = []
FEATURES = []