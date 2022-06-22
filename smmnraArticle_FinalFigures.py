import numpy
import pandas
import pandas as pd
import geopandas as gpd
from scipy.stats import sem
import matplotlib.pyplot as plt
import matplotlib.cm as CM
import libpysal as lps
from geopandas import GeoDataFrame
from shapely.geometry import Point, Polygon
from tabulate import tabulate
from scipy.optimize import curve_fit
#### Figure 3
from splot.esda import moran_scatterplot
from esda.moran import Moran, Moran_BV, Moran_Local, Moran_Local_BV
from splot.esda import lisa_cluster
from matplotlib import colors
import matplotlib.pyplot as plt

# from mpl_toolkits.basemap import Basemap

### Figure 4
import heapq
from operator import itemgetter

##### DATA COLLECTION AND CLASSES #####
class CollectionPoint(object):
    def __init__(self, line):
        #print(line)
        col = line.split(",")
        #print(col)
        self.park = [col[0].strip('"')]
        #print(self.park)
        self.sitecode = [col[2].strip('"').zfill(4)]
        # print(type(self.sitecode), self.sitecode)
        self.year = [int(col[7].strip('"'))]
        date = col[8].strip('"').split(' ')[0]
        self.date = [date]
        # print("height", col[11])
        try:
            self.plant_height = [float(col[11].strip('"'))]
        except:
            self.plant_height = [numpy.nan]
            #print("no plant height measurement")
        # month = int(date.split("/")[0])
        # day = int(date.split("/")[1])
        # year = int(date.split("/")[2])
        #
        # self.mjd = [asttime.jd2mjd(asttime.ymd2jd(year, month, day))]
        self.pointnumber = [int(col[9].strip('"'))]
        self.species_code = [col[14].strip('"')]
        if col[14].strip('"') == "BROMAD":
            self.species_code = ["BRORUB"]
        self.scientific_name = [col[15].strip('"')]
        self.fxngroup = [col[16].strip('"')]
        if col[16].strip('"') == "Vine":
            self.fxngroup = ["Herbaceous"]
        if col[16].strip('"') == "Fern":
            self.fxngroup = ["Herbaceous"]
        self.native_status = [col[17].strip('"')]
        self.ann_per = [col[18].strip('"')]

class SitePoint(object):
    def __init__(self, line):
        col = line.split(",")
        self.plotname = col[0].strip('"').strip("MS-")
        # print(type(self.plotname), self.plotname)
        self.decimalLatitude = float(col[5].strip('"'))
        self.decimalLongitude = float(col[6].strip('"'))
        self.accuracyscore = float(col[14].strip('"'))

def readCollectionFile(fname):
    # print(fname)
    f = open(fname, "r")
    next(f) #skip 1st line to remove header

    objects = {}
    for line in f:
        obj = CollectionPoint(line)
        #print(obj.sitecode, obj.year, obj.scientific_name, obj.species_code)

        try:
            objects[obj.sitecode[0]].park.append(obj.park[0])
            objects[obj.sitecode[0]].sitecode.append(obj.sitecode[0])
            objects[obj.sitecode[0]].year.append(obj.year[0])
            objects[obj.sitecode[0]].date.append(obj.date[0])
            #objects[obj.sitecode[0]].mjd.append(obj.mjd[0])
            objects[obj.sitecode[0]].pointnumber.append(obj.pointnumber[0])
            objects[obj.sitecode[0]].species_code.append(obj.species_code[0])
            objects[obj.sitecode[0]].scientific_name.append(obj.scientific_name[0])
            objects[obj.sitecode[0]].plant_height.append(obj.plant_height[0])
            objects[obj.sitecode[0]].native_status.append(obj.native_status[0])
            objects[obj.sitecode[0]].fxngroup.append(obj.fxngroup[0])
            objects[obj.sitecode[0]].ann_per.append(obj.ann_per[0])
            #print(objects[obj.sitecode[0]].sitecode, objects[obj.sitecode[0]].park, objects[obj.sitecode[0]].scientific_name)
        except KeyError:
            objects[obj.sitecode[0]] = obj
            #print(objects[obj.sitecode[0]].park, objects[obj.sitecode[0]].sitecode)

    #now go through and convert each attribute to a numpy array for easier manipulation.
    for name, obj in objects.items():
        obj.park = numpy.array([obj.park])[0]
        obj.sitecode = numpy.array([obj.sitecode])[0]
        obj.year = numpy.array([obj.year])[0]
        obj.date = numpy.array([obj.date])[0]
        #obj.mjd = numpy.array([obj.mjd])[0]
        obj.pointnumber = numpy.array([obj.pointnumber])[0]
        obj.species_code = numpy.array([obj.species_code])[0]
        obj.scientific_name = numpy.array([obj.scientific_name])[0]
        obj.plant_height = numpy.array([obj.plant_height])[0]
        obj.native_status = numpy.array([obj.native_status])[0]
        obj.fxngroup = numpy.array([obj.fxngroup])[0]
        obj.ann_per = numpy.array([obj.ann_per])[0]

    return objects

#ingest the individual site coordinates
sitefile = open("/Users/nicolettastork/Documents/Geography/SMMNRA_Invasive_Species/NPSData/coords_from_MASTER_VegMon_Sampling_Record_all_panels_Combined_beg2021.csv", "r")
next(sitefile) #skip 1st and 2nd lines to remove headers
next(sitefile)
siteproperties = {}
for line in sitefile:
    #print(line)
    obj = SitePoint(line)
#    print(obj.plotname, obj.decimalLongitude, obj.decimalLatitude)
    siteproperties[obj.plotname] = obj

#Now load up the vegetation monitoring data for the individual sites
datafile = "/Users/nicolettastork/Documents/Geography/SMMNRA_Invasive_Species/NPSData/NPS_RawData.csv"
#datafile = "/data1/Conservation/Veg_Monitoring_20201216/temp.csv"
monitoringpoints = readCollectionFile(datafile)
# print('length of monitoring points:', len(monitoringpoints))
WoolseyPoints = pd.read_csv("/Users/nicolettastork/Documents/Geography/SMMNRA_Invasive_Species/NPSData/NPS_MasterCoords_inWoolseyFire.csv", sep=',')
WoolseyPoints = [x.strip("MS-") for x in WoolseyPoints["PlotName"]]

##### Now let's read the KMZ file that contains the Woolsey Fire boundary and get it into a polygon format that can be used as a boundary.
#first unzip the KMZ file - it's just a zip file, turns out. Just have to rename the resulting doc.kml file to the original name.
gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
fp = "/Users/nicolettastork/Documents/Geography/SMMNRA_Invasive_Species/GIS_KMZ/20181108Woolsey_perim_revisedNPS_v7RT2.kml"
WoolseyBoundary = gpd.read_file(fp, driver="KML")
WoolsBound = gpd.GeoDataFrame(WoolseyBoundary)

gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
fp = "/Users/nicolettastork/Documents/Geography/SMMNRA_Invasive_Species/GIS_KMZ/20130502SpringsFire_perim_final_bestSAMOversionCunburned_U11N83.kml"
SpringsBoundary = gpd.read_file(fp, driver="KML")

gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
fp = "/Users/nicolettastork/Documents/Geography/SMMNRA_Invasive_Species/GIS_KMZ/UpperLasVirgenesOpenSpacePreserve.kml"
LasVirgenes = gpd.read_file(fp, driver="KML")

gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
smm = "/Users/nicolettastork/Documents/Geography/SMMNRA_Invasive_Species/GIS_KMZ/SMMNRA_Boundary.kml"
SMMNRAboundary = gpd.read_file(smm, driver="KML")

#### MAJOR FREEWAYS #####
gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
fwy = "/Users/nicolettastork/Documents/Geography/SMMNRA_Invasive_Species/GIS_QGIS/MainRoads/Freeway101_Full.kml"
Freeway101 = gpd.read_file(fwy, driver="KML")

gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
fwy = "/Users/nicolettastork/Documents/Geography/SMMNRA_Invasive_Species/GIS_QGIS/MainRoads/PCH_Full.kml"
PCH = gpd.read_file(fwy, driver="KML")

gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
fwy = "/Users/nicolettastork/Documents/Geography/SMMNRA_Invasive_Species/GIS_QGIS/MainRoads/Freeway405.kml"
Freeway405 = gpd.read_file(fwy, driver="KML")

gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
fwy = "/Users/nicolettastork/Documents/Geography/SMMNRA_Invasive_Species/GIS_QGIS/MainRoads/I10_Freeway.kml"
Freeway_i10 = gpd.read_file(fwy, driver="KML")

gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
fwy = "/Users/nicolettastork/Documents/Geography/SMMNRA_Invasive_Species/GIS_KMZ/Coastline_Polygon.kml"
Coastline = gpd.read_file(fwy, driver="KML")

for name, obj in monitoringpoints.items():
    # Now add the latitude & longitude from the site description file.
    # try:
    obj.decimalLatitude = siteproperties[name].decimalLatitude
    obj.decimalLongitude = siteproperties[name].decimalLongitude
    # except KeyError:
    #     obj.decimalLatitude = []
    #     obj.decimalLongitude = []
    monitoringpoints_gdf = Point(obj.decimalLongitude, obj.decimalLatitude)
    for index, row in WoolsBound.iterrows():
        if monitoringpoints_gdf.within(row['geometry']) == True:
            obj.inWoolsey = True
        else:
            obj.inWoolsey = False

numpy.set_printoptions(threshold=numpy.inf)

ea_axisnum = 6
ea_legend = 7
ea_axislabel = 8
ea_panellabel = 8
ea_panelletter = 10

darkestblue = '#042243'
darkblue = '#044177'
blue = '#0f75a9'
lightblue = '#52bedb'
lightestblue = '#ace6ea'
yellow = '#FFCE64'
gold = '#E79741'
rust = '#B84311'
darkred = '#781008'

monsites2014 = 0
monsites2015 = 0
monsites2016 = 0
monsites2017 = 0
monsites2018 = 0
monsites2019 = 0
monsites2020 = 0

monsites_inW_2014 = 0
monsites_inW_2015 = 0
monsites_inW_2016 = 0
monsites_inW_2017 = 0
monsites_inW_2018 = 0
monsites_inW_2019 = 0
monsites_inW_2020 = 0

monsites_outW_2014 = 0
monsites_outW_2015 = 0
monsites_outW_2016 = 0
monsites_outW_2017 = 0
monsites_outW_2018 = 0
monsites_outW_2019 = 0
monsites_outW_2020 = 0

latlong20 = numpy.array([])

# for name, obj in monitoringpoints.items():
#     test = numpy.where(obj.year == 2020)
#     print(numpy.unique(obj.year[test]), obj.decimalLongitude, obj.decimalLatitude)

for name, obj in monitoringpoints.items():
    if (obj.year == 2014).any():
        print(name, obj.decimalLongitude, obj.decimalLatitude)

for name, obj in monitoringpoints.items():
    testindices = numpy.where(obj.year == 2014)
    monsites2014 += len(numpy.unique(obj.sitecode[testindices]))
    testindices = numpy.where(obj.year == 2015)
    monsites2015 += len(numpy.unique(obj.sitecode[testindices]))
    testindices = numpy.where(obj.year == 2016)
    monsites2016 += len(numpy.unique(obj.sitecode[testindices]))
    testindices = numpy.where(obj.year == 2017)
    monsites2017 += len(numpy.unique(obj.sitecode[testindices]))
    testindices = numpy.where(obj.year == 2018)
    monsites2018 += len(numpy.unique(obj.sitecode[testindices]))
    testindices = numpy.where(obj.year == 2019)
    monsites2019 += len(numpy.unique(obj.sitecode[testindices]))
    testindices = numpy.where(obj.year == 2020)
    monsites2020 += len(numpy.unique(obj.sitecode[testindices]))

    testindices = numpy.where((obj.year == 2014) & (obj.inWoolsey == True))
    monsites_inW_2014 += len(numpy.unique(obj.sitecode[testindices]))
    testindices = numpy.where((obj.year == 2015) & (obj.inWoolsey == True))
    monsites_inW_2015 += len(numpy.unique(obj.sitecode[testindices]))
    testindices = numpy.where((obj.year == 2016) & (obj.inWoolsey == True))
    monsites_inW_2016 += len(numpy.unique(obj.sitecode[testindices]))
    testindices = numpy.where((obj.year == 2017) & (obj.inWoolsey == True))
    monsites_inW_2017 += len(numpy.unique(obj.sitecode[testindices]))
    testindices = numpy.where((obj.year == 2018) & (obj.inWoolsey == True))
    monsites_inW_2018 += len(numpy.unique(obj.sitecode[testindices]))
    testindices = numpy.where((obj.year == 2019) & (obj.inWoolsey == True))
    monsites_inW_2019 += len(numpy.unique(obj.sitecode[testindices]))
    testindices = numpy.where((obj.year == 2020) & (obj.inWoolsey == True))
    monsites_inW_2020 += len(numpy.unique(obj.sitecode[testindices]))

    testindices = numpy.where((obj.year == 2014) & (obj.inWoolsey == False))
    monsites_outW_2014 += len(numpy.unique(obj.sitecode[testindices]))
    testindices = numpy.where((obj.year == 2015) & (obj.inWoolsey == False))
    monsites_outW_2015 += len(numpy.unique(obj.sitecode[testindices]))
    testindices = numpy.where((obj.year == 2016) & (obj.inWoolsey == False))
    monsites_outW_2016 += len(numpy.unique(obj.sitecode[testindices]))
    testindices = numpy.where((obj.year == 2017) & (obj.inWoolsey == False))
    monsites_outW_2017 += len(numpy.unique(obj.sitecode[testindices]))
    testindices = numpy.where((obj.year == 2018) & (obj.inWoolsey == False))
    monsites_outW_2018 += len(numpy.unique(obj.sitecode[testindices]))
    testindices = numpy.where((obj.year == 2019) & (obj.inWoolsey == False))
    monsites_outW_2019 += len(numpy.unique(obj.sitecode[testindices]))
    testindices = numpy.where((obj.year == 2020) & (obj.inWoolsey == False))
    monsites_outW_2020 += len(numpy.unique(obj.sitecode[testindices]))

# monitoringsite_table = [['Monitoring sites surveyed each year'],
#                         ['Years', 2014, 2015, 2016, 2017, 2018, 2019, 2020],
#                         ['Whole park', monsites2014, monsites2015, monsites2016, monsites2017,
#                          monsites2018, monsites2019, monsites2020],
#                         ['Inside Woolsey Fire boundary', monsites_inW_2014, monsites_inW_2015,
#                          monsites_inW_2016, monsites_inW_2017, monsites_inW_2018, monsites_inW_2019, monsites_inW_2020],
#                         ['Outside Woolsey Fire boundary ', monsites_outW_2014, monsites_outW_2015,
#                         monsites_outW_2016, monsites_outW_2017, monsites_outW_2018, monsites_outW_2019, monsites_outW_2020
#                         ]]
# print(tabulate(monitoringsite_table))

# monitoringsite_table = [[2014, 2015, 2016, 2017, 2018, 2019, 2020],
#                         [monsites2014, monsites2015, monsites2016, monsites2017, monsites2018, monsites2019, monsites2020],
#                         [monsites_inW_2014, monsites_inW_2015, monsites_inW_2016, monsites_inW_2017, monsites_inW_2018, monsites_inW_2019, monsites_inW_2020],
#                         [monsites_outW_2014, monsites_outW_2015, monsites_outW_2016, monsites_outW_2017, monsites_outW_2018, monsites_outW_2019, monsites_outW_2020
#                         ]]
#
# print('\\begin{tabular}{lllllll} \\toprule')
# for row in monitoringsite_table:
#     for item in row:
#         y = ' & '.join([str(item) for item in row])
#     print(y + ' \\''\\')
# print('\\end{tabular}')

#######################################################################
#### FIGURE 2: Native and non-native cover percentages chart
#######################################################################
#
# #### STANDARD ERROR OF DATASET #####
# ### Creating arrays #####
# sitecodes_inW = numpy.array([])
# sitecodes_outW = numpy.array([])
#
# sitecode_totals_2014 = numpy.array([])
# sitecode_totals_2015 = numpy.array([])
# sitecode_totals_2016 = numpy.array([])
# sitecode_totals_2017 = numpy.array([])
# sitecode_totals_2018 = numpy.array([])
# sitecode_totals_2019 = numpy.array([])
# sitecode_totals_2020 = numpy.array([])
# sitecode_native2014 = numpy.array([])
# sitecode_native2015 = numpy.array([])
# sitecode_native2016 = numpy.array([])
# sitecode_native2017 = numpy.array([])
# sitecode_native2018 = numpy.array([])
# sitecode_native2019 = numpy.array([])
# sitecode_native2020 = numpy.array([])
# sitecode_nonnative2014 = numpy.array([])
# sitecode_nonnative2015 = numpy.array([])
# sitecode_nonnative2016 = numpy.array([])
# sitecode_nonnative2017 = numpy.array([])
# sitecode_nonnative2018 = numpy.array([])
# sitecode_nonnative2019 = numpy.array([])
# sitecode_nonnative2020 = numpy.array([])
#
# sitecode_inW_totals_2014 = numpy.array([])
# sitecode_inW_totals_2015 = numpy.array([])
# sitecode_inW_totals_2016 = numpy.array([])
# sitecode_inW_totals_2017 = numpy.array([])
# sitecode_inW_totals_2018 = numpy.array([])
# sitecode_inW_totals_2019 = numpy.array([])
# sitecode_inW_totals_2020 = numpy.array([])
# sitecode_inW_native2014 = numpy.array([])
# sitecode_inW_native2015 = numpy.array([])
# sitecode_inW_native2016 = numpy.array([])
# sitecode_inW_native2017 = numpy.array([])
# sitecode_inW_native2018 = numpy.array([])
# sitecode_inW_native2019 = numpy.array([])
# sitecode_inW_native2020 = numpy.array([])
# sitecode_inW_nonnative2014 = numpy.array([])
# sitecode_inW_nonnative2015 = numpy.array([])
# sitecode_inW_nonnative2016 = numpy.array([])
# sitecode_inW_nonnative2017 = numpy.array([])
# sitecode_inW_nonnative2018 = numpy.array([])
# sitecode_inW_nonnative2019 = numpy.array([])
# sitecode_inW_nonnative2020 = numpy.array([])
#
# sitecode_outW_totals_2014 = numpy.array([])
# sitecode_outW_totals_2015 = numpy.array([])
# sitecode_outW_totals_2016 = numpy.array([])
# sitecode_outW_totals_2017 = numpy.array([])
# sitecode_outW_totals_2018 = numpy.array([])
# sitecode_outW_totals_2019 = numpy.array([])
# sitecode_outW_totals_2020 = numpy.array([])
# sitecode_outW_native2014 = numpy.array([])
# sitecode_outW_native2015 = numpy.array([])
# sitecode_outW_native2016 = numpy.array([])
# sitecode_outW_native2017 = numpy.array([])
# sitecode_outW_native2018 = numpy.array([])
# sitecode_outW_native2019 = numpy.array([])
# sitecode_outW_native2020 = numpy.array([])
# sitecode_outW_nonnative2014 = numpy.array([])
# sitecode_outW_nonnative2015 = numpy.array([])
# sitecode_outW_nonnative2016 = numpy.array([])
# sitecode_outW_nonnative2017 = numpy.array([])
# sitecode_outW_nonnative2018 = numpy.array([])
# sitecode_outW_nonnative2019 = numpy.array([])
# sitecode_outW_nonnative2020 = numpy.array([])
#
# #### Fill out arrays #####
# for name, obj in monitoringpoints.items():
#     test_indices = numpy.where(obj.inWoolsey == True)
#     sitecodes_inW = numpy.append(sitecodes_inW, obj.sitecode[test_indices])
#     test_indices = numpy.where(obj.inWoolsey == False)
#     sitecodes_outW = numpy.append(sitecodes_outW, obj.sitecode[test_indices])
#
#     test_indices = numpy.where(obj.year == 2014)
#     sitecode_totals_2014 = numpy.append(sitecode_totals_2014, obj.sitecode[test_indices])
#     test_indices = numpy.where(obj.year == 2015)
#     sitecode_totals_2015 = numpy.append(sitecode_totals_2015, obj.sitecode[test_indices])
#     test_indices = numpy.where(obj.year == 2016)
#     sitecode_totals_2016 = numpy.append(sitecode_totals_2016, obj.sitecode[test_indices])
#     test_indices = numpy.where(obj.year == 2017)
#     sitecode_totals_2017 = numpy.append(sitecode_totals_2017, obj.sitecode[test_indices])
#     test_indices = numpy.where(obj.year == 2018)
#     sitecode_totals_2018 = numpy.append(sitecode_totals_2018, obj.sitecode[test_indices])
#     test_indices = numpy.where(obj.year == 2019)
#     sitecode_totals_2019 = numpy.append(sitecode_totals_2019, obj.sitecode[test_indices])
#     test_indices = numpy.where(obj.year == 2020)
#     sitecode_totals_2020 = numpy.append(sitecode_totals_2020, obj.sitecode[test_indices])
#
#     test_indices = numpy.where((obj.year == 2014) & (obj.native_status == 'Native'))
#     sitecode_native2014 = numpy.append(sitecode_native2014, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2015) & (obj.native_status == 'Native'))
#     sitecode_native2015 = numpy.append(sitecode_native2015, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2016) & (obj.native_status == 'Native'))
#     sitecode_native2016 = numpy.append(sitecode_native2016, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2017) & (obj.native_status == 'Native'))
#     sitecode_native2017 = numpy.append(sitecode_native2017, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2018) & (obj.native_status == 'Native'))
#     sitecode_native2018 = numpy.append(sitecode_native2018, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2019) & (obj.native_status == 'Native'))
#     sitecode_native2019 = numpy.append(sitecode_native2019, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2020) & (obj.native_status == 'Native'))
#     sitecode_native2020 = numpy.append(sitecode_native2020, obj.sitecode[test_indices])
#
#     test_indices = numpy.where((obj.year == 2014) & (obj.native_status == 'Nonnative'))
#     sitecode_nonnative2014 = numpy.append(sitecode_nonnative2014, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2015) & (obj.native_status == 'Nonnative'))
#     sitecode_nonnative2015 = numpy.append(sitecode_nonnative2015, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2016) & (obj.native_status == 'Nonnative'))
#     sitecode_nonnative2016 = numpy.append(sitecode_nonnative2016, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2017) & (obj.native_status == 'Nonnative'))
#     sitecode_nonnative2017 = numpy.append(sitecode_nonnative2017, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2018) & (obj.native_status == 'Nonnative'))
#     sitecode_nonnative2018 = numpy.append(sitecode_nonnative2018, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2019) & (obj.native_status == 'Nonnative'))
#     sitecode_nonnative2019 = numpy.append(sitecode_nonnative2019, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2020) & (obj.native_status == 'Nonnative'))
#     sitecode_nonnative2020 = numpy.append(sitecode_nonnative2020, obj.sitecode[test_indices])
#
# ##### Inside Woolsey #####
#     test_indices = numpy.where((obj.year == 2014) & (obj.inWoolsey == True))
#     sitecode_inW_totals_2014 = numpy.append(sitecode_inW_totals_2014, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2015) & (obj.inWoolsey == True))
#     sitecode_inW_totals_2015 = numpy.append(sitecode_inW_totals_2015, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2016) & (obj.inWoolsey == True))
#     sitecode_inW_totals_2016 = numpy.append(sitecode_inW_totals_2016, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2017) & (obj.inWoolsey == True))
#     sitecode_inW_totals_2017 = numpy.append(sitecode_inW_totals_2017, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2018) & (obj.inWoolsey == True))
#     sitecode_inW_totals_2018 = numpy.append(sitecode_inW_totals_2018, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2019) & (obj.inWoolsey == True))
#     sitecode_inW_totals_2019 = numpy.append(sitecode_inW_totals_2019, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2020) & (obj.inWoolsey == True))
#     sitecode_inW_totals_2020 = numpy.append(sitecode_inW_totals_2020, obj.sitecode[test_indices])
#
#     test_indices = numpy.where((obj.year == 2014) & (obj.native_status == 'Native') & (obj.inWoolsey == True))
#     sitecode_inW_native2014 = numpy.append(sitecode_inW_native2014, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2015) & (obj.native_status == 'Native') & (obj.inWoolsey == True))
#     sitecode_inW_native2015 = numpy.append(sitecode_inW_native2015, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2016) & (obj.native_status == 'Native') & (obj.inWoolsey == True))
#     sitecode_inW_native2016 = numpy.append(sitecode_inW_native2016, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2017) & (obj.native_status == 'Native') & (obj.inWoolsey == True))
#     sitecode_inW_native2017 = numpy.append(sitecode_inW_native2017, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2018) & (obj.native_status == 'Native') & (obj.inWoolsey == True))
#     sitecode_inW_native2018 = numpy.append(sitecode_inW_native2018, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2019) & (obj.native_status == 'Native') & (obj.inWoolsey == True))
#     sitecode_inW_native2019 = numpy.append(sitecode_inW_native2019, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2020) & (obj.native_status == 'Native') & (obj.inWoolsey == True))
#     sitecode_inW_native2020 = numpy.append(sitecode_inW_native2020, obj.sitecode[test_indices])
#
#     test_indices = numpy.where((obj.year == 2014) & (obj.native_status == 'Nonnative') & (obj.inWoolsey == True))
#     sitecode_inW_nonnative2014 = numpy.append(sitecode_inW_nonnative2014, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2015) & (obj.native_status == 'Nonnative') & (obj.inWoolsey == True))
#     sitecode_inW_nonnative2015 = numpy.append(sitecode_inW_nonnative2015, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2016) & (obj.native_status == 'Nonnative') & (obj.inWoolsey == True))
#     sitecode_inW_nonnative2016 = numpy.append(sitecode_inW_nonnative2016, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2017) & (obj.native_status == 'Nonnative') & (obj.inWoolsey == True))
#     sitecode_inW_nonnative2017 = numpy.append(sitecode_inW_nonnative2017, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2018) & (obj.native_status == 'Nonnative') & (obj.inWoolsey == True))
#     sitecode_inW_nonnative2018 = numpy.append(sitecode_inW_nonnative2018, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2019) & (obj.native_status == 'Nonnative') & (obj.inWoolsey == True))
#     sitecode_inW_nonnative2019 = numpy.append(sitecode_inW_nonnative2019, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2020) & (obj.native_status == 'Nonnative') & (obj.inWoolsey == True))
#     sitecode_inW_nonnative2020 = numpy.append(sitecode_inW_nonnative2020, obj.sitecode[test_indices])
#
# ##### Outside Woolsey #####
#     test_indices = numpy.where((obj.year == 2014) & (obj.inWoolsey == False))
#     sitecode_outW_totals_2014 = numpy.append(sitecode_outW_totals_2014, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2015) & (obj.inWoolsey == False))
#     sitecode_outW_totals_2015 = numpy.append(sitecode_outW_totals_2015, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2016) & (obj.inWoolsey == False))
#     sitecode_outW_totals_2016 = numpy.append(sitecode_outW_totals_2016, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2017) & (obj.inWoolsey == False))
#     sitecode_outW_totals_2017 = numpy.append(sitecode_outW_totals_2017, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2018) & (obj.inWoolsey == False))
#     sitecode_outW_totals_2018 = numpy.append(sitecode_outW_totals_2018, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2019) & (obj.inWoolsey == False))
#     sitecode_outW_totals_2019 = numpy.append(sitecode_outW_totals_2019, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2020) & (obj.inWoolsey == False))
#     sitecode_outW_totals_2020 = numpy.append(sitecode_outW_totals_2020, obj.sitecode[test_indices])
#
#     test_indices = numpy.where((obj.year == 2014) & (obj.native_status == 'Native') & (obj.inWoolsey == False))
#     sitecode_outW_native2014 = numpy.append(sitecode_outW_native2014, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2015) & (obj.native_status == 'Native') & (obj.inWoolsey == False))
#     sitecode_outW_native2015 = numpy.append(sitecode_outW_native2015, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2016) & (obj.native_status == 'Native') & (obj.inWoolsey == False))
#     sitecode_outW_native2016 = numpy.append(sitecode_outW_native2016, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2017) & (obj.native_status == 'Native') & (obj.inWoolsey == False))
#     sitecode_outW_native2017 = numpy.append(sitecode_outW_native2017, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2018) & (obj.native_status == 'Native') & (obj.inWoolsey == False))
#     sitecode_outW_native2018 = numpy.append(sitecode_outW_native2018, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2019) & (obj.native_status == 'Native') & (obj.inWoolsey == False))
#     sitecode_outW_native2019 = numpy.append(sitecode_outW_native2019, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2020) & (obj.native_status == 'Native') & (obj.inWoolsey == False))
#     sitecode_outW_native2020 = numpy.append(sitecode_outW_native2020, obj.sitecode[test_indices])
#
#     test_indices = numpy.where((obj.year == 2014) & (obj.native_status == 'Nonnative') & (obj.inWoolsey == False))
#     sitecode_outW_nonnative2014 = numpy.append(sitecode_outW_nonnative2014, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2015) & (obj.native_status == 'Nonnative') & (obj.inWoolsey == False))
#     sitecode_outW_nonnative2015 = numpy.append(sitecode_outW_nonnative2015, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2016) & (obj.native_status == 'Nonnative') & (obj.inWoolsey == False))
#     sitecode_outW_nonnative2016 = numpy.append(sitecode_outW_nonnative2016, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2017) & (obj.native_status == 'Nonnative') & (obj.inWoolsey == False))
#     sitecode_outW_nonnative2017 = numpy.append(sitecode_outW_nonnative2017, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2018) & (obj.native_status == 'Nonnative') & (obj.inWoolsey == False))
#     sitecode_outW_nonnative2018 = numpy.append(sitecode_outW_nonnative2018, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2019) & (obj.native_status == 'Nonnative') & (obj.inWoolsey == False))
#     sitecode_outW_nonnative2019 = numpy.append(sitecode_outW_nonnative2019, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2020) & (obj.native_status == 'Nonnative') & (obj.inWoolsey == False))
#     sitecode_outW_nonnative2020 = numpy.append(sitecode_outW_nonnative2020, obj.sitecode[test_indices])
#
# print('sitecodes in Woolsey')
# print(numpy.unique(sitecodes_inW))
# print('sitecodes outside Woolsey')
# print(numpy.unique(sitecodes_outW))
#
# def totalsdict(tot14, nat14, nonnat14,
#                tot15, nat15, nonnat15,
#                tot16, nat16, nonnat16,
#                tot17, nat17, nonnat17,
#                tot18, nat18, nonnat18,
#                tot19, nat19, nonnat19,
#                tot20, nat20, nonnat20,
#                totinW14, natinW14, nonnatinW14,
#                totinW15, natinW15, nonnatinW15,
#                totinW16, natinW16, nonnatinW16,
#                totinW17, natinW17, nonnatinW17,
#                totinW18, natinW18, nonnatinW18,
#                totinW19, natinW19, nonnatinW19,
#                totinW20, natinW20, nonnatinW20,
#                totoutW14, natoutW14, nonnatoutW14,
#                totoutW15, natoutW15, nonnatoutW15,
#                totoutW16, natoutW16, nonnatoutW16,
#                totoutW17, natoutW17, nonnatoutW17,
#                totoutW18, natoutW18, nonnatoutW18,
#                totoutW19, natoutW19, nonnatoutW19,
#                totoutW20, natoutW20, nonnatoutW20
#                ):
#             labelstotals14, countstotals14 = numpy.unique(tot14, return_counts=True)
#             totals14 = {labelstotals14[i]: countstotals14[i] for i in range(len(labelstotals14))}
#             labelsnat14, countsnat = numpy.unique(nat14, return_counts=True)
#             natives14 = {labelsnat14[i]: countsnat[i] for i in range(len(labelsnat14))}
#             labelsnonnat14, countsnonnat14 = numpy.unique(nonnat14, return_counts=True)
#             nonnatives14 = {labelsnonnat14[i]: countsnonnat14[i] for i in range(len(labelsnonnat14))}
#             fulldict14 = {}
#             for key in set(list(totals14.keys()) + list(natives14.keys())):
#                 try:
#                     fulldict14.setdefault(key, []).append(totals14[key])
#                 except KeyError:
#                     fulldict14.setdefault(key, []).append(0)
#                 try:
#                     fulldict14.setdefault(key, []).append(natives14[key])
#                 except KeyError:
#                     fulldict14.setdefault(key, []).append(0)
#                 try:
#                     fulldict14.setdefault(key, []).append(nonnatives14[key])
#                 except KeyError:
#                     fulldict14.setdefault(key, []).append(0)
#             values14 = numpy.array(list(fulldict14.values()))
#             valuestotals14 = [sublist[0] for sublist in values14]
#             valuesnatives14 = [sublist[1] for sublist in values14]
#             valuesnonnatives14 = [sublist[-1] for sublist in values14]
#             nptotals14 = numpy.asarray(valuestotals14)
#             npnatives14 = numpy.asarray(valuesnatives14)
#             npnonnatives14 = numpy.asarray(valuesnonnatives14)
#             nativepercentage14 = numpy.divide(npnatives14, nptotals14)
#             nonnativepercentage14 = numpy.divide(npnonnatives14, nptotals14)
#             nativemean14 = numpy.around(numpy.mean(nativepercentage14) * 100, 2)
#             nativesd14 = numpy.around(numpy.std(nativepercentage14) * 100, 2)
#             nativesem14 = numpy.around(sem(nativepercentage14) * 100, 2)
#             nonnativemean14 = numpy.around(numpy.mean(nonnativepercentage14) * 100, 2)
#             nonnativesd14 = numpy.around(numpy.std(nonnativepercentage14) * 100, 2)
#             nonnativesem14 = numpy.around(sem(nonnativepercentage14) * 100, 2)
#
#             ########## 2015 ##########
#
#             labelstotals15, countstotals15 = numpy.unique(tot15, return_counts=True)
#             totals15 = {labelstotals15[i]: countstotals15[i] for i in range(len(labelstotals15))}
#             labelsnat15, countsnat = numpy.unique(nat15, return_counts=True)
#             natives15 = {labelsnat15[i]: countsnat[i] for i in range(len(labelsnat15))}
#             labelsnonnat15, countsnonnat15 = numpy.unique(nonnat15, return_counts=True)
#             nonnatives15 = {labelsnonnat15[i]: countsnonnat15[i] for i in range(len(labelsnonnat15))}
#             fulldict15 = {}
#             for key in set(list(totals15.keys()) + list(natives15.keys())):
#                 try:
#                     fulldict15.setdefault(key, []).append(totals15[key])
#                 except KeyError:
#                     fulldict15.setdefault(key, []).append(0)
#                 try:
#                     fulldict15.setdefault(key, []).append(natives15[key])
#                 except KeyError:
#                     fulldict15.setdefault(key, []).append(0)
#                 try:
#                     fulldict15.setdefault(key, []).append(nonnatives15[key])
#                 except KeyError:
#                     fulldict15.setdefault(key, []).append(0)
#             values15 = numpy.array(list(fulldict15.values()))
#             valuestotals15 = [sublist[0] for sublist in values15]
#             valuesnatives15 = [sublist[1] for sublist in values15]
#             valuesnonnatives15 = [sublist[-1] for sublist in values15]
#             nptotals15 = numpy.asarray(valuestotals15)
#             npnatives15 = numpy.asarray(valuesnatives15)
#             npnonnatives15 = numpy.asarray(valuesnonnatives15)
#             nativepercentage15 = numpy.divide(npnatives15, nptotals15)
#             nonnativepercentage15 = numpy.divide(npnonnatives15, nptotals15)
#             nativemean15 = numpy.around(numpy.mean(nativepercentage15) * 100, 2)
#             nativesd15 = numpy.around(numpy.std(nativepercentage15) * 100, 2)
#             nativesem15 = numpy.around(sem(nativepercentage15) * 100, 2)
#             nonnativemean15 = numpy.around(numpy.mean(nonnativepercentage15) * 100, 2)
#             nonnativesd15 = numpy.around(numpy.std(nonnativepercentage15) * 100, 2)
#             nonnativesem15 = numpy.around(sem(nonnativepercentage15) * 100, 2)
#
#             ########## 2016 ##########
#
#             labelstotals16, countstotals16 = numpy.unique(tot16, return_counts=True)
#             totals16 = {labelstotals16[i]: countstotals16[i] for i in range(len(labelstotals16))}
#             labelsnat16, countsnat = numpy.unique(nat16, return_counts=True)
#             natives16 = {labelsnat16[i]: countsnat[i] for i in range(len(labelsnat16))}
#             labelsnonnat16, countsnonnat16 = numpy.unique(nonnat16, return_counts=True)
#             nonnatives16 = {labelsnonnat16[i]: countsnonnat16[i] for i in range(len(labelsnonnat16))}
#             fulldict16 = {}
#             for key in set(list(totals16.keys()) + list(natives16.keys())):
#                 try:
#                     fulldict16.setdefault(key, []).append(totals16[key])
#                 except KeyError:
#                     fulldict16.setdefault(key, []).append(0)
#                 try:
#                     fulldict16.setdefault(key, []).append(natives16[key])
#                 except KeyError:
#                     fulldict16.setdefault(key, []).append(0)
#                 try:
#                     fulldict16.setdefault(key, []).append(nonnatives16[key])
#                 except KeyError:
#                     fulldict16.setdefault(key, []).append(0)
#             values16 = numpy.array(list(fulldict16.values()))
#             valuestotals16 = [sublist[0] for sublist in values16]
#             valuesnatives16 = [sublist[1] for sublist in values16]
#             valuesnonnatives16 = [sublist[-1] for sublist in values16]
#             nptotals16 = numpy.asarray(valuestotals16)
#             npnatives16 = numpy.asarray(valuesnatives16)
#             npnonnatives16 = numpy.asarray(valuesnonnatives16)
#             nativepercentage16 = numpy.divide(npnatives16, nptotals16)
#             nonnativepercentage16 = numpy.divide(npnonnatives16, nptotals16)
#             nativemean16 = numpy.around(numpy.mean(nativepercentage16) * 100, 2)
#             nativesd16 = numpy.around(numpy.std(nativepercentage16) * 100, 2)
#             nativesem16 = numpy.around(sem(nativepercentage16) * 100, 2)
#             nonnativemean16 = numpy.around(numpy.mean(nonnativepercentage16) * 100, 2)
#             nonnativesd16 = numpy.around(numpy.std(nonnativepercentage16) * 100, 2)
#             nonnativesem16 = numpy.around(sem(nonnativepercentage16) * 100, 2)
#
#             ########## 2017 ##########
#
#             labelstotals17, countstotals17 = numpy.unique(tot17, return_counts=True)
#             totals17 = {labelstotals17[i]: countstotals17[i] for i in range(len(labelstotals17))}
#             labelsnat17, countsnat = numpy.unique(nat17, return_counts=True)
#             natives17 = {labelsnat17[i]: countsnat[i] for i in range(len(labelsnat17))}
#             labelsnonnat17, countsnonnat17 = numpy.unique(nonnat17, return_counts=True)
#             nonnatives17 = {labelsnonnat17[i]: countsnonnat17[i] for i in range(len(labelsnonnat17))}
#             fulldict17 = {}
#             for key in set(list(totals17.keys()) + list(natives17.keys())):
#                 try:
#                     fulldict17.setdefault(key, []).append(totals17[key])
#                 except KeyError:
#                     fulldict17.setdefault(key, []).append(0)
#                 try:
#                     fulldict17.setdefault(key, []).append(natives17[key])
#                 except KeyError:
#                     fulldict17.setdefault(key, []).append(0)
#                 try:
#                     fulldict17.setdefault(key, []).append(nonnatives17[key])
#                 except KeyError:
#                     fulldict17.setdefault(key, []).append(0)
#             values17 = numpy.array(list(fulldict17.values()))
#             valuestotals17 = [sublist[0] for sublist in values17]
#             valuesnatives17 = [sublist[1] for sublist in values17]
#             valuesnonnatives17 = [sublist[-1] for sublist in values17]
#             nptotals17 = numpy.asarray(valuestotals17)
#             npnatives17 = numpy.asarray(valuesnatives17)
#             npnonnatives17 = numpy.asarray(valuesnonnatives17)
#             nativepercentage17 = numpy.divide(npnatives17, nptotals17)
#             nonnativepercentage17 = numpy.divide(npnonnatives17, nptotals17)
#             nativemean17 = numpy.around(numpy.mean(nativepercentage17) * 100, 2)
#             nativesd17 = numpy.around(numpy.std(nativepercentage17) * 100, 2)
#             nativesem17 = numpy.around(sem(nativepercentage17) * 100, 2)
#             nonnativemean17 = numpy.around(numpy.mean(nonnativepercentage17) * 100, 2)
#             nonnativesd17 = numpy.around(numpy.std(nonnativepercentage17) * 100, 2)
#             nonnativesem17 = numpy.around(sem(nonnativepercentage17) * 100, 2)
#
#             ########## 2018 ##########
#
#             labelstotals18, countstotals18 = numpy.unique(tot18, return_counts=True)
#             totals18 = {labelstotals18[i]: countstotals18[i] for i in range(len(labelstotals18))}
#             labelsnat18, countsnat = numpy.unique(nat18, return_counts=True)
#             natives18 = {labelsnat18[i]: countsnat[i] for i in range(len(labelsnat18))}
#             labelsnonnat18, countsnonnat18 = numpy.unique(nonnat18, return_counts=True)
#             nonnatives18 = {labelsnonnat18[i]: countsnonnat18[i] for i in range(len(labelsnonnat18))}
#             fulldict18 = {}
#             for key in set(list(totals18.keys()) + list(natives18.keys())):
#                 try:
#                     fulldict18.setdefault(key, []).append(totals18[key])
#                 except KeyError:
#                     fulldict18.setdefault(key, []).append(0)
#                 try:
#                     fulldict18.setdefault(key, []).append(natives18[key])
#                 except KeyError:
#                     fulldict18.setdefault(key, []).append(0)
#                 try:
#                     fulldict18.setdefault(key, []).append(nonnatives18[key])
#                 except KeyError:
#                     fulldict18.setdefault(key, []).append(0)
#             values18 = numpy.array(list(fulldict18.values()))
#             valuestotals18 = [sublist[0] for sublist in values18]
#             valuesnatives18 = [sublist[1] for sublist in values18]
#             valuesnonnatives18 = [sublist[-1] for sublist in values18]
#             nptotals18 = numpy.asarray(valuestotals18)
#             npnatives18 = numpy.asarray(valuesnatives18)
#             npnonnatives18 = numpy.asarray(valuesnonnatives18)
#             nativepercentage18 = numpy.divide(npnatives18, nptotals18)
#             nonnativepercentage18 = numpy.divide(npnonnatives18, nptotals18)
#             nativemean18 = numpy.around(numpy.mean(nativepercentage18) * 100, 2)
#             nativesd18 = numpy.around(numpy.std(nativepercentage18) * 100, 2)
#             nativesem18 = numpy.around(sem(nativepercentage18) * 100, 2)
#             nonnativemean18 = numpy.around(numpy.mean(nonnativepercentage18) * 100, 2)
#             nonnativesd18 = numpy.around(numpy.std(nonnativepercentage18) * 100, 2)
#             nonnativesem18 = numpy.around(sem(nonnativepercentage18) * 100, 2)
#
#             ########## 2019 ##########
#
#             labelstotals19, countstotals19 = numpy.unique(tot19, return_counts=True)
#             totals19 = {labelstotals19[i]: countstotals19[i] for i in range(len(labelstotals19))}
#             labelsnat19, countsnat = numpy.unique(nat19, return_counts=True)
#             natives19 = {labelsnat19[i]: countsnat[i] for i in range(len(labelsnat19))}
#             labelsnonnat19, countsnonnat19 = numpy.unique(nonnat19, return_counts=True)
#             nonnatives19 = {labelsnonnat19[i]: countsnonnat19[i] for i in range(len(labelsnonnat19))}
#             fulldict19 = {}
#             for key in set(list(totals19.keys()) + list(natives19.keys())):
#                 try:
#                     fulldict19.setdefault(key, []).append(totals19[key])
#                 except KeyError:
#                     fulldict19.setdefault(key, []).append(0)
#                 try:
#                     fulldict19.setdefault(key, []).append(natives19[key])
#                 except KeyError:
#                     fulldict19.setdefault(key, []).append(0)
#                 try:
#                     fulldict19.setdefault(key, []).append(nonnatives19[key])
#                 except KeyError:
#                     fulldict19.setdefault(key, []).append(0)
#             values19 = numpy.array(list(fulldict19.values()))
#             valuestotals19 = [sublist[0] for sublist in values19]
#             valuesnatives19 = [sublist[1] for sublist in values19]
#             valuesnonnatives19 = [sublist[-1] for sublist in values19]
#             nptotals19 = numpy.asarray(valuestotals19)
#             npnatives19 = numpy.asarray(valuesnatives19)
#             npnonnatives19 = numpy.asarray(valuesnonnatives19)
#             nativepercentage19 = numpy.divide(npnatives19, nptotals19)
#             nonnativepercentage19 = numpy.divide(npnonnatives19, nptotals19)
#             nativemean19 = numpy.around(numpy.mean(nativepercentage19) * 100, 2)
#             nativesd19 = numpy.around(numpy.std(nativepercentage19) * 100, 2)
#             nativesem19 = numpy.around(sem(nativepercentage19) * 100, 2)
#             nonnativemean19 = numpy.around(numpy.mean(nonnativepercentage19) * 100, 2)
#             nonnativesd19 = numpy.around(numpy.std(nonnativepercentage19) * 100, 2)
#             nonnativesem19 = numpy.around(sem(nonnativepercentage19) * 100, 2)
#
#             ########## 2020 ##########
#
#             labelstotals20, countstotals20 = numpy.unique(tot20, return_counts=True)
#             totals20 = {labelstotals20[i]: countstotals20[i] for i in range(len(labelstotals20))}
#             labelsnat20, countsnat = numpy.unique(nat20, return_counts=True)
#             natives20 = {labelsnat20[i]: countsnat[i] for i in range(len(labelsnat20))}
#             labelsnonnat20, countsnonnat20 = numpy.unique(nonnat20, return_counts=True)
#             nonnatives20 = {labelsnonnat20[i]: countsnonnat20[i] for i in range(len(labelsnonnat20))}
#             #Dictionary = total counts, native counts, non-native counts
#             fulldict20 = {}
#             for key in set(list(totals20.keys()) + list(natives20.keys())):
#                 try:
#                     fulldict20.setdefault(key, []).append(totals20[key])
#                 except KeyError:
#                     fulldict20.setdefault(key, []).append(0)
#                 try:
#                     fulldict20.setdefault(key, []).append(natives20[key])
#                 except KeyError:
#                     fulldict20.setdefault(key, []).append(0)
#                 try:
#                     fulldict20.setdefault(key, []).append(nonnatives20[key])
#                 except KeyError:
#                     fulldict20.setdefault(key, []).append(0)
#             values20 = numpy.array(list(fulldict20.values()))
#             valuestotals20 = [sublist[0] for sublist in values20]
#             valuesnatives20 = [sublist[1] for sublist in values20]
#             valuesnonnatives20 = [sublist[-1] for sublist in values20]
#             nptotals20 = numpy.asarray(valuestotals20)
#             npnatives20 = numpy.asarray(valuesnatives20)
#             npnonnatives20 = numpy.asarray(valuesnonnatives20)
#             nativepercentage20 = numpy.divide(npnatives20, nptotals20)
#             nonnativepercentage20 = numpy.divide(npnonnatives20, nptotals20)
#             nativemean20 = numpy.around(numpy.mean(nativepercentage20) * 100, 2)
#             nativesd20 = numpy.around(numpy.std(nativepercentage20) * 100, 2)
#             nativesem20 = numpy.around(sem(nativepercentage20) * 100, 2)
#             nonnativemean20 = numpy.around(numpy.mean(nonnativepercentage20) * 100, 2)
#             nonnativesd20 = numpy.around(numpy.std(nonnativepercentage20) * 100, 2)
#             nonnativesem20 = numpy.around(sem(nonnativepercentage20) * 100, 2)
#
# ##### Inside Woolsey #####
#             labelstotals_inW_14, countstotals_inW_14 = numpy.unique(totinW14, return_counts=True)
#             totals_inW_14 = {labelstotals_inW_14[i]: countstotals_inW_14[i] for i in range(len(labelstotals_inW_14))}
#             labelsnat_inW_14, countsnat_inW_ = numpy.unique(natinW14, return_counts=True)
#             natives14 = {labelsnat_inW_14[i]: countsnat_inW_[i] for i in range(len(labelsnat_inW_14))}
#             labelsnonnat_inW_14, countsnonnat_inW_14 = numpy.unique(nonnatinW14, return_counts=True)
#             nonnatives14 = {labelsnonnat_inW_14[i]: countsnonnat_inW_14[i] for i in range(len(labelsnonnat_inW_14))}
#             fulldict_inW_14 = {}
#             for key in set(list(totals_inW_14.keys()) + list(natives14.keys())):
#                 try:
#                     fulldict_inW_14.setdefault(key, []).append(totals_inW_14[key])
#                 except KeyError:
#                     fulldict_inW_14.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_14.setdefault(key, []).append(natives14[key])
#                 except KeyError:
#                     fulldict_inW_14.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_14.setdefault(key, []).append(nonnatives14[key])
#                 except KeyError:
#                     fulldict_inW_14.setdefault(key, []).append(0)
#             values_inW_14 = numpy.array(list(fulldict_inW_14.values()))
#             valuestotals_inW_14 = [sublist[0] for sublist in values_inW_14]
#             valuesnat_inW_ives14 = [sublist[1] for sublist in values_inW_14]
#             valuesnonnatives_inW_14 = [sublist[-1] for sublist in values_inW_14]
#             nptotals_inW_14 = numpy.asarray(valuestotals_inW_14)
#             npnatives_inW_14 = numpy.asarray(valuesnat_inW_ives14)
#             npnonnatives_inW_14 = numpy.asarray(valuesnonnatives_inW_14)
#             nativepercentage_inW_14 = numpy.divide(npnatives_inW_14, nptotals_inW_14)
#             nonnativepercentage_inW_14 = numpy.divide(npnonnatives_inW_14, nptotals_inW_14)
#             nativemean_inW_14 = numpy.around(numpy.mean(nativepercentage_inW_14) * 100, 2)
#             nativesd_inW_14 = numpy.around(numpy.std(nativepercentage_inW_14) * 100, 2)
#             nativesem_inW_14 = numpy.around(sem(nativepercentage_inW_14) * 100, 2)
#             nonnativemean_inW_14 = numpy.around(numpy.mean(nonnativepercentage_inW_14) * 100, 2)
#             nonnativesd_inW_14 = numpy.around(numpy.std(nonnativepercentage_inW_14) * 100, 2)
#             nonnativesem_inW_14 = numpy.around(sem(nonnativepercentage_inW_14) * 100, 2)
#
#             labelstotals_inW_15, countstotals_inW_15 = numpy.unique(totinW15, return_counts=True)
#             totals_inW_15 = {labelstotals_inW_15[i]: countstotals_inW_15[i] for i in range(len(labelstotals_inW_15))}
#             labelsnat_inW_15, countsnat_inW_ = numpy.unique(natinW15, return_counts=True)
#             natives15 = {labelsnat_inW_15[i]: countsnat_inW_[i] for i in range(len(labelsnat_inW_15))}
#             labelsnonnat_inW_15, countsnonnat_inW_15 = numpy.unique(nonnatinW15, return_counts=True)
#             nonnatives15 = {labelsnonnat_inW_15[i]: countsnonnat_inW_15[i] for i in range(len(labelsnonnat_inW_15))}
#             fulldict_inW_15 = {}
#             for key in set(list(totals_inW_15.keys()) + list(natives15.keys())):
#                 try:
#                     fulldict_inW_15.setdefault(key, []).append(totals_inW_15[key])
#                 except KeyError:
#                     fulldict_inW_15.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_15.setdefault(key, []).append(natives15[key])
#                 except KeyError:
#                     fulldict_inW_15.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_15.setdefault(key, []).append(nonnatives15[key])
#                 except KeyError:
#                     fulldict_inW_15.setdefault(key, []).append(0)
#             values_inW_15 = numpy.array(list(fulldict_inW_15.values()))
#             valuestotals_inW_15 = [sublist[0] for sublist in values_inW_15]
#             valuesnat_inW_ives15 = [sublist[1] for sublist in values_inW_15]
#             valuesnonnatives_inW_15 = [sublist[-1] for sublist in values_inW_15]
#             nptotals_inW_15 = numpy.asarray(valuestotals_inW_15)
#             npnatives_inW_15 = numpy.asarray(valuesnat_inW_ives15)
#             npnonnatives_inW_15 = numpy.asarray(valuesnonnatives_inW_15)
#             nativepercentage_inW_15 = numpy.divide(npnatives_inW_15, nptotals_inW_15)
#             nonnativepercentage_inW_15 = numpy.divide(npnonnatives_inW_15, nptotals_inW_15)
#             nativemean_inW_15 = numpy.around(numpy.mean(nativepercentage_inW_15) * 100, 2)
#             nativesd_inW_15 = numpy.around(numpy.std(nativepercentage_inW_15) * 100, 2)
#             nativesem_inW_15 = numpy.around(sem(nativepercentage_inW_15) * 100, 2)
#             nonnativemean_inW_15 = numpy.around(numpy.mean(nonnativepercentage_inW_15) * 100, 2)
#             nonnativesd_inW_15 = numpy.around(numpy.std(nonnativepercentage_inW_15) * 100, 2)
#             nonnativesem_inW_15 = numpy.around(sem(nonnativepercentage_inW_15) * 100, 2)
#
#             labelstotals_inW_16, countstotals_inW_16 = numpy.unique(totinW16, return_counts=True)
#             totals_inW_16 = {labelstotals_inW_16[i]: countstotals_inW_16[i] for i in range(len(labelstotals_inW_16))}
#             labelsnat_inW_16, countsnat_inW_ = numpy.unique(natinW16, return_counts=True)
#             natives16 = {labelsnat_inW_16[i]: countsnat_inW_[i] for i in range(len(labelsnat_inW_16))}
#             labelsnonnat_inW_16, countsnonnat_inW_16 = numpy.unique(nonnatinW16, return_counts=True)
#             nonnatives16 = {labelsnonnat_inW_16[i]: countsnonnat_inW_16[i] for i in range(len(labelsnonnat_inW_16))}
#             fulldict_inW_16 = {}
#             for key in set(list(totals_inW_16.keys()) + list(natives16.keys())):
#                 try:
#                     fulldict_inW_16.setdefault(key, []).append(totals_inW_16[key])
#                 except KeyError:
#                     fulldict_inW_16.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_16.setdefault(key, []).append(natives16[key])
#                 except KeyError:
#                     fulldict_inW_16.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_16.setdefault(key, []).append(nonnatives16[key])
#                 except KeyError:
#                     fulldict_inW_16.setdefault(key, []).append(0)
#             values_inW_16 = numpy.array(list(fulldict_inW_16.values()))
#             valuestotals_inW_16 = [sublist[0] for sublist in values_inW_16]
#             valuesnat_inW_ives16 = [sublist[1] for sublist in values_inW_16]
#             valuesnonnatives_inW_16 = [sublist[-1] for sublist in values_inW_16]
#             nptotals_inW_16 = numpy.asarray(valuestotals_inW_16)
#             npnatives_inW_16 = numpy.asarray(valuesnat_inW_ives16)
#             npnonnatives_inW_16 = numpy.asarray(valuesnonnatives_inW_16)
#             nativepercentage_inW_16 = numpy.divide(npnatives_inW_16, nptotals_inW_16)
#             nonnativepercentage_inW_16 = numpy.divide(npnonnatives_inW_16, nptotals_inW_16)
#             nativemean_inW_16 = numpy.around(numpy.mean(nativepercentage_inW_16) * 100, 2)
#             nativesd_inW_16 = numpy.around(numpy.std(nativepercentage_inW_16) * 100, 2)
#             nativesem_inW_16 = numpy.around(sem(nativepercentage_inW_16) * 100, 2)
#             nonnativemean_inW_16 = numpy.around(numpy.mean(nonnativepercentage_inW_16) * 100, 2)
#             nonnativesd_inW_16 = numpy.around(numpy.std(nonnativepercentage_inW_16) * 100, 2)
#             nonnativesem_inW_16 = numpy.around(sem(nonnativepercentage_inW_16) * 100, 2)
#
#             labelstotals_inW_17, countstotals_inW_17 = numpy.unique(totinW17, return_counts=True)
#             totals_inW_17 = {labelstotals_inW_17[i]: countstotals_inW_17[i] for i in range(len(labelstotals_inW_17))}
#             labelsnat_inW_17, countsnat_inW_ = numpy.unique(natinW17, return_counts=True)
#             natives17 = {labelsnat_inW_17[i]: countsnat_inW_[i] for i in range(len(labelsnat_inW_17))}
#             labelsnonnat_inW_17, countsnonnat_inW_17 = numpy.unique(nonnatinW17, return_counts=True)
#             nonnatives17 = {labelsnonnat_inW_17[i]: countsnonnat_inW_17[i] for i in range(len(labelsnonnat_inW_17))}
#             fulldict_inW_17 = {}
#             for key in set(list(totals_inW_17.keys()) + list(natives17.keys())):
#                 try:
#                     fulldict_inW_17.setdefault(key, []).append(totals_inW_17[key])
#                 except KeyError:
#                     fulldict_inW_17.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_17.setdefault(key, []).append(natives17[key])
#                 except KeyError:
#                     fulldict_inW_17.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_17.setdefault(key, []).append(nonnatives17[key])
#                 except KeyError:
#                     fulldict_inW_17.setdefault(key, []).append(0)
#             values_inW_17 = numpy.array(list(fulldict_inW_17.values()))
#             valuestotals_inW_17 = [sublist[0] for sublist in values_inW_17]
#             valuesnat_inW_ives17 = [sublist[1] for sublist in values_inW_17]
#             valuesnonnatives_inW_17 = [sublist[-1] for sublist in values_inW_17]
#             nptotals_inW_17 = numpy.asarray(valuestotals_inW_17)
#             npnatives_inW_17 = numpy.asarray(valuesnat_inW_ives17)
#             npnonnatives_inW_17 = numpy.asarray(valuesnonnatives_inW_17)
#             nativepercentage_inW_17 = numpy.divide(npnatives_inW_17, nptotals_inW_17)
#             nonnativepercentage_inW_17 = numpy.divide(npnonnatives_inW_17, nptotals_inW_17)
#             nativemean_inW_17 = numpy.around(numpy.mean(nativepercentage_inW_17) * 100, 2)
#             nativesd_inW_17 = numpy.around(numpy.std(nativepercentage_inW_17) * 100, 2)
#             nativesem_inW_17 = numpy.around(sem(nativepercentage_inW_17) * 100, 2)
#             nonnativemean_inW_17 = numpy.around(numpy.mean(nonnativepercentage_inW_17) * 100, 2)
#             nonnativesd_inW_17 = numpy.around(numpy.std(nonnativepercentage_inW_17) * 100, 2)
#             nonnativesem_inW_17 = numpy.around(sem(nonnativepercentage_inW_17) * 100, 2)
#
#             labelstotals_inW_18, countstotals_inW_18 = numpy.unique(totinW18, return_counts=True)
#             totals_inW_18 = {labelstotals_inW_18[i]: countstotals_inW_18[i] for i in range(len(labelstotals_inW_18))}
#             labelsnat_inW_18, countsnat_inW_ = numpy.unique(natinW18, return_counts=True)
#             natives18 = {labelsnat_inW_18[i]: countsnat_inW_[i] for i in range(len(labelsnat_inW_18))}
#             labelsnonnat_inW_18, countsnonnat_inW_18 = numpy.unique(nonnatinW18, return_counts=True)
#             nonnatives18 = {labelsnonnat_inW_18[i]: countsnonnat_inW_18[i] for i in range(len(labelsnonnat_inW_18))}
#             fulldict_inW_18 = {}
#             for key in set(list(totals_inW_18.keys()) + list(natives18.keys())):
#                 try:
#                     fulldict_inW_18.setdefault(key, []).append(totals_inW_18[key])
#                 except KeyError:
#                     fulldict_inW_18.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_18.setdefault(key, []).append(natives18[key])
#                 except KeyError:
#                     fulldict_inW_18.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_18.setdefault(key, []).append(nonnatives18[key])
#                 except KeyError:
#                     fulldict_inW_18.setdefault(key, []).append(0)
#             values_inW_18 = numpy.array(list(fulldict_inW_18.values()))
#             valuestotals_inW_18 = [sublist[0] for sublist in values_inW_18]
#             valuesnat_inW_ives18 = [sublist[1] for sublist in values_inW_18]
#             valuesnonnatives_inW_18 = [sublist[-1] for sublist in values_inW_18]
#             nptotals_inW_18 = numpy.asarray(valuestotals_inW_18)
#             npnatives_inW_18 = numpy.asarray(valuesnat_inW_ives18)
#             npnonnatives_inW_18 = numpy.asarray(valuesnonnatives_inW_18)
#             nativepercentage_inW_18 = numpy.divide(npnatives_inW_18, nptotals_inW_18)
#             nonnativepercentage_inW_18 = numpy.divide(npnonnatives_inW_18, nptotals_inW_18)
#             nativemean_inW_18 = numpy.around(numpy.mean(nativepercentage_inW_18) * 100, 2)
#             nativesd_inW_18 = numpy.around(numpy.std(nativepercentage_inW_18) * 100, 2)
#             nativesem_inW_18 = numpy.around(sem(nativepercentage_inW_18) * 100, 2)
#             nonnativemean_inW_18 = numpy.around(numpy.mean(nonnativepercentage_inW_18) * 100, 2)
#             nonnativesd_inW_18 = numpy.around(numpy.std(nonnativepercentage_inW_18) * 100, 2)
#             nonnativesem_inW_18 = numpy.around(sem(nonnativepercentage_inW_18) * 100, 2)
#
#             labelstotals_inW_19, countstotals_inW_19 = numpy.unique(totinW19, return_counts=True)
#             totals_inW_19 = {labelstotals_inW_19[i]: countstotals_inW_19[i] for i in range(len(labelstotals_inW_19))}
#             labelsnat_inW_19, countsnat_inW_ = numpy.unique(natinW19, return_counts=True)
#             natives19 = {labelsnat_inW_19[i]: countsnat_inW_[i] for i in range(len(labelsnat_inW_19))}
#             labelsnonnat_inW_19, countsnonnat_inW_19 = numpy.unique(nonnatinW19, return_counts=True)
#             nonnatives19 = {labelsnonnat_inW_19[i]: countsnonnat_inW_19[i] for i in range(len(labelsnonnat_inW_19))}
#             fulldict_inW_19 = {}
#             for key in set(list(totals_inW_19.keys()) + list(natives19.keys())):
#                 try:
#                     fulldict_inW_19.setdefault(key, []).append(totals_inW_19[key])
#                 except KeyError:
#                     fulldict_inW_19.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_19.setdefault(key, []).append(natives19[key])
#                 except KeyError:
#                     fulldict_inW_19.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_19.setdefault(key, []).append(nonnatives19[key])
#                 except KeyError:
#                     fulldict_inW_19.setdefault(key, []).append(0)
#             values_inW_19 = numpy.array(list(fulldict_inW_19.values()))
#             valuestotals_inW_19 = [sublist[0] for sublist in values_inW_19]
#             valuesnat_inW_ives19 = [sublist[1] for sublist in values_inW_19]
#             valuesnonnatives_inW_19 = [sublist[-1] for sublist in values_inW_19]
#             nptotals_inW_19 = numpy.asarray(valuestotals_inW_19)
#             npnatives_inW_19 = numpy.asarray(valuesnat_inW_ives19)
#             npnonnatives_inW_19 = numpy.asarray(valuesnonnatives_inW_19)
#             nativepercentage_inW_19 = numpy.divide(npnatives_inW_19, nptotals_inW_19)
#             nonnativepercentage_inW_19 = numpy.divide(npnonnatives_inW_19, nptotals_inW_19)
#             nativemean_inW_19 = numpy.around(numpy.mean(nativepercentage_inW_19) * 100, 2)
#             nativesd_inW_19 = numpy.around(numpy.std(nativepercentage_inW_19) * 100, 2)
#             nativesem_inW_19 = numpy.around(sem(nativepercentage_inW_19) * 100, 2)
#             nonnativemean_inW_19 = numpy.around(numpy.mean(nonnativepercentage_inW_19) * 100, 2)
#             nonnativesd_inW_19 = numpy.around(numpy.std(nonnativepercentage_inW_19) * 100, 2)
#             nonnativesem_inW_19 = numpy.around(sem(nonnativepercentage_inW_19) * 100, 2)
#
#             labelstotals_inW_20, countstotals_inW_20 = numpy.unique(totinW20, return_counts=True)
#             totals_inW_20 = {labelstotals_inW_20[i]: countstotals_inW_20[i] for i in range(len(labelstotals_inW_20))}
#             labelsnat_inW_20, countsnat_inW_ = numpy.unique(natinW20, return_counts=True)
#             natives20 = {labelsnat_inW_20[i]: countsnat_inW_[i] for i in range(len(labelsnat_inW_20))}
#             labelsnonnat_inW_20, countsnonnat_inW_20 = numpy.unique(nonnatinW20, return_counts=True)
#             nonnatives20 = {labelsnonnat_inW_20[i]: countsnonnat_inW_20[i] for i in range(len(labelsnonnat_inW_20))}
#             fulldict_inW_20 = {}
#             for key in set(list(totals_inW_20.keys()) + list(natives20.keys())):
#                 try:
#                     fulldict_inW_20.setdefault(key, []).append(totals_inW_20[key])
#                 except KeyError:
#                     fulldict_inW_20.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_20.setdefault(key, []).append(natives20[key])
#                 except KeyError:
#                     fulldict_inW_20.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_20.setdefault(key, []).append(nonnatives20[key])
#                 except KeyError:
#                     fulldict_inW_20.setdefault(key, []).append(0)
#             values_inW_20 = numpy.array(list(fulldict_inW_20.values()))
#             valuestotals_inW_20 = [sublist[0] for sublist in values_inW_20]
#             valuesnat_inW_ives20 = [sublist[1] for sublist in values_inW_20]
#             valuesnonnatives_inW_20 = [sublist[-1] for sublist in values_inW_20]
#             nptotals_inW_20 = numpy.asarray(valuestotals_inW_20)
#             npnatives_inW_20 = numpy.asarray(valuesnat_inW_ives20)
#             npnonnatives_inW_20 = numpy.asarray(valuesnonnatives_inW_20)
#             nativepercentage_inW_20 = numpy.divide(npnatives_inW_20, nptotals_inW_20)
#             nonnativepercentage_inW_20 = numpy.divide(npnonnatives_inW_20, nptotals_inW_20)
#             nativemean_inW_20 = numpy.around(numpy.mean(nativepercentage_inW_20) * 100, 2)
#             nativesd_inW_20 = numpy.around(numpy.std(nativepercentage_inW_20) * 100, 2)
#             nativesem_inW_20 = numpy.around(sem(nativepercentage_inW_20) * 100, 2)
#             nonnativemean_inW_20 = numpy.around(numpy.mean(nonnativepercentage_inW_20) * 100, 2)
#             nonnativesd_inW_20 = numpy.around(numpy.std(nonnativepercentage_inW_20) * 100, 2)
#             nonnativesem_inW_20 = numpy.around(sem(nonnativepercentage_inW_20) * 100, 2)
#
#             ##### Inside Woolsey #####
#             labelstotals_inW_14, countstotals_inW_14 = numpy.unique(totinW14, return_counts=True)
#             totals_inW_14 = {labelstotals_inW_14[i]: countstotals_inW_14[i] for i in range(len(labelstotals_inW_14))}
#             labelsnat_inW_14, countsnat_inW_ = numpy.unique(natinW14, return_counts=True)
#             natives14 = {labelsnat_inW_14[i]: countsnat_inW_[i] for i in range(len(labelsnat_inW_14))}
#             labelsnonnat_inW_14, countsnonnat_inW_14 = numpy.unique(nonnatinW14, return_counts=True)
#             nonnatives14 = {labelsnonnat_inW_14[i]: countsnonnat_inW_14[i] for i in range(len(labelsnonnat_inW_14))}
#             fulldict_inW_14 = {}
#             for key in set(list(totals_inW_14.keys()) + list(natives14.keys())):
#                 try:
#                     fulldict_inW_14.setdefault(key, []).append(totals_inW_14[key])
#                 except KeyError:
#                     fulldict_inW_14.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_14.setdefault(key, []).append(natives14[key])
#                 except KeyError:
#                     fulldict_inW_14.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_14.setdefault(key, []).append(nonnatives14[key])
#                 except KeyError:
#                     fulldict_inW_14.setdefault(key, []).append(0)
#             values_inW_14 = numpy.array(list(fulldict_inW_14.values()))
#             valuestotals_inW_14 = [sublist[0] for sublist in values_inW_14]
#             valuesnat_inW_ives14 = [sublist[1] for sublist in values_inW_14]
#             valuesnonnatives_inW_14 = [sublist[-1] for sublist in values_inW_14]
#             nptotals_inW_14 = numpy.asarray(valuestotals_inW_14)
#             npnatives_inW_14 = numpy.asarray(valuesnat_inW_ives14)
#             npnonnatives_inW_14 = numpy.asarray(valuesnonnatives_inW_14)
#             nativepercentage_inW_14 = numpy.divide(npnatives_inW_14, nptotals_inW_14)
#             nonnativepercentage_inW_14 = numpy.divide(npnonnatives_inW_14, nptotals_inW_14)
#             nativemean_inW_14 = numpy.around(numpy.mean(nativepercentage_inW_14) * 100, 2)
#             nativesd_inW_14 = numpy.around(numpy.std(nativepercentage_inW_14) * 100, 2)
#             nativesem_inW_14 = numpy.around(sem(nativepercentage_inW_14) * 100, 2)
#             nonnativemean_inW_14 = numpy.around(numpy.mean(nonnativepercentage_inW_14) * 100, 2)
#             nonnativesd_inW_14 = numpy.around(numpy.std(nonnativepercentage_inW_14) * 100, 2)
#             nonnativesem_inW_14 = numpy.around(sem(nonnativepercentage_inW_14) * 100, 2)
#
#             labelstotals_inW_15, countstotals_inW_15 = numpy.unique(totinW15, return_counts=True)
#             totals_inW_15 = {labelstotals_inW_15[i]: countstotals_inW_15[i] for i in range(len(labelstotals_inW_15))}
#             labelsnat_inW_15, countsnat_inW_ = numpy.unique(natinW15, return_counts=True)
#             natives15 = {labelsnat_inW_15[i]: countsnat_inW_[i] for i in range(len(labelsnat_inW_15))}
#             labelsnonnat_inW_15, countsnonnat_inW_15 = numpy.unique(nonnatinW15, return_counts=True)
#             nonnatives15 = {labelsnonnat_inW_15[i]: countsnonnat_inW_15[i] for i in range(len(labelsnonnat_inW_15))}
#             fulldict_inW_15 = {}
#             for key in set(list(totals_inW_15.keys()) + list(natives15.keys())):
#                 try:
#                     fulldict_inW_15.setdefault(key, []).append(totals_inW_15[key])
#                 except KeyError:
#                     fulldict_inW_15.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_15.setdefault(key, []).append(natives15[key])
#                 except KeyError:
#                     fulldict_inW_15.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_15.setdefault(key, []).append(nonnatives15[key])
#                 except KeyError:
#                     fulldict_inW_15.setdefault(key, []).append(0)
#             values_inW_15 = numpy.array(list(fulldict_inW_15.values()))
#             valuestotals_inW_15 = [sublist[0] for sublist in values_inW_15]
#             valuesnat_inW_ives15 = [sublist[1] for sublist in values_inW_15]
#             valuesnonnatives_inW_15 = [sublist[-1] for sublist in values_inW_15]
#             nptotals_inW_15 = numpy.asarray(valuestotals_inW_15)
#             npnatives_inW_15 = numpy.asarray(valuesnat_inW_ives15)
#             npnonnatives_inW_15 = numpy.asarray(valuesnonnatives_inW_15)
#             nativepercentage_inW_15 = numpy.divide(npnatives_inW_15, nptotals_inW_15)
#             nonnativepercentage_inW_15 = numpy.divide(npnonnatives_inW_15, nptotals_inW_15)
#             nativemean_inW_15 = numpy.around(numpy.mean(nativepercentage_inW_15) * 100, 2)
#             nativesd_inW_15 = numpy.around(numpy.std(nativepercentage_inW_15) * 100, 2)
#             nativesem_inW_15 = numpy.around(sem(nativepercentage_inW_15) * 100, 2)
#             nonnativemean_inW_15 = numpy.around(numpy.mean(nonnativepercentage_inW_15) * 100, 2)
#             nonnativesd_inW_15 = numpy.around(numpy.std(nonnativepercentage_inW_15) * 100, 2)
#             nonnativesem_inW_15 = numpy.around(sem(nonnativepercentage_inW_15) * 100, 2)
#
#             labelstotals_inW_16, countstotals_inW_16 = numpy.unique(totinW16, return_counts=True)
#             totals_inW_16 = {labelstotals_inW_16[i]: countstotals_inW_16[i] for i in range(len(labelstotals_inW_16))}
#             labelsnat_inW_16, countsnat_inW_ = numpy.unique(natinW16, return_counts=True)
#             natives16 = {labelsnat_inW_16[i]: countsnat_inW_[i] for i in range(len(labelsnat_inW_16))}
#             labelsnonnat_inW_16, countsnonnat_inW_16 = numpy.unique(nonnatinW16, return_counts=True)
#             nonnatives16 = {labelsnonnat_inW_16[i]: countsnonnat_inW_16[i] for i in range(len(labelsnonnat_inW_16))}
#             fulldict_inW_16 = {}
#             for key in set(list(totals_inW_16.keys()) + list(natives16.keys())):
#                 try:
#                     fulldict_inW_16.setdefault(key, []).append(totals_inW_16[key])
#                 except KeyError:
#                     fulldict_inW_16.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_16.setdefault(key, []).append(natives16[key])
#                 except KeyError:
#                     fulldict_inW_16.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_16.setdefault(key, []).append(nonnatives16[key])
#                 except KeyError:
#                     fulldict_inW_16.setdefault(key, []).append(0)
#             values_inW_16 = numpy.array(list(fulldict_inW_16.values()))
#             valuestotals_inW_16 = [sublist[0] for sublist in values_inW_16]
#             valuesnat_inW_ives16 = [sublist[1] for sublist in values_inW_16]
#             valuesnonnatives_inW_16 = [sublist[-1] for sublist in values_inW_16]
#             nptotals_inW_16 = numpy.asarray(valuestotals_inW_16)
#             npnatives_inW_16 = numpy.asarray(valuesnat_inW_ives16)
#             npnonnatives_inW_16 = numpy.asarray(valuesnonnatives_inW_16)
#             nativepercentage_inW_16 = numpy.divide(npnatives_inW_16, nptotals_inW_16)
#             nonnativepercentage_inW_16 = numpy.divide(npnonnatives_inW_16, nptotals_inW_16)
#             nativemean_inW_16 = numpy.around(numpy.mean(nativepercentage_inW_16) * 100, 2)
#             nativesd_inW_16 = numpy.around(numpy.std(nativepercentage_inW_16) * 100, 2)
#             nativesem_inW_16 = numpy.around(sem(nativepercentage_inW_16) * 100, 2)
#             nonnativemean_inW_16 = numpy.around(numpy.mean(nonnativepercentage_inW_16) * 100, 2)
#             nonnativesd_inW_16 = numpy.around(numpy.std(nonnativepercentage_inW_16) * 100, 2)
#             nonnativesem_inW_16 = numpy.around(sem(nonnativepercentage_inW_16) * 100, 2)
#
#             labelstotals_inW_17, countstotals_inW_17 = numpy.unique(totinW17, return_counts=True)
#             totals_inW_17 = {labelstotals_inW_17[i]: countstotals_inW_17[i] for i in range(len(labelstotals_inW_17))}
#             labelsnat_inW_17, countsnat_inW_ = numpy.unique(natinW17, return_counts=True)
#             natives17 = {labelsnat_inW_17[i]: countsnat_inW_[i] for i in range(len(labelsnat_inW_17))}
#             labelsnonnat_inW_17, countsnonnat_inW_17 = numpy.unique(nonnatinW17, return_counts=True)
#             nonnatives17 = {labelsnonnat_inW_17[i]: countsnonnat_inW_17[i] for i in range(len(labelsnonnat_inW_17))}
#             fulldict_inW_17 = {}
#             for key in set(list(totals_inW_17.keys()) + list(natives17.keys())):
#                 try:
#                     fulldict_inW_17.setdefault(key, []).append(totals_inW_17[key])
#                 except KeyError:
#                     fulldict_inW_17.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_17.setdefault(key, []).append(natives17[key])
#                 except KeyError:
#                     fulldict_inW_17.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_17.setdefault(key, []).append(nonnatives17[key])
#                 except KeyError:
#                     fulldict_inW_17.setdefault(key, []).append(0)
#             values_inW_17 = numpy.array(list(fulldict_inW_17.values()))
#             valuestotals_inW_17 = [sublist[0] for sublist in values_inW_17]
#             valuesnat_inW_ives17 = [sublist[1] for sublist in values_inW_17]
#             valuesnonnatives_inW_17 = [sublist[-1] for sublist in values_inW_17]
#             nptotals_inW_17 = numpy.asarray(valuestotals_inW_17)
#             npnatives_inW_17 = numpy.asarray(valuesnat_inW_ives17)
#             npnonnatives_inW_17 = numpy.asarray(valuesnonnatives_inW_17)
#             nativepercentage_inW_17 = numpy.divide(npnatives_inW_17, nptotals_inW_17)
#             nonnativepercentage_inW_17 = numpy.divide(npnonnatives_inW_17, nptotals_inW_17)
#             nativemean_inW_17 = numpy.around(numpy.mean(nativepercentage_inW_17) * 100, 2)
#             nativesd_inW_17 = numpy.around(numpy.std(nativepercentage_inW_17) * 100, 2)
#             nativesem_inW_17 = numpy.around(sem(nativepercentage_inW_17) * 100, 2)
#             nonnativemean_inW_17 = numpy.around(numpy.mean(nonnativepercentage_inW_17) * 100, 2)
#             nonnativesd_inW_17 = numpy.around(numpy.std(nonnativepercentage_inW_17) * 100, 2)
#             nonnativesem_inW_17 = numpy.around(sem(nonnativepercentage_inW_17) * 100, 2)
#
#             labelstotals_inW_18, countstotals_inW_18 = numpy.unique(totinW18, return_counts=True)
#             totals_inW_18 = {labelstotals_inW_18[i]: countstotals_inW_18[i] for i in range(len(labelstotals_inW_18))}
#             labelsnat_inW_18, countsnat_inW_ = numpy.unique(natinW18, return_counts=True)
#             natives18 = {labelsnat_inW_18[i]: countsnat_inW_[i] for i in range(len(labelsnat_inW_18))}
#             labelsnonnat_inW_18, countsnonnat_inW_18 = numpy.unique(nonnatinW18, return_counts=True)
#             nonnatives18 = {labelsnonnat_inW_18[i]: countsnonnat_inW_18[i] for i in range(len(labelsnonnat_inW_18))}
#             fulldict_inW_18 = {}
#             for key in set(list(totals_inW_18.keys()) + list(natives18.keys())):
#                 try:
#                     fulldict_inW_18.setdefault(key, []).append(totals_inW_18[key])
#                 except KeyError:
#                     fulldict_inW_18.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_18.setdefault(key, []).append(natives18[key])
#                 except KeyError:
#                     fulldict_inW_18.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_18.setdefault(key, []).append(nonnatives18[key])
#                 except KeyError:
#                     fulldict_inW_18.setdefault(key, []).append(0)
#             values_inW_18 = numpy.array(list(fulldict_inW_18.values()))
#             valuestotals_inW_18 = [sublist[0] for sublist in values_inW_18]
#             valuesnat_inW_ives18 = [sublist[1] for sublist in values_inW_18]
#             valuesnonnatives_inW_18 = [sublist[-1] for sublist in values_inW_18]
#             nptotals_inW_18 = numpy.asarray(valuestotals_inW_18)
#             npnatives_inW_18 = numpy.asarray(valuesnat_inW_ives18)
#             npnonnatives_inW_18 = numpy.asarray(valuesnonnatives_inW_18)
#             nativepercentage_inW_18 = numpy.divide(npnatives_inW_18, nptotals_inW_18)
#             nonnativepercentage_inW_18 = numpy.divide(npnonnatives_inW_18, nptotals_inW_18)
#             nativemean_inW_18 = numpy.around(numpy.mean(nativepercentage_inW_18) * 100, 2)
#             nativesd_inW_18 = numpy.around(numpy.std(nativepercentage_inW_18) * 100, 2)
#             nativesem_inW_18 = numpy.around(sem(nativepercentage_inW_18) * 100, 2)
#             nonnativemean_inW_18 = numpy.around(numpy.mean(nonnativepercentage_inW_18) * 100, 2)
#             nonnativesd_inW_18 = numpy.around(numpy.std(nonnativepercentage_inW_18) * 100, 2)
#             nonnativesem_inW_18 = numpy.around(sem(nonnativepercentage_inW_18) * 100, 2)
#
#             labelstotals_inW_19, countstotals_inW_19 = numpy.unique(totinW19, return_counts=True)
#             totals_inW_19 = {labelstotals_inW_19[i]: countstotals_inW_19[i] for i in range(len(labelstotals_inW_19))}
#             labelsnat_inW_19, countsnat_inW_ = numpy.unique(natinW19, return_counts=True)
#             natives19 = {labelsnat_inW_19[i]: countsnat_inW_[i] for i in range(len(labelsnat_inW_19))}
#             labelsnonnat_inW_19, countsnonnat_inW_19 = numpy.unique(nonnatinW19, return_counts=True)
#             nonnatives19 = {labelsnonnat_inW_19[i]: countsnonnat_inW_19[i] for i in range(len(labelsnonnat_inW_19))}
#             fulldict_inW_19 = {}
#             for key in set(list(totals_inW_19.keys()) + list(natives19.keys())):
#                 try:
#                     fulldict_inW_19.setdefault(key, []).append(totals_inW_19[key])
#                 except KeyError:
#                     fulldict_inW_19.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_19.setdefault(key, []).append(natives19[key])
#                 except KeyError:
#                     fulldict_inW_19.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_19.setdefault(key, []).append(nonnatives19[key])
#                 except KeyError:
#                     fulldict_inW_19.setdefault(key, []).append(0)
#             values_inW_19 = numpy.array(list(fulldict_inW_19.values()))
#             valuestotals_inW_19 = [sublist[0] for sublist in values_inW_19]
#             valuesnat_inW_ives19 = [sublist[1] for sublist in values_inW_19]
#             valuesnonnatives_inW_19 = [sublist[-1] for sublist in values_inW_19]
#             nptotals_inW_19 = numpy.asarray(valuestotals_inW_19)
#             npnatives_inW_19 = numpy.asarray(valuesnat_inW_ives19)
#             npnonnatives_inW_19 = numpy.asarray(valuesnonnatives_inW_19)
#             nativepercentage_inW_19 = numpy.divide(npnatives_inW_19, nptotals_inW_19)
#             nonnativepercentage_inW_19 = numpy.divide(npnonnatives_inW_19, nptotals_inW_19)
#             nativemean_inW_19 = numpy.around(numpy.mean(nativepercentage_inW_19) * 100, 2)
#             nativesd_inW_19 = numpy.around(numpy.std(nativepercentage_inW_19) * 100, 2)
#             nativesem_inW_19 = numpy.around(sem(nativepercentage_inW_19) * 100, 2)
#             nonnativemean_inW_19 = numpy.around(numpy.mean(nonnativepercentage_inW_19) * 100, 2)
#             nonnativesd_inW_19 = numpy.around(numpy.std(nonnativepercentage_inW_19) * 100, 2)
#             nonnativesem_inW_19 = numpy.around(sem(nonnativepercentage_inW_19) * 100, 2)
#
#             labelstotals_inW_20, countstotals_inW_20 = numpy.unique(totinW20, return_counts=True)
#             totals_inW_20 = {labelstotals_inW_20[i]: countstotals_inW_20[i] for i in range(len(labelstotals_inW_20))}
#             labelsnat_inW_20, countsnat_inW_ = numpy.unique(natinW20, return_counts=True)
#             natives20 = {labelsnat_inW_20[i]: countsnat_inW_[i] for i in range(len(labelsnat_inW_20))}
#             labelsnonnat_inW_20, countsnonnat_inW_20 = numpy.unique(nonnatinW20, return_counts=True)
#             nonnatives20 = {labelsnonnat_inW_20[i]: countsnonnat_inW_20[i] for i in range(len(labelsnonnat_inW_20))}
#             fulldict_inW_20 = {}
#             for key in set(list(totals_inW_20.keys()) + list(natives20.keys())):
#                 try:
#                     fulldict_inW_20.setdefault(key, []).append(totals_inW_20[key])
#                 except KeyError:
#                     fulldict_inW_20.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_20.setdefault(key, []).append(natives20[key])
#                 except KeyError:
#                     fulldict_inW_20.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_20.setdefault(key, []).append(nonnatives20[key])
#                 except KeyError:
#                     fulldict_inW_20.setdefault(key, []).append(0)
#             values_inW_20 = numpy.array(list(fulldict_inW_20.values()))
#             valuestotals_inW_20 = [sublist[0] for sublist in values_inW_20]
#             valuesnat_inW_ives20 = [sublist[1] for sublist in values_inW_20]
#             valuesnonnatives_inW_20 = [sublist[-1] for sublist in values_inW_20]
#             nptotals_inW_20 = numpy.asarray(valuestotals_inW_20)
#             npnatives_inW_20 = numpy.asarray(valuesnat_inW_ives20)
#             npnonnatives_inW_20 = numpy.asarray(valuesnonnatives_inW_20)
#             nativepercentage_inW_20 = numpy.divide(npnatives_inW_20, nptotals_inW_20)
#             nonnativepercentage_inW_20 = numpy.divide(npnonnatives_inW_20, nptotals_inW_20)
#             nativemean_inW_20 = numpy.around(numpy.mean(nativepercentage_inW_20) * 100, 2)
#             nativesd_inW_20 = numpy.around(numpy.std(nativepercentage_inW_20) * 100, 2)
#             nativesem_inW_20 = numpy.around(sem(nativepercentage_inW_20) * 100, 2)
#             nonnativemean_inW_20 = numpy.around(numpy.mean(nonnativepercentage_inW_20) * 100, 2)
#             nonnativesd_inW_20 = numpy.around(numpy.std(nonnativepercentage_inW_20) * 100, 2)
#             nonnativesem_inW_20 = numpy.around(sem(nonnativepercentage_inW_20) * 100, 2)
#
#             ##### Inside Woolsey #####
#             labelstotals_inW_14, countstotals_inW_14 = numpy.unique(totinW14, return_counts=True)
#             totals_inW_14 = {labelstotals_inW_14[i]: countstotals_inW_14[i] for i in range(len(labelstotals_inW_14))}
#             labelsnat_inW_14, countsnat_inW_ = numpy.unique(natinW14, return_counts=True)
#             natives14 = {labelsnat_inW_14[i]: countsnat_inW_[i] for i in range(len(labelsnat_inW_14))}
#             labelsnonnat_inW_14, countsnonnat_inW_14 = numpy.unique(nonnatinW14, return_counts=True)
#             nonnatives14 = {labelsnonnat_inW_14[i]: countsnonnat_inW_14[i] for i in range(len(labelsnonnat_inW_14))}
#             fulldict_inW_14 = {}
#             for key in set(list(totals_inW_14.keys()) + list(natives14.keys())):
#                 try:
#                     fulldict_inW_14.setdefault(key, []).append(totals_inW_14[key])
#                 except KeyError:
#                     fulldict_inW_14.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_14.setdefault(key, []).append(natives14[key])
#                 except KeyError:
#                     fulldict_inW_14.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_14.setdefault(key, []).append(nonnatives14[key])
#                 except KeyError:
#                     fulldict_inW_14.setdefault(key, []).append(0)
#             values_inW_14 = numpy.array(list(fulldict_inW_14.values()))
#             valuestotals_inW_14 = [sublist[0] for sublist in values_inW_14]
#             valuesnat_inW_ives14 = [sublist[1] for sublist in values_inW_14]
#             valuesnonnatives_inW_14 = [sublist[-1] for sublist in values_inW_14]
#             nptotals_inW_14 = numpy.asarray(valuestotals_inW_14)
#             npnatives_inW_14 = numpy.asarray(valuesnat_inW_ives14)
#             npnonnatives_inW_14 = numpy.asarray(valuesnonnatives_inW_14)
#             nativepercentage_inW_14 = numpy.divide(npnatives_inW_14, nptotals_inW_14)
#             nonnativepercentage_inW_14 = numpy.divide(npnonnatives_inW_14, nptotals_inW_14)
#             nativemean_inW_14 = numpy.around(numpy.mean(nativepercentage_inW_14) * 100, 2)
#             nativesd_inW_14 = numpy.around(numpy.std(nativepercentage_inW_14) * 100, 2)
#             nativesem_inW_14 = numpy.around(sem(nativepercentage_inW_14) * 100, 2)
#             nonnativemean_inW_14 = numpy.around(numpy.mean(nonnativepercentage_inW_14) * 100, 2)
#             nonnativesd_inW_14 = numpy.around(numpy.std(nonnativepercentage_inW_14) * 100, 2)
#             nonnativesem_inW_14 = numpy.around(sem(nonnativepercentage_inW_14) * 100, 2)
#
#             labelstotals_inW_15, countstotals_inW_15 = numpy.unique(totinW15, return_counts=True)
#             totals_inW_15 = {labelstotals_inW_15[i]: countstotals_inW_15[i] for i in range(len(labelstotals_inW_15))}
#             labelsnat_inW_15, countsnat_inW_ = numpy.unique(natinW15, return_counts=True)
#             natives15 = {labelsnat_inW_15[i]: countsnat_inW_[i] for i in range(len(labelsnat_inW_15))}
#             labelsnonnat_inW_15, countsnonnat_inW_15 = numpy.unique(nonnatinW15, return_counts=True)
#             nonnatives15 = {labelsnonnat_inW_15[i]: countsnonnat_inW_15[i] for i in range(len(labelsnonnat_inW_15))}
#             fulldict_inW_15 = {}
#             for key in set(list(totals_inW_15.keys()) + list(natives15.keys())):
#                 try:
#                     fulldict_inW_15.setdefault(key, []).append(totals_inW_15[key])
#                 except KeyError:
#                     fulldict_inW_15.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_15.setdefault(key, []).append(natives15[key])
#                 except KeyError:
#                     fulldict_inW_15.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_15.setdefault(key, []).append(nonnatives15[key])
#                 except KeyError:
#                     fulldict_inW_15.setdefault(key, []).append(0)
#             values_inW_15 = numpy.array(list(fulldict_inW_15.values()))
#             valuestotals_inW_15 = [sublist[0] for sublist in values_inW_15]
#             valuesnat_inW_ives15 = [sublist[1] for sublist in values_inW_15]
#             valuesnonnatives_inW_15 = [sublist[-1] for sublist in values_inW_15]
#             nptotals_inW_15 = numpy.asarray(valuestotals_inW_15)
#             npnatives_inW_15 = numpy.asarray(valuesnat_inW_ives15)
#             npnonnatives_inW_15 = numpy.asarray(valuesnonnatives_inW_15)
#             nativepercentage_inW_15 = numpy.divide(npnatives_inW_15, nptotals_inW_15)
#             nonnativepercentage_inW_15 = numpy.divide(npnonnatives_inW_15, nptotals_inW_15)
#             nativemean_inW_15 = numpy.around(numpy.mean(nativepercentage_inW_15) * 100, 2)
#             nativesd_inW_15 = numpy.around(numpy.std(nativepercentage_inW_15) * 100, 2)
#             nativesem_inW_15 = numpy.around(sem(nativepercentage_inW_15) * 100, 2)
#             nonnativemean_inW_15 = numpy.around(numpy.mean(nonnativepercentage_inW_15) * 100, 2)
#             nonnativesd_inW_15 = numpy.around(numpy.std(nonnativepercentage_inW_15) * 100, 2)
#             nonnativesem_inW_15 = numpy.around(sem(nonnativepercentage_inW_15) * 100, 2)
#
#             labelstotals_inW_16, countstotals_inW_16 = numpy.unique(totinW16, return_counts=True)
#             totals_inW_16 = {labelstotals_inW_16[i]: countstotals_inW_16[i] for i in range(len(labelstotals_inW_16))}
#             labelsnat_inW_16, countsnat_inW_ = numpy.unique(natinW16, return_counts=True)
#             natives16 = {labelsnat_inW_16[i]: countsnat_inW_[i] for i in range(len(labelsnat_inW_16))}
#             labelsnonnat_inW_16, countsnonnat_inW_16 = numpy.unique(nonnatinW16, return_counts=True)
#             nonnatives16 = {labelsnonnat_inW_16[i]: countsnonnat_inW_16[i] for i in range(len(labelsnonnat_inW_16))}
#             fulldict_inW_16 = {}
#             for key in set(list(totals_inW_16.keys()) + list(natives16.keys())):
#                 try:
#                     fulldict_inW_16.setdefault(key, []).append(totals_inW_16[key])
#                 except KeyError:
#                     fulldict_inW_16.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_16.setdefault(key, []).append(natives16[key])
#                 except KeyError:
#                     fulldict_inW_16.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_16.setdefault(key, []).append(nonnatives16[key])
#                 except KeyError:
#                     fulldict_inW_16.setdefault(key, []).append(0)
#             values_inW_16 = numpy.array(list(fulldict_inW_16.values()))
#             valuestotals_inW_16 = [sublist[0] for sublist in values_inW_16]
#             valuesnat_inW_ives16 = [sublist[1] for sublist in values_inW_16]
#             valuesnonnatives_inW_16 = [sublist[-1] for sublist in values_inW_16]
#             nptotals_inW_16 = numpy.asarray(valuestotals_inW_16)
#             npnatives_inW_16 = numpy.asarray(valuesnat_inW_ives16)
#             npnonnatives_inW_16 = numpy.asarray(valuesnonnatives_inW_16)
#             nativepercentage_inW_16 = numpy.divide(npnatives_inW_16, nptotals_inW_16)
#             nonnativepercentage_inW_16 = numpy.divide(npnonnatives_inW_16, nptotals_inW_16)
#             nativemean_inW_16 = numpy.around(numpy.mean(nativepercentage_inW_16) * 100, 2)
#             nativesd_inW_16 = numpy.around(numpy.std(nativepercentage_inW_16) * 100, 2)
#             nativesem_inW_16 = numpy.around(sem(nativepercentage_inW_16) * 100, 2)
#             nonnativemean_inW_16 = numpy.around(numpy.mean(nonnativepercentage_inW_16) * 100, 2)
#             nonnativesd_inW_16 = numpy.around(numpy.std(nonnativepercentage_inW_16) * 100, 2)
#             nonnativesem_inW_16 = numpy.around(sem(nonnativepercentage_inW_16) * 100, 2)
#
#             labelstotals_inW_17, countstotals_inW_17 = numpy.unique(totinW17, return_counts=True)
#             totals_inW_17 = {labelstotals_inW_17[i]: countstotals_inW_17[i] for i in range(len(labelstotals_inW_17))}
#             labelsnat_inW_17, countsnat_inW_ = numpy.unique(natinW17, return_counts=True)
#             natives17 = {labelsnat_inW_17[i]: countsnat_inW_[i] for i in range(len(labelsnat_inW_17))}
#             labelsnonnat_inW_17, countsnonnat_inW_17 = numpy.unique(nonnatinW17, return_counts=True)
#             nonnatives17 = {labelsnonnat_inW_17[i]: countsnonnat_inW_17[i] for i in range(len(labelsnonnat_inW_17))}
#             fulldict_inW_17 = {}
#             for key in set(list(totals_inW_17.keys()) + list(natives17.keys())):
#                 try:
#                     fulldict_inW_17.setdefault(key, []).append(totals_inW_17[key])
#                 except KeyError:
#                     fulldict_inW_17.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_17.setdefault(key, []).append(natives17[key])
#                 except KeyError:
#                     fulldict_inW_17.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_17.setdefault(key, []).append(nonnatives17[key])
#                 except KeyError:
#                     fulldict_inW_17.setdefault(key, []).append(0)
#             values_inW_17 = numpy.array(list(fulldict_inW_17.values()))
#             valuestotals_inW_17 = [sublist[0] for sublist in values_inW_17]
#             valuesnat_inW_ives17 = [sublist[1] for sublist in values_inW_17]
#             valuesnonnatives_inW_17 = [sublist[-1] for sublist in values_inW_17]
#             nptotals_inW_17 = numpy.asarray(valuestotals_inW_17)
#             npnatives_inW_17 = numpy.asarray(valuesnat_inW_ives17)
#             npnonnatives_inW_17 = numpy.asarray(valuesnonnatives_inW_17)
#             nativepercentage_inW_17 = numpy.divide(npnatives_inW_17, nptotals_inW_17)
#             nonnativepercentage_inW_17 = numpy.divide(npnonnatives_inW_17, nptotals_inW_17)
#             nativemean_inW_17 = numpy.around(numpy.mean(nativepercentage_inW_17) * 100, 2)
#             nativesd_inW_17 = numpy.around(numpy.std(nativepercentage_inW_17) * 100, 2)
#             nativesem_inW_17 = numpy.around(sem(nativepercentage_inW_17) * 100, 2)
#             nonnativemean_inW_17 = numpy.around(numpy.mean(nonnativepercentage_inW_17) * 100, 2)
#             nonnativesd_inW_17 = numpy.around(numpy.std(nonnativepercentage_inW_17) * 100, 2)
#             nonnativesem_inW_17 = numpy.around(sem(nonnativepercentage_inW_17) * 100, 2)
#
#             labelstotals_inW_18, countstotals_inW_18 = numpy.unique(totinW18, return_counts=True)
#             totals_inW_18 = {labelstotals_inW_18[i]: countstotals_inW_18[i] for i in range(len(labelstotals_inW_18))}
#             labelsnat_inW_18, countsnat_inW_ = numpy.unique(natinW18, return_counts=True)
#             natives18 = {labelsnat_inW_18[i]: countsnat_inW_[i] for i in range(len(labelsnat_inW_18))}
#             labelsnonnat_inW_18, countsnonnat_inW_18 = numpy.unique(nonnatinW18, return_counts=True)
#             nonnatives18 = {labelsnonnat_inW_18[i]: countsnonnat_inW_18[i] for i in range(len(labelsnonnat_inW_18))}
#             fulldict_inW_18 = {}
#             for key in set(list(totals_inW_18.keys()) + list(natives18.keys())):
#                 try:
#                     fulldict_inW_18.setdefault(key, []).append(totals_inW_18[key])
#                 except KeyError:
#                     fulldict_inW_18.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_18.setdefault(key, []).append(natives18[key])
#                 except KeyError:
#                     fulldict_inW_18.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_18.setdefault(key, []).append(nonnatives18[key])
#                 except KeyError:
#                     fulldict_inW_18.setdefault(key, []).append(0)
#             values_inW_18 = numpy.array(list(fulldict_inW_18.values()))
#             valuestotals_inW_18 = [sublist[0] for sublist in values_inW_18]
#             valuesnat_inW_ives18 = [sublist[1] for sublist in values_inW_18]
#             valuesnonnatives_inW_18 = [sublist[-1] for sublist in values_inW_18]
#             nptotals_inW_18 = numpy.asarray(valuestotals_inW_18)
#             npnatives_inW_18 = numpy.asarray(valuesnat_inW_ives18)
#             npnonnatives_inW_18 = numpy.asarray(valuesnonnatives_inW_18)
#             nativepercentage_inW_18 = numpy.divide(npnatives_inW_18, nptotals_inW_18)
#             nonnativepercentage_inW_18 = numpy.divide(npnonnatives_inW_18, nptotals_inW_18)
#             nativemean_inW_18 = numpy.around(numpy.mean(nativepercentage_inW_18) * 100, 2)
#             nativesd_inW_18 = numpy.around(numpy.std(nativepercentage_inW_18) * 100, 2)
#             nativesem_inW_18 = numpy.around(sem(nativepercentage_inW_18) * 100, 2)
#             nonnativemean_inW_18 = numpy.around(numpy.mean(nonnativepercentage_inW_18) * 100, 2)
#             nonnativesd_inW_18 = numpy.around(numpy.std(nonnativepercentage_inW_18) * 100, 2)
#             nonnativesem_inW_18 = numpy.around(sem(nonnativepercentage_inW_18) * 100, 2)
#
#             labelstotals_inW_19, countstotals_inW_19 = numpy.unique(totinW19, return_counts=True)
#             totals_inW_19 = {labelstotals_inW_19[i]: countstotals_inW_19[i] for i in range(len(labelstotals_inW_19))}
#             labelsnat_inW_19, countsnat_inW_ = numpy.unique(natinW19, return_counts=True)
#             natives19 = {labelsnat_inW_19[i]: countsnat_inW_[i] for i in range(len(labelsnat_inW_19))}
#             labelsnonnat_inW_19, countsnonnat_inW_19 = numpy.unique(nonnatinW19, return_counts=True)
#             nonnatives19 = {labelsnonnat_inW_19[i]: countsnonnat_inW_19[i] for i in range(len(labelsnonnat_inW_19))}
#             fulldict_inW_19 = {}
#             for key in set(list(totals_inW_19.keys()) + list(natives19.keys())):
#                 try:
#                     fulldict_inW_19.setdefault(key, []).append(totals_inW_19[key])
#                 except KeyError:
#                     fulldict_inW_19.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_19.setdefault(key, []).append(natives19[key])
#                 except KeyError:
#                     fulldict_inW_19.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_19.setdefault(key, []).append(nonnatives19[key])
#                 except KeyError:
#                     fulldict_inW_19.setdefault(key, []).append(0)
#             values_inW_19 = numpy.array(list(fulldict_inW_19.values()))
#             valuestotals_inW_19 = [sublist[0] for sublist in values_inW_19]
#             valuesnat_inW_ives19 = [sublist[1] for sublist in values_inW_19]
#             valuesnonnatives_inW_19 = [sublist[-1] for sublist in values_inW_19]
#             nptotals_inW_19 = numpy.asarray(valuestotals_inW_19)
#             npnatives_inW_19 = numpy.asarray(valuesnat_inW_ives19)
#             npnonnatives_inW_19 = numpy.asarray(valuesnonnatives_inW_19)
#             nativepercentage_inW_19 = numpy.divide(npnatives_inW_19, nptotals_inW_19)
#             nonnativepercentage_inW_19 = numpy.divide(npnonnatives_inW_19, nptotals_inW_19)
#             nativemean_inW_19 = numpy.around(numpy.mean(nativepercentage_inW_19) * 100, 2)
#             nativesd_inW_19 = numpy.around(numpy.std(nativepercentage_inW_19) * 100, 2)
#             nativesem_inW_19 = numpy.around(sem(nativepercentage_inW_19) * 100, 2)
#             nonnativemean_inW_19 = numpy.around(numpy.mean(nonnativepercentage_inW_19) * 100, 2)
#             nonnativesd_inW_19 = numpy.around(numpy.std(nonnativepercentage_inW_19) * 100, 2)
#             nonnativesem_inW_19 = numpy.around(sem(nonnativepercentage_inW_19) * 100, 2)
#
#             labelstotals_inW_20, countstotals_inW_20 = numpy.unique(totinW20, return_counts=True)
#             totals_inW_20 = {labelstotals_inW_20[i]: countstotals_inW_20[i] for i in range(len(labelstotals_inW_20))}
#             labelsnat_inW_20, countsnat_inW_ = numpy.unique(natinW20, return_counts=True)
#             natives20 = {labelsnat_inW_20[i]: countsnat_inW_[i] for i in range(len(labelsnat_inW_20))}
#             labelsnonnat_inW_20, countsnonnat_inW_20 = numpy.unique(nonnatinW20, return_counts=True)
#             nonnatives20 = {labelsnonnat_inW_20[i]: countsnonnat_inW_20[i] for i in range(len(labelsnonnat_inW_20))}
#             fulldict_inW_20 = {}
#             for key in set(list(totals_inW_20.keys()) + list(natives20.keys())):
#                 try:
#                     fulldict_inW_20.setdefault(key, []).append(totals_inW_20[key])
#                 except KeyError:
#                     fulldict_inW_20.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_20.setdefault(key, []).append(natives20[key])
#                 except KeyError:
#                     fulldict_inW_20.setdefault(key, []).append(0)
#                 try:
#                     fulldict_inW_20.setdefault(key, []).append(nonnatives20[key])
#                 except KeyError:
#                     fulldict_inW_20.setdefault(key, []).append(0)
#             values_inW_20 = numpy.array(list(fulldict_inW_20.values()))
#             valuestotals_inW_20 = [sublist[0] for sublist in values_inW_20]
#             valuesnat_inW_ives20 = [sublist[1] for sublist in values_inW_20]
#             valuesnonnatives_inW_20 = [sublist[-1] for sublist in values_inW_20]
#             nptotals_inW_20 = numpy.asarray(valuestotals_inW_20)
#             npnatives_inW_20 = numpy.asarray(valuesnat_inW_ives20)
#             npnonnatives_inW_20 = numpy.asarray(valuesnonnatives_inW_20)
#             nativepercentage_inW_20 = numpy.divide(npnatives_inW_20, nptotals_inW_20)
#             nonnativepercentage_inW_20 = numpy.divide(npnonnatives_inW_20, nptotals_inW_20)
#             nativemean_inW_20 = numpy.around(numpy.mean(nativepercentage_inW_20) * 100, 2)
#             nativesd_inW_20 = numpy.around(numpy.std(nativepercentage_inW_20) * 100, 2)
#             nativesem_inW_20 = numpy.around(sem(nativepercentage_inW_20) * 100, 2)
#             nonnativemean_inW_20 = numpy.around(numpy.mean(nonnativepercentage_inW_20) * 100, 2)
#             nonnativesd_inW_20 = numpy.around(numpy.std(nonnativepercentage_inW_20) * 100, 2)
#             nonnativesem_inW_20 = numpy.around(sem(nonnativepercentage_inW_20) * 100, 2)
#
#             ##### Outside Woolsey #####
#             labelstotals_outW_14, countstotals_outW_14 = numpy.unique(totoutW14, return_counts=True)
#             totals_outW_14 = {labelstotals_outW_14[i]: countstotals_outW_14[i] for i in range(len(labelstotals_outW_14))}
#             labelsnat_outW_14, countsnat_outW_ = numpy.unique(natoutW14, return_counts=True)
#             natives14 = {labelsnat_outW_14[i]: countsnat_outW_[i] for i in range(len(labelsnat_outW_14))}
#             labelsnonnat_outW_14, countsnonnat_outW_14 = numpy.unique(nonnatoutW14, return_counts=True)
#             nonnatives14 = {labelsnonnat_outW_14[i]: countsnonnat_outW_14[i] for i in range(len(labelsnonnat_outW_14))}
#             fulldict_outW_14 = {}
#             for key in set(list(totals_outW_14.keys()) + list(natives14.keys())):
#                 try:
#                     fulldict_outW_14.setdefault(key, []).append(totals_outW_14[key])
#                 except KeyError:
#                     fulldict_outW_14.setdefault(key, []).append(0)
#                 try:
#                     fulldict_outW_14.setdefault(key, []).append(natives14[key])
#                 except KeyError:
#                     fulldict_outW_14.setdefault(key, []).append(0)
#                 try:
#                     fulldict_outW_14.setdefault(key, []).append(nonnatives14[key])
#                 except KeyError:
#                     fulldict_outW_14.setdefault(key, []).append(0)
#             values_outW_14 = numpy.array(list(fulldict_outW_14.values()))
#             valuestotals_outW_14 = [sublist[0] for sublist in values_outW_14]
#             valuesnat_outW_ives14 = [sublist[1] for sublist in values_outW_14]
#             valuesnonnatives_outW_14 = [sublist[-1] for sublist in values_outW_14]
#             nptotals_outW_14 = numpy.asarray(valuestotals_outW_14)
#             npnatives_outW_14 = numpy.asarray(valuesnat_outW_ives14)
#             npnonnatives_outW_14 = numpy.asarray(valuesnonnatives_outW_14)
#             nativepercentage_outW_14 = numpy.divide(npnatives_outW_14, nptotals_outW_14)
#             nonnativepercentage_outW_14 = numpy.divide(npnonnatives_outW_14, nptotals_outW_14)
#             nativemean_outW_14 = numpy.around(numpy.mean(nativepercentage_outW_14) * 100, 2)
#             nativesd_outW_14 = numpy.around(numpy.std(nativepercentage_outW_14) * 100, 2)
#             nativesem_outW_14 = numpy.around(sem(nativepercentage_outW_14) * 100, 2)
#             nonnativemean_outW_14 = numpy.around(numpy.mean(nonnativepercentage_outW_14) * 100, 2)
#             nonnativesd_outW_14 = numpy.around(numpy.std(nonnativepercentage_outW_14) * 100, 2)
#             nonnativesem_outW_14 = numpy.around(sem(nonnativepercentage_outW_14) * 100, 2)
#
#             labelstotals_outW_15, countstotals_outW_15 = numpy.unique(totoutW15, return_counts=True)
#             totals_outW_15 = {labelstotals_outW_15[i]: countstotals_outW_15[i] for i in range(len(labelstotals_outW_15))}
#             labelsnat_outW_15, countsnat_outW_ = numpy.unique(natoutW15, return_counts=True)
#             natives15 = {labelsnat_outW_15[i]: countsnat_outW_[i] for i in range(len(labelsnat_outW_15))}
#             labelsnonnat_outW_15, countsnonnat_outW_15 = numpy.unique(nonnatoutW15, return_counts=True)
#             nonnatives15 = {labelsnonnat_outW_15[i]: countsnonnat_outW_15[i] for i in range(len(labelsnonnat_outW_15))}
#             fulldict_outW_15 = {}
#             for key in set(list(totals_outW_15.keys()) + list(natives15.keys())):
#                 try:
#                     fulldict_outW_15.setdefault(key, []).append(totals_outW_15[key])
#                 except KeyError:
#                     fulldict_outW_15.setdefault(key, []).append(0)
#                 try:
#                     fulldict_outW_15.setdefault(key, []).append(natives15[key])
#                 except KeyError:
#                     fulldict_outW_15.setdefault(key, []).append(0)
#                 try:
#                     fulldict_outW_15.setdefault(key, []).append(nonnatives15[key])
#                 except KeyError:
#                     fulldict_outW_15.setdefault(key, []).append(0)
#             values_outW_15 = numpy.array(list(fulldict_outW_15.values()))
#             valuestotals_outW_15 = [sublist[0] for sublist in values_outW_15]
#             valuesnat_outW_ives15 = [sublist[1] for sublist in values_outW_15]
#             valuesnonnatives_outW_15 = [sublist[-1] for sublist in values_outW_15]
#             nptotals_outW_15 = numpy.asarray(valuestotals_outW_15)
#             npnatives_outW_15 = numpy.asarray(valuesnat_outW_ives15)
#             npnonnatives_outW_15 = numpy.asarray(valuesnonnatives_outW_15)
#             nativepercentage_outW_15 = numpy.divide(npnatives_outW_15, nptotals_outW_15)
#             nonnativepercentage_outW_15 = numpy.divide(npnonnatives_outW_15, nptotals_outW_15)
#             nativemean_outW_15 = numpy.around(numpy.mean(nativepercentage_outW_15) * 100, 2)
#             nativesd_outW_15 = numpy.around(numpy.std(nativepercentage_outW_15) * 100, 2)
#             nativesem_outW_15 = numpy.around(sem(nativepercentage_outW_15) * 100, 2)
#             nonnativemean_outW_15 = numpy.around(numpy.mean(nonnativepercentage_outW_15) * 100, 2)
#             nonnativesd_outW_15 = numpy.around(numpy.std(nonnativepercentage_outW_15) * 100, 2)
#             nonnativesem_outW_15 = numpy.around(sem(nonnativepercentage_outW_15) * 100, 2)
#
#             labelstotals_outW_16, countstotals_outW_16 = numpy.unique(totoutW16, return_counts=True)
#             totals_outW_16 = {labelstotals_outW_16[i]: countstotals_outW_16[i] for i in range(len(labelstotals_outW_16))}
#             labelsnat_outW_16, countsnat_outW_ = numpy.unique(natoutW16, return_counts=True)
#             natives16 = {labelsnat_outW_16[i]: countsnat_outW_[i] for i in range(len(labelsnat_outW_16))}
#             labelsnonnat_outW_16, countsnonnat_outW_16 = numpy.unique(nonnatoutW16, return_counts=True)
#             nonnatives16 = {labelsnonnat_outW_16[i]: countsnonnat_outW_16[i] for i in range(len(labelsnonnat_outW_16))}
#             fulldict_outW_16 = {}
#             for key in set(list(totals_outW_16.keys()) + list(natives16.keys())):
#                 try:
#                     fulldict_outW_16.setdefault(key, []).append(totals_outW_16[key])
#                 except KeyError:
#                     fulldict_outW_16.setdefault(key, []).append(0)
#                 try:
#                     fulldict_outW_16.setdefault(key, []).append(natives16[key])
#                 except KeyError:
#                     fulldict_outW_16.setdefault(key, []).append(0)
#                 try:
#                     fulldict_outW_16.setdefault(key, []).append(nonnatives16[key])
#                 except KeyError:
#                     fulldict_outW_16.setdefault(key, []).append(0)
#             values_outW_16 = numpy.array(list(fulldict_outW_16.values()))
#             valuestotals_outW_16 = [sublist[0] for sublist in values_outW_16]
#             valuesnat_outW_ives16 = [sublist[1] for sublist in values_outW_16]
#             valuesnonnatives_outW_16 = [sublist[-1] for sublist in values_outW_16]
#             nptotals_outW_16 = numpy.asarray(valuestotals_outW_16)
#             npnatives_outW_16 = numpy.asarray(valuesnat_outW_ives16)
#             npnonnatives_outW_16 = numpy.asarray(valuesnonnatives_outW_16)
#             nativepercentage_outW_16 = numpy.divide(npnatives_outW_16, nptotals_outW_16)
#             nonnativepercentage_outW_16 = numpy.divide(npnonnatives_outW_16, nptotals_outW_16)
#             nativemean_outW_16 = numpy.around(numpy.mean(nativepercentage_outW_16) * 100, 2)
#             nativesd_outW_16 = numpy.around(numpy.std(nativepercentage_outW_16) * 100, 2)
#             nativesem_outW_16 = numpy.around(sem(nativepercentage_outW_16) * 100, 2)
#             nonnativemean_outW_16 = numpy.around(numpy.mean(nonnativepercentage_outW_16) * 100, 2)
#             nonnativesd_outW_16 = numpy.around(numpy.std(nonnativepercentage_outW_16) * 100, 2)
#             nonnativesem_outW_16 = numpy.around(sem(nonnativepercentage_outW_16) * 100, 2)
#
#             labelstotals_outW_17, countstotals_outW_17 = numpy.unique(totoutW17, return_counts=True)
#             totals_outW_17 = {labelstotals_outW_17[i]: countstotals_outW_17[i] for i in range(len(labelstotals_outW_17))}
#             labelsnat_outW_17, countsnat_outW_ = numpy.unique(natoutW17, return_counts=True)
#             natives17 = {labelsnat_outW_17[i]: countsnat_outW_[i] for i in range(len(labelsnat_outW_17))}
#             labelsnonnat_outW_17, countsnonnat_outW_17 = numpy.unique(nonnatoutW17, return_counts=True)
#             nonnatives17 = {labelsnonnat_outW_17[i]: countsnonnat_outW_17[i] for i in range(len(labelsnonnat_outW_17))}
#             fulldict_outW_17 = {}
#             for key in set(list(totals_outW_17.keys()) + list(natives17.keys())):
#                 try:
#                     fulldict_outW_17.setdefault(key, []).append(totals_outW_17[key])
#                 except KeyError:
#                     fulldict_outW_17.setdefault(key, []).append(0)
#                 try:
#                     fulldict_outW_17.setdefault(key, []).append(natives17[key])
#                 except KeyError:
#                     fulldict_outW_17.setdefault(key, []).append(0)
#                 try:
#                     fulldict_outW_17.setdefault(key, []).append(nonnatives17[key])
#                 except KeyError:
#                     fulldict_outW_17.setdefault(key, []).append(0)
#             values_outW_17 = numpy.array(list(fulldict_outW_17.values()))
#             valuestotals_outW_17 = [sublist[0] for sublist in values_outW_17]
#             valuesnat_outW_ives17 = [sublist[1] for sublist in values_outW_17]
#             valuesnonnatives_outW_17 = [sublist[-1] for sublist in values_outW_17]
#             nptotals_outW_17 = numpy.asarray(valuestotals_outW_17)
#             npnatives_outW_17 = numpy.asarray(valuesnat_outW_ives17)
#             npnonnatives_outW_17 = numpy.asarray(valuesnonnatives_outW_17)
#             nativepercentage_outW_17 = numpy.divide(npnatives_outW_17, nptotals_outW_17)
#             nonnativepercentage_outW_17 = numpy.divide(npnonnatives_outW_17, nptotals_outW_17)
#             nativemean_outW_17 = numpy.around(numpy.mean(nativepercentage_outW_17) * 100, 2)
#             nativesd_outW_17 = numpy.around(numpy.std(nativepercentage_outW_17) * 100, 2)
#             nativesem_outW_17 = numpy.around(sem(nativepercentage_outW_17) * 100, 2)
#             nonnativemean_outW_17 = numpy.around(numpy.mean(nonnativepercentage_outW_17) * 100, 2)
#             nonnativesd_outW_17 = numpy.around(numpy.std(nonnativepercentage_outW_17) * 100, 2)
#             nonnativesem_outW_17 = numpy.around(sem(nonnativepercentage_outW_17) * 100, 2)
#
#             labelstotals_outW_18, countstotals_outW_18 = numpy.unique(totoutW18, return_counts=True)
#             totals_outW_18 = {labelstotals_outW_18[i]: countstotals_outW_18[i] for i in range(len(labelstotals_outW_18))}
#             labelsnat_outW_18, countsnat_outW_ = numpy.unique(natoutW18, return_counts=True)
#             natives18 = {labelsnat_outW_18[i]: countsnat_outW_[i] for i in range(len(labelsnat_outW_18))}
#             labelsnonnat_outW_18, countsnonnat_outW_18 = numpy.unique(nonnatoutW18, return_counts=True)
#             nonnatives18 = {labelsnonnat_outW_18[i]: countsnonnat_outW_18[i] for i in range(len(labelsnonnat_outW_18))}
#             fulldict_outW_18 = {}
#             for key in set(list(totals_outW_18.keys()) + list(natives18.keys())):
#                 try:
#                     fulldict_outW_18.setdefault(key, []).append(totals_outW_18[key])
#                 except KeyError:
#                     fulldict_outW_18.setdefault(key, []).append(0)
#                 try:
#                     fulldict_outW_18.setdefault(key, []).append(natives18[key])
#                 except KeyError:
#                     fulldict_outW_18.setdefault(key, []).append(0)
#                 try:
#                     fulldict_outW_18.setdefault(key, []).append(nonnatives18[key])
#                 except KeyError:
#                     fulldict_outW_18.setdefault(key, []).append(0)
#             values_outW_18 = numpy.array(list(fulldict_outW_18.values()))
#             valuestotals_outW_18 = [sublist[0] for sublist in values_outW_18]
#             valuesnat_outW_ives18 = [sublist[1] for sublist in values_outW_18]
#             valuesnonnatives_outW_18 = [sublist[-1] for sublist in values_outW_18]
#             nptotals_outW_18 = numpy.asarray(valuestotals_outW_18)
#             npnatives_outW_18 = numpy.asarray(valuesnat_outW_ives18)
#             npnonnatives_outW_18 = numpy.asarray(valuesnonnatives_outW_18)
#             nativepercentage_outW_18 = numpy.divide(npnatives_outW_18, nptotals_outW_18)
#             nonnativepercentage_outW_18 = numpy.divide(npnonnatives_outW_18, nptotals_outW_18)
#             nativemean_outW_18 = numpy.around(numpy.mean(nativepercentage_outW_18) * 100, 2)
#             nativesd_outW_18 = numpy.around(numpy.std(nativepercentage_outW_18) * 100, 2)
#             nativesem_outW_18 = numpy.around(sem(nativepercentage_outW_18) * 100, 2)
#             nonnativemean_outW_18 = numpy.around(numpy.mean(nonnativepercentage_outW_18) * 100, 2)
#             nonnativesd_outW_18 = numpy.around(numpy.std(nonnativepercentage_outW_18) * 100, 2)
#             nonnativesem_outW_18 = numpy.around(sem(nonnativepercentage_outW_18) * 100, 2)
#
#             labelstotals_outW_19, countstotals_outW_19 = numpy.unique(totoutW19, return_counts=True)
#             totals_outW_19 = {labelstotals_outW_19[i]: countstotals_outW_19[i] for i in range(len(labelstotals_outW_19))}
#             labelsnat_outW_19, countsnat_outW_ = numpy.unique(natoutW19, return_counts=True)
#             natives19 = {labelsnat_outW_19[i]: countsnat_outW_[i] for i in range(len(labelsnat_outW_19))}
#             labelsnonnat_outW_19, countsnonnat_outW_19 = numpy.unique(nonnatoutW19, return_counts=True)
#             nonnatives19 = {labelsnonnat_outW_19[i]: countsnonnat_outW_19[i] for i in range(len(labelsnonnat_outW_19))}
#             fulldict_outW_19 = {}
#             for key in set(list(totals_outW_19.keys()) + list(natives19.keys())):
#                 try:
#                     fulldict_outW_19.setdefault(key, []).append(totals_outW_19[key])
#                 except KeyError:
#                     fulldict_outW_19.setdefault(key, []).append(0)
#                 try:
#                     fulldict_outW_19.setdefault(key, []).append(natives19[key])
#                 except KeyError:
#                     fulldict_outW_19.setdefault(key, []).append(0)
#                 try:
#                     fulldict_outW_19.setdefault(key, []).append(nonnatives19[key])
#                 except KeyError:
#                     fulldict_outW_19.setdefault(key, []).append(0)
#             values_outW_19 = numpy.array(list(fulldict_outW_19.values()))
#             valuestotals_outW_19 = [sublist[0] for sublist in values_outW_19]
#             valuesnat_outW_ives19 = [sublist[1] for sublist in values_outW_19]
#             valuesnonnatives_outW_19 = [sublist[-1] for sublist in values_outW_19]
#             nptotals_outW_19 = numpy.asarray(valuestotals_outW_19)
#             npnatives_outW_19 = numpy.asarray(valuesnat_outW_ives19)
#             npnonnatives_outW_19 = numpy.asarray(valuesnonnatives_outW_19)
#             nativepercentage_outW_19 = numpy.divide(npnatives_outW_19, nptotals_outW_19)
#             nonnativepercentage_outW_19 = numpy.divide(npnonnatives_outW_19, nptotals_outW_19)
#             nativemean_outW_19 = numpy.around(numpy.mean(nativepercentage_outW_19) * 100, 2)
#             nativesd_outW_19 = numpy.around(numpy.std(nativepercentage_outW_19) * 100, 2)
#             nativesem_outW_19 = numpy.around(sem(nativepercentage_outW_19) * 100, 2)
#             nonnativemean_outW_19 = numpy.around(numpy.mean(nonnativepercentage_outW_19) * 100, 2)
#             nonnativesd_outW_19 = numpy.around(numpy.std(nonnativepercentage_outW_19) * 100, 2)
#             nonnativesem_outW_19 = numpy.around(sem(nonnativepercentage_outW_19) * 100, 2)
#
#             labelstotals_outW_20, countstotals_outW_20 = numpy.unique(totoutW20, return_counts=True)
#             totals_outW_20 = {labelstotals_outW_20[i]: countstotals_outW_20[i] for i in range(len(labelstotals_outW_20))}
#             labelsnat_outW_20, countsnat_outW_ = numpy.unique(natoutW20, return_counts=True)
#             natives20 = {labelsnat_outW_20[i]: countsnat_outW_[i] for i in range(len(labelsnat_outW_20))}
#             labelsnonnat_outW_20, countsnonnat_outW_20 = numpy.unique(nonnatoutW20, return_counts=True)
#             nonnatives20 = {labelsnonnat_outW_20[i]: countsnonnat_outW_20[i] for i in range(len(labelsnonnat_outW_20))}
#             fulldict_outW_20 = {}
#             for key in set(list(totals_outW_20.keys()) + list(natives20.keys())):
#                 try:
#                     fulldict_outW_20.setdefault(key, []).append(totals_outW_20[key])
#                 except KeyError:
#                     fulldict_outW_20.setdefault(key, []).append(0)
#                 try:
#                     fulldict_outW_20.setdefault(key, []).append(natives20[key])
#                 except KeyError:
#                     fulldict_outW_20.setdefault(key, []).append(0)
#                 try:
#                     fulldict_outW_20.setdefault(key, []).append(nonnatives20[key])
#                 except KeyError:
#                     fulldict_outW_20.setdefault(key, []).append(0)
#             values_outW_20 = numpy.array(list(fulldict_outW_20.values()))
#             valuestotals_outW_20 = [sublist[0] for sublist in values_outW_20]
#             valuesnat_outW_ives20 = [sublist[1] for sublist in values_outW_20]
#             valuesnonnatives_outW_20 = [sublist[-1] for sublist in values_outW_20]
#             nptotals_outW_20 = numpy.asarray(valuestotals_outW_20)
#             npnatives_outW_20 = numpy.asarray(valuesnat_outW_ives20)
#             npnonnatives_outW_20 = numpy.asarray(valuesnonnatives_outW_20)
#             nativepercentage_outW_20 = numpy.divide(npnatives_outW_20, nptotals_outW_20)
#             nonnativepercentage_outW_20 = numpy.divide(npnonnatives_outW_20, nptotals_outW_20)
#             nativemean_outW_20 = numpy.around(numpy.mean(nativepercentage_outW_20) * 100, 2)
#             nativesd_outW_20 = numpy.around(numpy.std(nativepercentage_outW_20) * 100, 2)
#             nativesem_outW_20 = numpy.around(sem(nativepercentage_outW_20) * 100, 2)
#             nonnativemean_outW_20 = numpy.around(numpy.mean(nonnativepercentage_outW_20) * 100, 2)
#             nonnativesd_outW_20 = numpy.around(numpy.std(nonnativepercentage_outW_20) * 100, 2)
#             nonnativesem_outW_20 = numpy.around(sem(nonnativepercentage_outW_20) * 100, 2)
#
#             native_wholepark_mean = [nativemean14, nativemean15, nativemean16, nativemean17, nativemean18, nativemean19, nativemean20]
#             native_wholepark_sd = [nativesd14, nativesd15, nativesd16, nativesd17, nativesd18, nativesd19, nativesd20]
#             native_wholepark_sem = [nativesem14, nativesem15, nativesem16, nativesem17, nativesem18, nativesem19, nativesem20]
#             nonnative_wholepark_mean = [nonnativemean14, nonnativemean15, nonnativemean16, nonnativemean17, nonnativemean18, nonnativemean19, nonnativemean20]
#             nonnative_wholepark_sd = [nonnativesd14, nonnativesd15, nonnativesd16, nonnativesd17, nonnativesd18, nonnativesd19, nonnativesd20]
#             nonnative_wholepark_sem = [nonnativesem14, nonnativesem15, nonnativesem16, nonnativesem17, nonnativesem18, nonnativesem19, nonnativesem20]
#
#             native_inW_mean = [nativemean_inW_17, nativemean_inW_18, nativemean_inW_19, nativemean_inW_20]
#             native_inW_sd = [nativesd_inW_17, nativesd_inW_18, nativesd_inW_19, nativesd_inW_20]
#             native_inW_sem = [nativesem_inW_17, nativesem_inW_18, nativesem_inW_19, nativesem_inW_20]
#             nonnative_inW_mean = [nonnativemean_inW_17, nonnativemean_inW_18, nonnativemean_inW_19, nonnativemean_inW_20]
#             nonnative_inW_sd = [nonnativesd_inW_17, nonnativesd_inW_18, nonnativesd_inW_19, nonnativesd_inW_20]
#             nonnative_inW_sem = [nonnativesem_inW_17, nonnativesem_inW_18, nonnativesem_inW_19, nonnativesem_inW_20]
#
#             native_outW_mean = [nativemean_outW_17, nativemean_outW_18, nativemean_outW_19, nativemean_outW_20]
#             native_outW_sd = [nativesd_outW_17, nativesd_outW_18, nativesd_outW_19, nativesd_outW_20]
#             native_outW_sem = [nativesem_outW_17, nativesem_outW_18, nativesem_outW_19, nativesem_outW_20]
#             nonnative_outW_mean = [nonnativemean_outW_17, nonnativemean_outW_18, nonnativemean_outW_19, nonnativemean_outW_20]
#             nonnative_outW_sd = [nonnativesd_outW_17, nonnativesd_outW_18, nonnativesd_outW_19, nonnativesd_outW_20]
#             nonnative_outW_sem = [nonnativesem_outW_17, nonnativesem_outW_18, nonnativesem_outW_19, nonnativesem_outW_20]
#
#             years = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
#             years_Woolsey = [2017, 2018, 2019, 2020]
#
#             fig, ax = plt.subplots(1, figsize=(6, 3))
#             ax.tick_params(labelsize=ea_axisnum)
#             ax.errorbar(years, native_wholepark_mean, yerr=native_wholepark_sem, color=blue, marker='.', capsize=3)
#             ax.errorbar(years, nonnative_wholepark_mean, yerr=nonnative_wholepark_sem, color=rust, marker='.', capsize=3)
#             ax.set_ylabel('Cover (%)', fontsize=ea_axislabel)
#             ax.set_xlabel('Year', fontsize=ea_axislabel)
#             ax.axvline(x=2018.5, color='black', ls='--', lw=1)
#             plt.tight_layout()
#             plt.savefig("/Users/nicolettastork/Documents/Geography/SMMNRA_Invasive_Species/NPSData/VegData_FromSMMNRA_Modified/Fig_NativeNonnative_WholePark.png", dpi=600)
#             # plt.show()
#
#             fig, ax = plt.subplots(1, figsize=(4, 3))
#             ax.tick_params(labelsize=ea_axisnum)
#             ax.errorbar(years_Woolsey, native_inW_mean, yerr=native_inW_sem, color=darkblue, marker='.', capsize=3)
#             ax.errorbar(years_Woolsey, nonnative_inW_mean, yerr=nonnative_inW_sem, color=darkred, marker='.', capsize=3)
#             ax.errorbar(years_Woolsey, native_outW_mean, yerr=native_outW_sem, linestyle='--', color=blue, marker='.', capsize=3)
#             ax.errorbar(years_Woolsey, nonnative_outW_mean, yerr=nonnative_outW_sem, linestyle='--', color=rust, marker='.', capsize=3)
#             ax.set_ylabel('Cover (%)', fontsize=ea_axislabel)
#             ax.set_xticks(years_Woolsey)
#
#             ax.set_xlabel('Year', fontsize=ea_axislabel)
#             ax.axvline(x=2018.5, color='black', ls='--', lw=1)
#             plt.tight_layout()
#             plt.savefig("/Users/nicolettastork/Documents/Geography/SMMNRA_Invasive_Species/NPSData/VegData_FromSMMNRA_Modified/Fig_NativeNonnative_Woolsey.png", dpi=600)
#             # plt.show()
#
#             ###### RAW NUMBERS TABLE ######
#             # table_natnonnat = [['years:', 2014, 2015, 2016, 2017, 2018, 2019, 2020],
#             #                    ['-----------------------------'],
#             #                    ['WHOLE PARK'],
#             #                    ['native mean:', numpy.around(nativemean14), numpy.around(nativemean15),
#             #                                      numpy.around(nativemean16), numpy.around(nativemean17),
#             #                                      numpy.around(nativemean18), numpy.around(nativemean19),
#             #                                      numpy.around(nativemean20)],
#             #                    ['native standard error:', nativesem14, nativesem15, nativesem16, nativesem17,
#             #                                                 nativesem18, nativesem19, nativesem20],
#             #                    ['native standard deviation:', nativesd14, nativesd15, nativesd16, nativesd17,
#             #                                                 nativesd18, nativesd19, nativesd20],
#             #                    ['nonnative mean:', numpy.around(nonnativemean14), numpy.around(nonnativemean15),
#             #                                         numpy.around(nonnativemean16), numpy.around(nonnativemean17),
#             #                                         numpy.around(nonnativemean18), numpy.around(nonnativemean19),
#             #                                         numpy.around(nonnativemean20)],
#             #                    ['nonnative standard error:', nonnativesem14, nonnativesem15, nonnativesem16,
#             #                                         nonnativesem17, nonnativesem18, nonnativesem19, nonnativesem20],
#             #                    ['nonnative standard deviation:', nonnativesd14, nonnativesd15, nonnativesd16,
#             #                                         nonnativesd17, nonnativesd18, nonnativesd19, nonnativesd20]]
#             # table_natnonnat_Woolsey = [['years:', 2017, 2018, 2019, 2020],
#             #                    ['-----------------------------'],
#             #                    ['INSIDE WOOLSEY BOUNDARY'],
#             #                    ['native mean:',
#             #                     numpy.around(nativemean_inW_17),
#             #                     numpy.around(nativemean_inW_18),
#             #                     numpy.around(nativemean_inW_19),
#             #                     numpy.around(nativemean_inW_20)],
#             #                    ['native standard error:',
#             #                     nativesem_inW_17, nativesem_inW_18, nativesem_inW_19, nativesem_inW_20],
#             #                    ['native standard deviation:',
#             #                     nativesd_inW_17, nativesd_inW_18,  nativesd_inW_19, nativesd_inW_20],
#             #                    ['nonnative mean:', numpy.around(nonnativemean_inW_17),
#             #                     numpy.around(nonnativemean_inW_18), numpy.around(nonnativemean_inW_19),
#             #                     numpy.around(nonnativemean_inW_20)],
#             #                    ['nonnative standard error:',
#             #                     nonnativesem_inW_17, nonnativesem_inW_18, nonnativesem_inW_19, nonnativesem_inW_20],
#             #                    ['nonnative standard deviation:',
#             #                     nonnativesd_inW_17, nonnativesd_inW_18, nonnativesd_inW_19, nonnativesd_inW_20],
#             #                    ['-----------------------------'],
#             #                    ['OUTSIDE WOOLSEY BOUNDARY'],
#             #                    ['native mean:', numpy.around(nativemean_outW_17),
#             #                     numpy.around(nativemean_outW_18), numpy.around(nativemean_outW_19),
#             #                     numpy.around(nativemean_outW_20)],
#             #                    ['native standard error:', nativesem_outW_17,
#             #                     nativesem_outW_18, nativesem_outW_19, nativesem_outW_20],
#             #                    ['native standard deviation:', nativesd_outW_17,
#             #                     nativesd_outW_18, nativesd_outW_19, nativesd_outW_20],
#             #                    ['nonnative mean:', numpy.around(nonnativemean_outW_17),
#             #                     numpy.around(nonnativemean_outW_18), numpy.around(nonnativemean_outW_19),
#             #                     numpy.around(nonnativemean_outW_20)],
#             #                    ['nonnative standard error:',
#             #                     nonnativesem_outW_17, nonnativesem_outW_18, nonnativesem_outW_19, nonnativesem_outW_20],
#             #                    ['nonnative standard deviation:',
#             #                     nonnativesd_outW_17, nonnativesd_outW_18, nonnativesd_outW_19, nonnativesd_outW_20]]
#
#             # print(tabulate(table_natnonnat))
#             # print(tabulate(table_natnonnat_Woolsey))
#
# totalsdict(sitecode_totals_2014, sitecode_native2014, sitecode_nonnative2014,
#           sitecode_totals_2015, sitecode_native2015, sitecode_nonnative2015,
#           sitecode_totals_2016, sitecode_native2016, sitecode_nonnative2016,
#           sitecode_totals_2017, sitecode_native2017, sitecode_nonnative2017,
#           sitecode_totals_2018, sitecode_native2018, sitecode_nonnative2018,
#           sitecode_totals_2019, sitecode_native2019, sitecode_nonnative2019,
#           sitecode_totals_2020, sitecode_native2020, sitecode_nonnative2020,
#            sitecode_inW_totals_2014, sitecode_inW_native2014, sitecode_inW_nonnative2014,
#            sitecode_inW_totals_2015, sitecode_inW_native2015, sitecode_inW_nonnative2015,
#            sitecode_inW_totals_2016, sitecode_inW_native2016, sitecode_inW_nonnative2016,
#            sitecode_inW_totals_2017, sitecode_inW_native2017, sitecode_inW_nonnative2017,
#            sitecode_inW_totals_2018, sitecode_inW_native2018, sitecode_inW_nonnative2018,
#            sitecode_inW_totals_2019, sitecode_inW_native2019, sitecode_inW_nonnative2019,
#            sitecode_inW_totals_2020, sitecode_inW_native2020, sitecode_inW_nonnative2020,
#            sitecode_outW_totals_2014, sitecode_outW_native2014, sitecode_outW_nonnative2014,
#            sitecode_outW_totals_2015, sitecode_outW_native2015, sitecode_outW_nonnative2015,
#            sitecode_outW_totals_2016, sitecode_outW_native2016, sitecode_outW_nonnative2016,
#            sitecode_outW_totals_2017, sitecode_outW_native2017, sitecode_outW_nonnative2017,
#            sitecode_outW_totals_2018, sitecode_outW_native2018, sitecode_outW_nonnative2018,
#            sitecode_outW_totals_2019, sitecode_outW_native2019, sitecode_outW_nonnative2019,
#            sitecode_outW_totals_2020, sitecode_outW_native2020, sitecode_outW_nonnative2020)
#
#
#######################################################################
#### FIGURE 3: RAREFACTION - Inside and outside Woolsey Fire #####
#######################################################################
# totalobsinW_17 = 0
# totalobsinW_18 = 0
# totalobsinW_19 = 0
# totalobsinW_20 = 0
# totalobsoutW_17 = 0
# totalobsoutW_18 = 0
# totalobsoutW_19 = 0
# totalobsoutW_20 = 0
#
# SpeciesListInWoolsey2017 = numpy.array([])
# SpeciesListInWoolsey2018 = numpy.array([])
# SpeciesListInWoolsey2019 = numpy.array([])
# SpeciesListInWoolsey2020 = numpy.array([])
#
# SpeciesListOutWoolsey2017 = numpy.array([])
# SpeciesListOutWoolsey2018 = numpy.array([])
# SpeciesListOutWoolsey2019 = numpy.array([])
# SpeciesListOutWoolsey2020 = numpy.array([])
#
# SpeciesListNativeInWoolsey2017 = numpy.array([])
# SpeciesListNativeInWoolsey2018 = numpy.array([])
# SpeciesListNativeInWoolsey2019 = numpy.array([])
# SpeciesListNativeInWoolsey2020 = numpy.array([])
#
# SpeciesListNativeOutWoolsey2017 = numpy.array([])
# SpeciesListNativeOutWoolsey2018 = numpy.array([])
# SpeciesListNativeOutWoolsey2019 = numpy.array([])
# SpeciesListNativeOutWoolsey2020 = numpy.array([])
#
# SpeciesListNonnativeInWoolsey2017 = numpy.array([])
# SpeciesListNonnativeInWoolsey2018 = numpy.array([])
# SpeciesListNonnativeInWoolsey2019 = numpy.array([])
# SpeciesListNonnativeInWoolsey2020 = numpy.array([])
#
# SpeciesListNonnativeOutWoolsey2017 = numpy.array([])
# SpeciesListNonnativeOutWoolsey2018 = numpy.array([])
# SpeciesListNonnativeOutWoolsey2019 = numpy.array([])
# SpeciesListNonnativeOutWoolsey2020 = numpy.array([])
#
# for name, obj in monitoringpoints.items():
#     test_indices = numpy.where((obj.year == 2017) & (obj.inWoolsey == True))
#     totalobsinW_17 += len(obj.sitecode[test_indices])
#     # print('totalobsinw_17:', totalobsinW_17)
#     test_indices = numpy.where((obj.year == 2018) & (obj.inWoolsey == True))
#     totalobsinW_18 += len(obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2019) & (obj.inWoolsey == True))
#     totalobsinW_19 += len(obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2020) & (obj.inWoolsey == True))
#     totalobsinW_20 += len(obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2017) & (obj.inWoolsey == False))
#     totalobsoutW_17 += len(obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2018) & (obj.inWoolsey == False))
#     totalobsoutW_18 += len(obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2019) & (obj.inWoolsey == False))
#     totalobsoutW_19 += len(obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.year == 2020) & (obj.inWoolsey == False))
#     totalobsoutW_20 += len(obj.sitecode[test_indices])
#
#     test_indices = numpy.where((obj.year == 2017) & (obj.inWoolsey == True))
#     SpeciesListInWoolsey2017 = numpy.append(SpeciesListInWoolsey2017, obj.species_code[test_indices])
#     test_indices = numpy.where((obj.year == 2018) & (obj.inWoolsey == True))
#     SpeciesListInWoolsey2018 = numpy.append(SpeciesListInWoolsey2018, obj.species_code[test_indices])
#     test_indices = numpy.where((obj.year == 2019) & (obj.inWoolsey == True))
#     SpeciesListInWoolsey2019 = numpy.append(SpeciesListInWoolsey2019, obj.species_code[test_indices])
#     test_indices = numpy.where((obj.year == 2020) & (obj.inWoolsey == True))
#     SpeciesListInWoolsey2020 = numpy.append(SpeciesListInWoolsey2020, obj.species_code[test_indices])
#
#     test_indices = numpy.where((obj.year == 2017) & (obj.inWoolsey == False))
#     SpeciesListOutWoolsey2017 = numpy.append(SpeciesListOutWoolsey2017, obj.species_code[test_indices])
#     test_indices = numpy.where((obj.year == 2018) & (obj.inWoolsey == False))
#     SpeciesListOutWoolsey2018 = numpy.append(SpeciesListOutWoolsey2018, obj.species_code[test_indices])
#     test_indices = numpy.where((obj.year == 2019) & (obj.inWoolsey == False))
#     SpeciesListOutWoolsey2019 = numpy.append(SpeciesListOutWoolsey2019, obj.species_code[test_indices])
#     test_indices = numpy.where((obj.year == 2020) & (obj.inWoolsey == False))
#     SpeciesListOutWoolsey2020 = numpy.append(SpeciesListOutWoolsey2020, obj.species_code[test_indices])
#
#     test_indices = numpy.where((obj.year == 2017) & (obj.native_status == "Native") & (obj.inWoolsey == True))
#     SpeciesListNativeInWoolsey2017 = numpy.append(SpeciesListNativeInWoolsey2017, obj.species_code[test_indices])
#     test_indices = numpy.where((obj.year == 2018) & (obj.native_status == "Native") & (obj.inWoolsey == True))
#     SpeciesListNativeInWoolsey2018 = numpy.append(SpeciesListNativeInWoolsey2018, obj.species_code[test_indices])
#     test_indices = numpy.where((obj.year == 2019) & (obj.native_status == "Native") & (obj.inWoolsey == True))
#     SpeciesListNativeInWoolsey2019 = numpy.append(SpeciesListNativeInWoolsey2019, obj.species_code[test_indices])
#     test_indices = numpy.where((obj.year == 2020) & (obj.native_status == "Native") & (obj.inWoolsey == True))
#     SpeciesListNativeInWoolsey2020 = numpy.append(SpeciesListNativeInWoolsey2020, obj.species_code[test_indices])
#
#     test_indices = numpy.where((obj.year == 2017) & (obj.native_status == "Native") & (obj.inWoolsey == False))
#     SpeciesListNativeOutWoolsey2017 = numpy.append(SpeciesListNativeOutWoolsey2017, obj.species_code[test_indices])
#     test_indices = numpy.where((obj.year == 2018) & (obj.native_status == "Native") & (obj.inWoolsey == False))
#     SpeciesListNativeOutWoolsey2018 = numpy.append(SpeciesListNativeOutWoolsey2018, obj.species_code[test_indices])
#     test_indices = numpy.where((obj.year == 2019) & (obj.native_status == "Native") & (obj.inWoolsey == False))
#     SpeciesListNativeOutWoolsey2019 = numpy.append(SpeciesListNativeOutWoolsey2019, obj.species_code[test_indices])
#     test_indices = numpy.where((obj.year == 2020) & (obj.native_status == "Native") & (obj.inWoolsey == False))
#     SpeciesListNativeOutWoolsey2020 = numpy.append(SpeciesListNativeOutWoolsey2020, obj.species_code[test_indices])
#
#     test_indices = numpy.where((obj.year == 2017) & (obj.native_status == "Nonnative") & (obj.inWoolsey == True))
#     SpeciesListNonnativeInWoolsey2017 = numpy.append(SpeciesListNonnativeInWoolsey2017, obj.species_code[test_indices])
#     test_indices = numpy.where((obj.year == 2018) & (obj.native_status == "Nonnative") & (obj.inWoolsey == True))
#     SpeciesListNonnativeInWoolsey2018 = numpy.append(SpeciesListNonnativeInWoolsey2018, obj.species_code[test_indices])
#     test_indices = numpy.where((obj.year == 2019) & (obj.native_status == "Nonnative") & (obj.inWoolsey == True))
#     SpeciesListNonnativeInWoolsey2019 = numpy.append(SpeciesListNonnativeInWoolsey2019, obj.species_code[test_indices])
#     test_indices = numpy.where((obj.year == 2020) & (obj.native_status == "Nonnative") & (obj.inWoolsey == True))
#     SpeciesListNonnativeInWoolsey2020 = numpy.append(SpeciesListNonnativeInWoolsey2020, obj.species_code[test_indices])
#
#     test_indices = numpy.where((obj.year == 2017) & (obj.native_status == "Nonnative") & (obj.inWoolsey == False))
#     SpeciesListNonnativeOutWoolsey2017 = numpy.append(SpeciesListNonnativeOutWoolsey2017, obj.species_code[test_indices])
#     test_indices = numpy.where((obj.year == 2018) & (obj.native_status == "Nonnative") & (obj.inWoolsey == False))
#     SpeciesListNonnativeOutWoolsey2018 = numpy.append(SpeciesListNonnativeOutWoolsey2018, obj.species_code[test_indices])
#     test_indices = numpy.where((obj.year == 2019) & (obj.native_status == "Nonnative") & (obj.inWoolsey == False))
#     SpeciesListNonnativeOutWoolsey2019 = numpy.append(SpeciesListNonnativeOutWoolsey2019, obj.species_code[test_indices])
#     test_indices = numpy.where((obj.year == 2020) & (obj.native_status == "Nonnative") & (obj.inWoolsey == False))
#     SpeciesListNonnativeOutWoolsey2020 = numpy.append(SpeciesListNonnativeOutWoolsey2020, obj.species_code[test_indices])
# numpy.set_printoptions(threshold=numpy.inf)
#
# ## Figure 2 numbers ###
# # table_rarefaction = [['Years', 2017, 2018, 2019, 2020],
# #                      ['----------------------------'],
# #                      ['IN WOOLSEY'],
# #                      ['Number of observations',
# #                       totalobsinW_17,
# #                       totalobsinW_18,
# #                       totalobsinW_19,
# #                       totalobsinW_20],
# #                      ['Native total counted',
# #                       len(numpy.unique(SpeciesListNativeInWoolsey2017)),
# #                       len(numpy.unique(SpeciesListNativeInWoolsey2018)),
# #                       len(numpy.unique(SpeciesListNativeInWoolsey2019)),
# #                       len(numpy.unique(SpeciesListNativeInWoolsey2020))],
# #                      ['Native estimated', 119.5, 107.5, 184.2, 190.4],
# #                      ['Native lower limit', 118.1, 106.4, 183.5, 189.9],
# #                      ['Native upper limit', 120.9, 108.7, 184.9, 190.9],
# #                      ['Nonnative total counted',
# #                       len(numpy.unique(SpeciesListNonnativeInWoolsey2017)),
# #                       len(numpy.unique(SpeciesListNonnativeInWoolsey2018)),
# #                       len(numpy.unique(SpeciesListNonnativeInWoolsey2019)),
# #                       len(numpy.unique(SpeciesListNonnativeInWoolsey2020))],
# #                      ['Nonnative estimated', 24.7, 28.7, 51.3, 46.3],
# #                      ['Nonnative lower limit', 24.4, 28.4, 51.1, 46.2],
# #                      ['Nonnative upper limit', 25.1, 29, 51.5, 46.4],
# #                      ['----------------------------'],
# #                      ['OUTSIDE WOOLSEY'],
# #                      ['Number of observations',
# #                       totalobsoutW_17,
# #                       totalobsoutW_18,
# #                       totalobsoutW_19,
# #                       totalobsoutW_20],
# #                      ['Native total counted',
# #                       len(numpy.unique(SpeciesListNativeOutWoolsey2017)),
# #                       len(numpy.unique(SpeciesListNativeOutWoolsey2018)),
# #                       len(numpy.unique(SpeciesListNativeOutWoolsey2019)),
# #                       len(numpy.unique(SpeciesListNativeOutWoolsey2020))],
# #                      ['Native estimated', 142.5, 100.9, 137.1, 119.7],
# #                      ['Native lower limit', 140.7, 99.4, 135.7, 118.4],
# #                      ['Native upper limit', 144.3, 102.5, 138.6, 121],
# #                      ['Nonnative total counted',
# #                       len(numpy.unique(SpeciesListNonnativeOutWoolsey2017)),
# #                       len(numpy.unique(SpeciesListNonnativeOutWoolsey2018)),
# #                       len(numpy.unique(SpeciesListNonnativeOutWoolsey2019)),
# #                       len(numpy.unique(SpeciesListNonnativeOutWoolsey2020))],
# #                      ['Nonnative estimated', 35.8, 28.3, 36.3, 39.5],
# #                      ['Nonnative lower limit', 35.4, 27.9, 35.9, 38.9],
# #                      ['Nonnative upper limit', 36.3, 28.8, 36.8, 40],
# #                      ]
# # print(tabulate(table_rarefaction))
#
# # table_rarefaction_latex = [[2017, 2018, 2019, 2020],
# #                      [len(numpy.unique(SpeciesListNativeInWoolsey2017)),
# #                       len(numpy.unique(SpeciesListNativeInWoolsey2018)),
# #                       len(numpy.unique(SpeciesListNativeInWoolsey2019)),
# #                       len(numpy.unique(SpeciesListNativeInWoolsey2020))],
# #                      [119.5, 107.5, 184.2, 190.4],
# #                      [118.1, 106.4, 183.5, 189.9],
# #                      [120.9, 108.7, 184.9, 190.9],
# #                      [len(numpy.unique(SpeciesListNonnativeInWoolsey2017)),
# #                       len(numpy.unique(SpeciesListNonnativeInWoolsey2018)),
# #                       len(numpy.unique(SpeciesListNonnativeInWoolsey2019)),
# #                       len(numpy.unique(SpeciesListNonnativeInWoolsey2020))],
# #                      [24.7, 28.7, 51.3, 46.3],
# #                      [24.4, 28.4, 51.1, 46.2],
# #                      [25.1, 29, 51.5, 46.4],
# #                      [len(numpy.unique(SpeciesListNativeOutWoolsey2017)),
# #                       len(numpy.unique(SpeciesListNativeOutWoolsey2018)),
# #                       len(numpy.unique(SpeciesListNativeOutWoolsey2019)),
# #                       len(numpy.unique(SpeciesListNativeOutWoolsey2020))],
# #                      [142.5, 100.9, 137.1, 119.7],
# #                      [140.7, 99.4, 135.7, 118.4],
# #                      [144.3, 102.5, 138.6, 121],
# #                      [len(numpy.unique(SpeciesListNonnativeOutWoolsey2017)),
# #                       len(numpy.unique(SpeciesListNonnativeOutWoolsey2018)),
# #                       len(numpy.unique(SpeciesListNonnativeOutWoolsey2019)),
# #                       len(numpy.unique(SpeciesListNonnativeOutWoolsey2020))],
# #                      [35.8, 28.3, 36.3, 39.5],
# #                      [35.4, 27.9, 35.9, 38.9],
# #                      [36.3, 28.8, 36.8, 40],
# #                      ]
# # print('\\begin{tabular}{lllllll} \\toprule')
# # for row in table_rarefaction_latex:
# #     for item in row:
# #         y = ' & '.join([str(item) for item in row])
# #     print(y + ' \\''\\')
# # print('\\end{tabular}')
#
# ##### Curve fitting ############################
# # def asymptote(x, lam, k, amp):
# #     func = amp*(1-numpy.e**(-(x/lam)**k))
# #     return func
# ################################################
#
# def rarefaction(native17, native18, native19, native20, nonnative17, nonnative18, nonnative19, nonnative20, species17,  species18, species19, species20):
#     trials = 10
#
#     nat_species17 = numpy.empty((0, trials), int)
#     nat_samples17 = numpy.arange(0, len(species17), 100)
#     for trial in range(0, trials):
#         for n in nat_samples17:
#             nat_index17 = numpy.random.choice(native17, n)
#             nat_species17 = numpy.append([nat_species17], [len(numpy.unique(nat_index17))])
#     nat_species18 = numpy.empty((0, trials), int)
#     nat_samples18 = numpy.arange(0, len(species18), 100)
#     for trial in range(0, trials):
#         for n in nat_samples18:
#             nat_index18 = numpy.random.choice(native18, n)
#             nat_species18 = numpy.append([nat_species18], [len(numpy.unique(nat_index18))])
#     nat_species19 = numpy.empty((0, trials), int)
#     nat_samples19 = numpy.arange(0, len(species19), 100)
#     for trial in range(0, trials):
#         for n in nat_samples19:
#             nat_index19 = numpy.random.choice(native19, n)
#             nat_species19 = numpy.append([nat_species19], [len(numpy.unique(nat_index19))])
#     nat_species20 = numpy.empty((0, trials), int)
#     nat_samples20 = numpy.arange(0, len(species20), 100)
#     for trial in range(0, trials):
#         for n in nat_samples20:
#             nat_index20 = numpy.random.choice(native20, n)
#             nat_species20 = numpy.append([nat_species20], [len(numpy.unique(nat_index20))])
#
#
#     nnat_species17 = numpy.empty((0, trials), int)
#     nnat_samples17 = numpy.arange(0, len(species17), 100)
#     for trial in range(0, trials):
#         for n in nnat_samples17:
#             nnat_index17 = numpy.random.choice(nonnative17, n)
#             nnat_species17 = numpy.append([nnat_species17], [len(numpy.unique(nnat_index17))])
#     nnat_species18 = numpy.empty((0, trials), int)
#     nnat_samples18 = numpy.arange(0, len(species18), 100)
#     for trial in range(0, trials):
#         for n in nnat_samples18:
#             nnat_index18 = numpy.random.choice(nonnative18, n)
#             nnat_species18 = numpy.append([nnat_species18], [len(numpy.unique(nnat_index18))])
#     nnat_species19 = numpy.empty((0, trials), int)
#     nnat_samples19 = numpy.arange(0, len(species19), 100)
#     for trial in range(0, trials):
#         for n in nnat_samples19:
#             nnat_index19 = numpy.random.choice(nonnative19, n)
#             nnat_species19 = numpy.append([nnat_species19], [len(numpy.unique(nnat_index19))])
#     nnat_species20 = numpy.empty((0, trials), int)
#     nnat_samples20 = numpy.arange(0, len(species20), 100)
#     for trial in range(0, trials):
#         for n in nnat_samples20:
#             nnat_index20 = numpy.random.choice(nonnative20, n)
#             nnat_species20 = numpy.append([nnat_species20], [len(numpy.unique(nnat_index20))])
#
#     sample_obs = numpy.arange(0, 14000, 100)
#
#     nat_array17 = numpy.split(nat_species17, trials)
#     nat_array18 = numpy.split(nat_species18, trials)
#     nat_array19 = numpy.split(nat_species19, trials)
#     nat_array20 = numpy.split(nat_species20, trials)
#
#     nnat_array17 = numpy.split(nnat_species17, trials)
#     nnat_array18 = numpy.split(nnat_species18, trials)
#     nnat_array19 = numpy.split(nnat_species19, trials)
#     nnat_array20 = numpy.split(nnat_species20, trials)
#
#     # numpyarray = numpy.array(list(zip(*newarray)))
#     nat_mean17 = numpy.array([sum(x)/len(x) for x in zip(*nat_array17)])
#     nat_stdeviation17 = numpy.std(nat_array17, axis=0)
#     nat_mean18 = numpy.array([sum(x)/len(x) for x in zip(*nat_array18)])
#     nat_stdeviation18 = numpy.std(nat_array18, axis=0)
#     nat_mean19 = numpy.array([sum(x)/len(x) for x in zip(*nat_array19)])
#     nat_stdeviation19 = numpy.std(nat_array19, axis=0)
#     nat_mean20 = numpy.array([sum(x)/len(x) for x in zip(*nat_array20)])
#     nat_stdeviation20 = numpy.std(nat_array20, axis=0)
#
#     nnat_mean17 = numpy.array([sum(x)/len(x) for x in zip(*nnat_array17)])
#     nnat_stdeviation17 = numpy.std(nnat_array17, axis=0)
#     nnat_mean18 = numpy.array([sum(x)/len(x) for x in zip(*nnat_array18)])
#     nnat_stdeviation18 = numpy.std(nnat_array18, axis=0)
#     nnat_mean19 = numpy.array([sum(x)/len(x) for x in zip(*nnat_array19)])
#     nnat_stdeviation19 = numpy.std(nnat_array19, axis=0)
#     nnat_mean20 = numpy.array([sum(x)/len(x) for x in zip(*nnat_array20)])
#     nnat_stdeviation20 = numpy.std(nnat_array20, axis=0)
#
#     # print('OUTSIDE WOOLSEY:')
#     # print('nat_mean17')
#     # for x in nat_mean17:
#     #     print(x)
#     # print('nat_mean18')
#     # for x in nat_mean18:
#     #     print(x)
#     # print('nat_mean19')
#     # for x in nat_mean19:
#     #     print(x)
#     # print('nat_mean20')
#     # for x in nat_mean20:
#     #     print(x)
#     # print('nnat_mean17')
#     # for x in nnat_mean17:
#     #     print(x)
#     # print('nnat_mean18')
#     # for x in nnat_mean18:
#     #     print(x)
#     # print('nnat_mean19')
#     # for x in nnat_mean19:
#     #     print(x)
#     # print('nnat_mean20')
#     # for x in nnat_mean20:
#     #     print(x)
#
#     ##### Curve fitting ###############################################
#     # popt_nat17, pcov_nat17 = curve_fit(asymptote, nat_samples17, nat_mean17, p0=[400, 0.5, 98])
#     # print('popt_nat17:', popt_nat17)
#     # print('pcov_nat17:', pcov_nat17)
#     # popt_nat18, pcov_nat18 = curve_fit(asymptote, nat_samples18, nat_mean18, p0=[400, 0.5, 98])
#     # print('popt_nat18:', popt_nat18)
#     # print('pcov_nat18:', pcov_nat18)
#     # popt_nat19, pcov_nat19 = curve_fit(asymptote, nat_samples19, nat_mean19, p0=[750, 0.5, 136])
#     # print('popt_nat19:', popt_nat19)
#     # popt_nat20, pcov_nat20 = curve_fit(asymptote, nat_samples20, nat_mean20, p0=[400, 0.5, 116])
#     # print('popt_nat20:', popt_nat20)
#     # print(len(nat_samples20))
#     #
#     # popt_nnat17, pcov_nnat17 = curve_fit(asymptote, nnat_samples17, nnat_mean17, p0=[300, 0.5, 27])
#     # print('popt_nnat17:', popt_nnat17)
#     # popt_nnat18, pcov_nnat18 = curve_fit(asymptote, nnat_samples18, nnat_mean18, p0=[300, 0.5, 27])
#     # print('popt_nnat18:', popt_nnat18)
#     # popt_nnat19, pcov_nnat19 = curve_fit(asymptote, nnat_samples19, nnat_mean19, p0=[400, 0.5, 35])
#     # print('popt_nnat19:', popt_nnat19)
#     # popt_nnat20, pcov_nnat20 = curve_fit(asymptote, nnat_samples20, nnat_mean20, p0=[350, 0.5, 38])
#     # print('popt_nnat20:', popt_nnat20)
#     #
#     # sampleplot_nat17 = numpy.array([])
#     # sampleplot_nat18 = numpy.array([])
#     # sampleplot_nat19 = numpy.array([])
#     # sampleplot_nat20 = numpy.array([])
#     #
#     # sampleplot_nnat17 = numpy.array([])
#     # sampleplot_nnat18 = numpy.array([])
#     # sampleplot_nnat19 = numpy.array([])
#     # sampleplot_nnat20 = numpy.array([])
#     #
#     # for x in nat_samples17:
#     #     y_nat17 = asymptote(x, 400, 0.5, 98)
#     #     sampleplot_nat17 = numpy.append(sampleplot_nat17, y_nat17)
#     # for x in nat_samples18:
#     #     y_nat18 = asymptote(x, 400, 0.5, 98)
#     #     sampleplot_nat18 = numpy.append(sampleplot_nat18, y_nat18)
#     # for x in nat_samples19:
#     #     y_nat19 = asymptote(x, 750, 0.5, 136)
#     #     sampleplot_nat19 = numpy.append(sampleplot_nat19, y_nat19)
#     # for x in nat_samples20:
#     #     y_nat20 = asymptote(x, 750, 0.5, 116)
#     #     sampleplot_nat20 = numpy.append(sampleplot_nat20, y_nat20)
#     #
#     # for x in nnat_samples17:
#     #     y_nnat17 = asymptote(x, 300, 0.5, 27)
#     #     sampleplot_nnat17 = numpy.append(sampleplot_nnat17, y_nnat17)
#     # for x in nnat_samples18:
#     #     y_nnat18 = asymptote(x, 300, 0.5, 27)
#     #     sampleplot_nnat18 = numpy.append(sampleplot_nnat18, y_nnat18)
#     # for x in nnat_samples19:
#     #     y_nnat19 = asymptote(x, 400, 0.5, 35)
#     #     sampleplot_nnat19 = numpy.append(sampleplot_nnat19, y_nnat19)
#     # for x in nnat_samples20:
#     #     y_nnat20 = asymptote(x, 350, 0.5, 38)
#     #     sampleplot_nnat20 = numpy.append(sampleplot_nnat20, y_nnat20)
#     ##################################################################
#
#     fig, ax = plt.subplots(1, figsize=(6, 6))
#     ax.plot(nat_samples17, nat_mean17, color=lightestblue, zorder=2)
#     ax.errorbar(nat_samples17, nat_mean17, nat_stdeviation17, color=lightestblue, zorder=1, capsize=1, elinewidth=1, markeredgewidth=1, label='Native 2017')
#     # ax.plot(nat_samples17, sampleplot_nat17, color='red', ls='--')
#
#     ax.plot(nat_samples18, nat_mean18, color=lightblue, zorder=2)
#     ax.errorbar(nat_samples18, nat_mean18, nat_stdeviation18, color=lightblue, zorder=1, capsize=1, elinewidth=1, markeredgewidth=1, label='Native 2018 (pre-fire)')
#     # ax.plot(nat_samples18, sampleplot_nat18, color='red', ls='--')
#
#     ax.plot(nat_samples19, nat_mean19, color=blue, zorder=2)
#     ax.errorbar(nat_samples19, nat_mean19, nat_stdeviation19, color=blue, zorder=1, capsize=1, elinewidth=1, markeredgewidth=1, label='Native 2019')
#     # ax.plot(nat_samples19, sampleplot_nat19, color='red', ls='--')
#
#     ax.plot(nat_samples20, nat_mean20, color=darkblue, zorder=2)
#     ax.errorbar(nat_samples20, nat_mean20, nat_stdeviation20, color=darkblue, zorder=1, capsize=1, elinewidth=1, markeredgewidth=1, label='Native 2020')
#     ax.plot(nat_samples20, sampleplot_nat20, color='red', ls='--')
#
#     ax.plot(nnat_samples17, nnat_mean17, color=yellow, zorder=2)
#     ax.errorbar(nnat_samples17, nnat_mean17, nnat_stdeviation17, color=yellow, zorder=1, capsize=1, elinewidth=1, markeredgewidth=1, label='Non-native 2017')
#     # ax.plot(nnat_samples17, sampleplot_nnat17, color='red', ls='--')
#
#     ax.plot(nnat_samples18, nnat_mean18, color=gold, zorder=2)
#     ax.errorbar(nnat_samples18, nnat_mean18, nnat_stdeviation18, color=gold, zorder=1, capsize=1, elinewidth=1, markeredgewidth=1, label='Non-native 2018 (pre-fire)')
#     # ax.plot(nnat_samples18, sampleplot_nnat18, color='red', ls='--')
#
#     ax.plot(nnat_samples19, nnat_mean19, color=rust, zorder=2)
#     ax.errorbar(nnat_samples19, nnat_mean19, nnat_stdeviation19, color=rust, zorder=1, capsize=1, elinewidth=1, markeredgewidth=1, label='Non-native 2019')
#     # ax.plot(nnat_samples19, sampleplot_nnat19, color='red', ls='--')
#
#     ax.plot(nnat_samples20, nnat_mean20, color=darkred, zorder=2)
#     ax.errorbar(nnat_samples20, nnat_mean20, nnat_stdeviation20, color=darkred, zorder=1, capsize=1, elinewidth=1, markeredgewidth=1, label='Non-native 2020')
#     # ax.plot(nnat_samples20, sampleplot_nnat20, color='red', ls='--')
#
#     ax.tick_params(labelsize=ea_axisnum)
#     ### plt.text(10, OutW18=98; OutW19=136; OutW20=116; InW20=190; InW19=183; InW18=108
#     plt.ylabel('Number of species', fontsize=ea_axislabel)
#     plt.xlabel('Samples', fontsize=ea_axislabel)
#     plt.tight_layout()
#     plt.ylim(0, 210)
#     # plt.legend(prop={"size":ea_axisnum}) #, loc='center right')
#     # plt.text(10, 200, 'a)', fontsize=ea_panelletter) # a) Native and non-native species inside the fire boundary
#     # plt.savefig("/Users/nicolettastork/Documents/Geography/SMMNRA_Invasive_Species/NPSData/VegData_FromSMMNRA_Modified/Fig_Rarefaction_InW.png", dpi=600)
#     plt.text(10, 200, 'b)', fontsize=ea_panelletter) # b) Native and non-native species outside the fire boundary
#     plt.savefig("/Users/nicolettastork/Documents/Geography/SMMNRA_Invasive_Species/NPSData/VegData_FromSMMNRA_Modified/Fig_Rarefaction_OutW.png", dpi=600)
#
#
# # rarefaction(SpeciesListNativeInWoolsey2017, SpeciesListNativeInWoolsey2018, SpeciesListNativeInWoolsey2019, SpeciesListNativeInWoolsey2020,
# #             SpeciesListNonnativeInWoolsey2017, SpeciesListNonnativeInWoolsey2018, SpeciesListNonnativeInWoolsey2019, SpeciesListNonnativeInWoolsey2020,
# #             SpeciesListInWoolsey2017, SpeciesListInWoolsey2018, SpeciesListInWoolsey2019, SpeciesListInWoolsey2020)
#
# rarefaction(SpeciesListNativeOutWoolsey2017, SpeciesListNativeOutWoolsey2018, SpeciesListNativeOutWoolsey2019, SpeciesListNativeOutWoolsey2020,
#             SpeciesListNonnativeOutWoolsey2017, SpeciesListNonnativeOutWoolsey2018, SpeciesListNonnativeOutWoolsey2019, SpeciesListNonnativeOutWoolsey2020,
#             SpeciesListOutWoolsey2017, SpeciesListOutWoolsey2018, SpeciesListOutWoolsey2019, SpeciesListOutWoolsey2020)
#

########################################################################
##### FIGURE 4: Functional groups barchart
########################################################################
# sitecode_totals_2014 = numpy.array([])
# sitecode_totals_2015 = numpy.array([])
# sitecode_totals_2016 = numpy.array([])
# sitecode_totals_2017 = numpy.array([])
# sitecode_totals_2018 = numpy.array([])
# sitecode_totals_2019 = numpy.array([])
# sitecode_totals_2020 = numpy.array([])
# sitecode_inW_totals_2014 = numpy.array([])
# sitecode_inW_totals_2015 = numpy.array([])
# sitecode_inW_totals_2016 = numpy.array([])
# sitecode_inW_totals_2017 = numpy.array([])
# sitecode_inW_totals_2018 = numpy.array([])
# sitecode_inW_totals_2019 = numpy.array([])
# sitecode_inW_totals_2020 = numpy.array([])
# sitecode_outW_totals_2014 = numpy.array([])
# sitecode_outW_totals_2015 = numpy.array([])
# sitecode_outW_totals_2016 = numpy.array([])
# sitecode_outW_totals_2017 = numpy.array([])
# sitecode_outW_totals_2018 = numpy.array([])
# sitecode_outW_totals_2019 = numpy.array([])
# sitecode_outW_totals_2020 = numpy.array([])
#
# natshrub_2017_inW = numpy.array([])
# natshrub_2018_inW = numpy.array([])
# natshrub_2019_inW = numpy.array([])
# natshrub_2020_inW = numpy.array([])
# natshrub_2017_outW = numpy.array([])
# natshrub_2018_outW = numpy.array([])
# natshrub_2019_outW = numpy.array([])
# natshrub_2020_outW = numpy.array([])
#
# natsubshrub_2017_inW = numpy.array([])
# natsubshrub_2018_inW = numpy.array([])
# natsubshrub_2019_inW = numpy.array([])
# natsubshrub_2020_inW = numpy.array([])
# natsubshrub_2017_outW = numpy.array([])
# natsubshrub_2018_outW = numpy.array([])
# natsubshrub_2019_outW = numpy.array([])
# natsubshrub_2020_outW = numpy.array([])
#
# natperherb_2017_inW = numpy.array([])
# natperherb_2018_inW = numpy.array([])
# natperherb_2019_inW = numpy.array([])
# natperherb_2020_inW = numpy.array([])
# natperherb_2017_outW = numpy.array([])
# natperherb_2018_outW = numpy.array([])
# natperherb_2019_outW = numpy.array([])
# natperherb_2020_outW = numpy.array([])
#
# natannherb_2017_inW = numpy.array([])
# natannherb_2018_inW = numpy.array([])
# natannherb_2019_inW = numpy.array([])
# natannherb_2020_inW = numpy.array([])
# natannherb_2017_outW = numpy.array([])
# natannherb_2018_outW = numpy.array([])
# natannherb_2019_outW = numpy.array([])
# natannherb_2020_outW = numpy.array([])
#
# natpergrass_2017_inW = numpy.array([])
# natpergrass_2018_inW = numpy.array([])
# natpergrass_2019_inW = numpy.array([])
# natpergrass_2020_inW = numpy.array([])
# natpergrass_2017_outW = numpy.array([])
# natpergrass_2018_outW = numpy.array([])
# natpergrass_2019_outW = numpy.array([])
# natpergrass_2020_outW = numpy.array([])
#
# nnatannherb_2017_inW = numpy.array([])
# nnatannherb_2018_inW = numpy.array([])
# nnatannherb_2019_inW = numpy.array([])
# nnatannherb_2020_inW = numpy.array([])
# nnatannherb_2017_outW = numpy.array([])
# nnatannherb_2018_outW = numpy.array([])
# nnatannherb_2019_outW = numpy.array([])
# nnatannherb_2020_outW = numpy.array([])
#
# nnatanngrass_2017_inW = numpy.array([])
# nnatanngrass_2018_inW = numpy.array([])
# nnatanngrass_2019_inW = numpy.array([])
# nnatanngrass_2020_inW = numpy.array([])
# nnatanngrass_2017_outW = numpy.array([])
# nnatanngrass_2018_outW = numpy.array([])
# nnatanngrass_2019_outW = numpy.array([])
# nnatanngrass_2020_outW = numpy.array([])
#
# for name, obj in monitoringpoints.items():
#     test_indices = numpy.where(obj.year == 2014)
#     sitecode_totals_2014 = numpy.append(sitecode_totals_2014, obj.sitecode[test_indices])
#     test_indices = numpy.where(obj.year == 2015)
#     sitecode_totals_2015 = numpy.append(sitecode_totals_2015, obj.sitecode[test_indices])
#     test_indices = numpy.where(obj.year == 2016)
#     sitecode_totals_2016 = numpy.append(sitecode_totals_2016, obj.sitecode[test_indices])
#     test_indices = numpy.where(obj.year == 2017)
#     sitecode_totals_2017 = numpy.append(sitecode_totals_2017, obj.sitecode[test_indices])
#     test_indices = numpy.where(obj.year == 2018)
#     sitecode_totals_2018 = numpy.append(sitecode_totals_2018, obj.sitecode[test_indices])
#     test_indices = numpy.where(obj.year == 2019)
#     sitecode_totals_2019 = numpy.append(sitecode_totals_2019, obj.sitecode[test_indices])
#     test_indices = numpy.where(obj.year == 2020)
#     sitecode_totals_2020 = numpy.append(sitecode_totals_2020, obj.sitecode[test_indices])
#
#     test_indices = numpy.where((obj.inWoolsey == True) & (obj.year == 2014))
#     sitecode_inW_totals_2014 = numpy.append(sitecode_inW_totals_2014, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.inWoolsey == True) & (obj.year == 2015))
#     sitecode_inW_totals_2015 = numpy.append(sitecode_inW_totals_2015, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.inWoolsey == True) & (obj.year == 2016))
#     sitecode_inW_totals_2016 = numpy.append(sitecode_inW_totals_2016, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.inWoolsey == True) & (obj.year == 2017))
#     sitecode_inW_totals_2017 = numpy.append(sitecode_inW_totals_2017, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.inWoolsey == True) & (obj.year == 2018))
#     sitecode_inW_totals_2018 = numpy.append(sitecode_inW_totals_2018, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.inWoolsey == True) & (obj.year == 2019))
#     sitecode_inW_totals_2019 = numpy.append(sitecode_inW_totals_2019, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.inWoolsey == True) & (obj.year == 2020))
#     sitecode_inW_totals_2020 = numpy.append(sitecode_inW_totals_2020, obj.sitecode[test_indices])
#
#     test_indices = numpy.where((obj.inWoolsey == False) & (obj.year == 2014))
#     sitecode_outW_totals_2014 = numpy.append(sitecode_outW_totals_2014, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.inWoolsey == False) & (obj.year == 2015))
#     sitecode_outW_totals_2015 = numpy.append(sitecode_outW_totals_2015, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.inWoolsey == False) & (obj.year == 2016))
#     sitecode_outW_totals_2016 = numpy.append(sitecode_outW_totals_2016, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.inWoolsey == False) & (obj.year == 2017))
#     sitecode_outW_totals_2017 = numpy.append(sitecode_outW_totals_2017, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.inWoolsey == False) & (obj.year == 2018))
#     sitecode_outW_totals_2018 = numpy.append(sitecode_outW_totals_2018, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.inWoolsey == False) & (obj.year == 2019))
#     sitecode_outW_totals_2019 = numpy.append(sitecode_outW_totals_2019, obj.sitecode[test_indices])
#     test_indices = numpy.where((obj.inWoolsey == False) & (obj.year == 2020))
#     sitecode_outW_totals_2020 = numpy.append(sitecode_outW_totals_2020, obj.sitecode[test_indices])
#
#     test = numpy.where(
#         (obj.year == 2017) & (obj.native_status == 'Native') & (obj.fxngroup == 'Shrub') & (obj.inWoolsey == True))
#     natshrub_2017_inW = numpy.append(natshrub_2017_inW, obj.sitecode[test])
#     test = numpy.where(
#         (obj.year == 2018) & (obj.native_status == 'Native') & (obj.fxngroup == 'Shrub') & (obj.inWoolsey == True))
#     natshrub_2018_inW = numpy.append(natshrub_2018_inW, obj.sitecode[test])
#     test = numpy.where(
#         (obj.year == 2019) & (obj.native_status == 'Native') & (obj.fxngroup == 'Shrub') & (obj.inWoolsey == True))
#     natshrub_2019_inW = numpy.append(natshrub_2019_inW, obj.sitecode[test])
#     test = numpy.where(
#         (obj.year == 2020) & (obj.native_status == 'Native') & (obj.fxngroup == 'Shrub') & (obj.inWoolsey == True))
#     natshrub_2020_inW = numpy.append(natshrub_2020_inW, obj.sitecode[test])
#     test = numpy.where(
#         (obj.year == 2017) & (obj.native_status == 'Native') & (obj.fxngroup == 'Shrub') & (obj.inWoolsey == False))
#     natshrub_2017_outW = numpy.append(natshrub_2017_outW, obj.sitecode[test])
#     test = numpy.where(
#         (obj.year == 2018) & (obj.native_status == 'Native') & (obj.fxngroup == 'Shrub') & (obj.inWoolsey == False))
#     natshrub_2018_outW = numpy.append(natshrub_2018_outW, obj.sitecode[test])
#     test = numpy.where(
#         (obj.year == 2019) & (obj.native_status == 'Native') & (obj.fxngroup == 'Shrub') & (obj.inWoolsey == False))
#     natshrub_2019_outW = numpy.append(natshrub_2019_outW, obj.sitecode[test])
#     test = numpy.where(
#         (obj.year == 2020) & (obj.native_status == 'Native') & (obj.fxngroup == 'Shrub') & (obj.inWoolsey == False))
#     natshrub_2020_outW = numpy.append(natshrub_2020_outW, obj.sitecode[test])
#
#     test = numpy.where(
#         (obj.year == 2017) & (obj.native_status == 'Native') & (obj.fxngroup == 'Sub-shrub') & (obj.inWoolsey == True))
#     natsubshrub_2017_inW = numpy.append(natsubshrub_2017_inW, obj.sitecode[test])
#     test = numpy.where(
#         (obj.year == 2018) & (obj.native_status == 'Native') & (obj.fxngroup == 'Sub-shrub') & (obj.inWoolsey == True))
#     natsubshrub_2018_inW = numpy.append(natsubshrub_2018_inW, obj.sitecode[test])
#     test = numpy.where(
#         (obj.year == 2019) & (obj.native_status == 'Native') & (obj.fxngroup == 'Sub-shrub') & (obj.inWoolsey == True))
#     natsubshrub_2019_inW = numpy.append(natsubshrub_2019_inW, obj.sitecode[test])
#     test = numpy.where(
#         (obj.year == 2020) & (obj.native_status == 'Native') & (obj.fxngroup == 'Sub-shrub') & (obj.inWoolsey == True))
#     natsubshrub_2020_inW = numpy.append(natsubshrub_2020_inW, obj.sitecode[test])
#     test = numpy.where(
#         (obj.year == 2017) & (obj.native_status == 'Native') & (obj.fxngroup == 'Sub-shrub') & (obj.inWoolsey == False))
#     natsubshrub_2017_outW = numpy.append(natsubshrub_2017_outW, obj.sitecode[test])
#     test = numpy.where(
#         (obj.year == 2018) & (obj.native_status == 'Native') & (obj.fxngroup == 'Sub-shrub') & (obj.inWoolsey == False))
#     natsubshrub_2018_outW = numpy.append(natsubshrub_2018_outW, obj.sitecode[test])
#     test = numpy.where(
#         (obj.year == 2019) & (obj.native_status == 'Native') & (obj.fxngroup == 'Sub-shrub') & (obj.inWoolsey == False))
#     natsubshrub_2019_outW = numpy.append(natsubshrub_2019_outW, obj.sitecode[test])
#     test = numpy.where(
#         (obj.year == 2020) & (obj.native_status == 'Native') & (obj.fxngroup == 'Sub-shrub') & (obj.inWoolsey == False))
#     natsubshrub_2020_outW = numpy.append(natsubshrub_2020_outW, obj.sitecode[test])
#
#     test = numpy.where((obj.year == 2017) & (obj.native_status == 'Native') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Perennial') & (obj.inWoolsey == True))
#     natperherb_2017_inW = numpy.append(natperherb_2017_inW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2018) & (obj.native_status == 'Native') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Perennial') & (obj.inWoolsey == True))
#     natperherb_2018_inW = numpy.append(natperherb_2018_inW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2019) & (obj.native_status == 'Native') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Perennial') & (obj.inWoolsey == True))
#     natperherb_2019_inW = numpy.append(natperherb_2019_inW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2020) & (obj.native_status == 'Native') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Perennial') & (obj.inWoolsey == True))
#     natperherb_2020_inW = numpy.append(natperherb_2020_inW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2017) & (obj.native_status == 'Native') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Perennial') & (obj.inWoolsey == False))
#     natperherb_2017_outW = numpy.append(natperherb_2017_outW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2018) & (obj.native_status == 'Native') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Perennial') & (obj.inWoolsey == False))
#     natperherb_2018_outW = numpy.append(natperherb_2018_outW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2019) & (obj.native_status == 'Native') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Perennial') & (obj.inWoolsey == False))
#     natperherb_2019_outW = numpy.append(natperherb_2019_outW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2020) & (obj.native_status == 'Native') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Perennial') & (obj.inWoolsey == False))
#     natperherb_2020_outW = numpy.append(natperherb_2020_outW, obj.sitecode[test])
#
#     test = numpy.where((obj.year == 2017) & (obj.native_status == 'Native') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == True))
#     natannherb_2017_inW = numpy.append(natannherb_2017_inW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2018) & (obj.native_status == 'Native') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == True))
#     natannherb_2018_inW = numpy.append(natannherb_2018_inW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2019) & (obj.native_status == 'Native') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == True))
#     natannherb_2019_inW = numpy.append(natannherb_2019_inW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2020) & (obj.native_status == 'Native') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == True))
#     natannherb_2020_inW = numpy.append(natannherb_2020_inW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2017) & (obj.native_status == 'Native') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == False))
#     natannherb_2017_outW = numpy.append(natannherb_2017_outW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2018) & (obj.native_status == 'Native') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == False))
#     natannherb_2018_outW = numpy.append(natannherb_2018_outW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2019) & (obj.native_status == 'Native') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == False))
#     natannherb_2019_outW = numpy.append(natannherb_2019_outW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2020) & (obj.native_status == 'Native') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == False))
#     natannherb_2020_outW = numpy.append(natannherb_2020_outW, obj.sitecode[test])
#
#     test = numpy.where((obj.year == 2017) & (obj.native_status == 'Native') & (obj.fxngroup == 'Grass') & (
#                 obj.ann_per == 'Perennial') & (obj.inWoolsey == True))
#     natpergrass_2017_inW = numpy.append(natpergrass_2017_inW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2018) & (obj.native_status == 'Native') & (obj.fxngroup == 'Grass') & (
#                 obj.ann_per == 'Perennial') & (obj.inWoolsey == True))
#     natpergrass_2018_inW = numpy.append(natpergrass_2018_inW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2019) & (obj.native_status == 'Native') & (obj.fxngroup == 'Grass') & (
#                 obj.ann_per == 'Perennial') & (obj.inWoolsey == True))
#     natpergrass_2019_inW = numpy.append(natpergrass_2019_inW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2020) & (obj.native_status == 'Native') & (obj.fxngroup == 'Grass') & (
#                 obj.ann_per == 'Perennial') & (obj.inWoolsey == True))
#     natpergrass_2020_inW = numpy.append(natpergrass_2020_inW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2017) & (obj.native_status == 'Native') & (obj.fxngroup == 'Grass') & (
#                 obj.ann_per == 'Perennial') & (obj.inWoolsey == False))
#     natpergrass_2017_outW = numpy.append(natpergrass_2017_outW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2018) & (obj.native_status == 'Native') & (obj.fxngroup == 'Grass') & (
#                 obj.ann_per == 'Perennial') & (obj.inWoolsey == False))
#     natpergrass_2018_outW = numpy.append(natpergrass_2018_outW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2019) & (obj.native_status == 'Native') & (obj.fxngroup == 'Grass') & (
#                 obj.ann_per == 'Perennial') & (obj.inWoolsey == False))
#     natpergrass_2019_outW = numpy.append(natpergrass_2019_outW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2020) & (obj.native_status == 'Native') & (obj.fxngroup == 'Grass') & (
#                 obj.ann_per == 'Perennial') & (obj.inWoolsey == False))
#     natpergrass_2020_outW = numpy.append(natpergrass_2020_outW, obj.sitecode[test])
#
#     test = numpy.where((obj.year == 2017) & (obj.native_status == 'Nonnative') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == True))
#     nnatannherb_2017_inW = numpy.append(nnatannherb_2017_inW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2018) & (obj.native_status == 'Nonnative') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == True))
#     nnatannherb_2018_inW = numpy.append(nnatannherb_2018_inW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2019) & (obj.native_status == 'Nonnative') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == True))
#     nnatannherb_2019_inW = numpy.append(nnatannherb_2019_inW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2020) & (obj.native_status == 'Nonnative') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == True))
#     nnatannherb_2020_inW = numpy.append(nnatannherb_2020_inW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2017) & (obj.native_status == 'Nonnative') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == False))
#     nnatannherb_2017_outW = numpy.append(nnatannherb_2017_outW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2018) & (obj.native_status == 'Nonnative') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == False))
#     nnatannherb_2018_outW = numpy.append(nnatannherb_2018_outW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2019) & (obj.native_status == 'Nonnative') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == False))
#     nnatannherb_2019_outW = numpy.append(nnatannherb_2019_outW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2020) & (obj.native_status == 'Nonnative') & (obj.fxngroup == 'Herbaceous') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == False))
#     nnatannherb_2020_outW = numpy.append(nnatannherb_2020_outW, obj.sitecode[test])
#
#     test = numpy.where((obj.year == 2017) & (obj.native_status == 'Nonnative') & (obj.fxngroup == 'Grass') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == True))
#     nnatanngrass_2017_inW = numpy.append(nnatanngrass_2017_inW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2018) & (obj.native_status == 'Nonnative') & (obj.fxngroup == 'Grass') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == True))
#     nnatanngrass_2018_inW = numpy.append(nnatanngrass_2018_inW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2019) & (obj.native_status == 'Nonnative') & (obj.fxngroup == 'Grass') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == True))
#     nnatanngrass_2019_inW = numpy.append(nnatanngrass_2019_inW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2020) & (obj.native_status == 'Nonnative') & (obj.fxngroup == 'Grass') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == True))
#     nnatanngrass_2020_inW = numpy.append(nnatanngrass_2020_inW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2017) & (obj.native_status == 'Nonnative') & (obj.fxngroup == 'Grass') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == False))
#     nnatanngrass_2017_outW = numpy.append(nnatanngrass_2017_outW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2018) & (obj.native_status == 'Nonnative') & (obj.fxngroup == 'Grass') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == False))
#     nnatanngrass_2018_outW = numpy.append(nnatanngrass_2018_outW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2019) & (obj.native_status == 'Nonnative') & (obj.fxngroup == 'Grass') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == False))
#     nnatanngrass_2019_outW = numpy.append(nnatanngrass_2019_outW, obj.sitecode[test])
#     test = numpy.where((obj.year == 2020) & (obj.native_status == 'Nonnative') & (obj.fxngroup == 'Grass') & (
#                 obj.ann_per == 'Annual') & (obj.inWoolsey == False))
#     nnatanngrass_2020_outW = numpy.append(nnatanngrass_2020_outW, obj.sitecode[test])
#
#
# def fxndict(tot17_inW, tot18_inW, tot19_inW, tot20_inW,
#             natshrub17inW, natshrub18inW, natshrub19inW, natshrub20inW,
#             natsubshrub17inW, natsubshrub18inW, natsubshrub19inW, natsubshrub20inW,
#             natperherb17inW, natperherb18inW, natperherb19inW, natperherb20inW,
#             natannherb17inW, natannherb18inW, natannherb19inW, natannherb20inW,
#             natpergrass17inW, natpergrass18inW, natpergrass19inW, natpergrass20inW,
#             nnatannherb17inW, nnatannherb18inW, nnatannherb19inW, nnatannherb20inW,
#             nnatanngrass17inW, nnatanngrass18inW, nnatanngrass19inW, nnatanngrass20inW,
#             tot17_outW, tot18_outW, tot19_outW, tot20_outW,
#             natshrub17outW, natshrub18outW, natshrub19outW, natshrub20outW,
#             natsubshrub17outW, natsubshrub18outW, natsubshrub19outW, natsubshrub20outW,
#             natperherb17outW, natperherb18outW, natperherb19outW, natperherb20outW,
#             natannherb17outW, natannherb18outW, natannherb19outW, natannherb20outW,
#             natpergrass17outW, natpergrass18outW, natpergrass19outW, natpergrass20outW,
#             nnatannherb17outW, nnatannherb18outW, nnatannherb19outW, nnatannherb20outW,
#             nnatanngrass17outW, nnatanngrass18outW, nnatanngrass19outW, nnatanngrass20outW
#             ):
#     labels_tot17_inW, counts_tot17_inW = numpy.unique(tot17_inW, return_counts=True)
#     totals17inW_dict = {labels_tot17_inW[i]: counts_tot17_inW[i] for i in
#                         range(len(labels_tot17_inW))}
#     labels_tot18_inW, counts_tot18_inW = numpy.unique(tot18_inW, return_counts=True)
#     totals18inW_dict = {labels_tot18_inW[i]: counts_tot18_inW[i] for i in
#                         range(len(labels_tot18_inW))}
#     labels_tot19_inW, counts_tot19_inW = numpy.unique(tot19_inW, return_counts=True)
#     totals19inW_dict = {labels_tot19_inW[i]: counts_tot19_inW[i] for i in
#                         range(len(labels_tot19_inW))}
#     labels_tot20_inW, counts_tot20_inW = numpy.unique(tot20_inW, return_counts=True)
#     totals20inW_dict = {labels_tot20_inW[i]: counts_tot20_inW[i] for i in
#                         range(len(labels_tot20_inW))}
#     labels_tot17_outW, counts_tot17_outW = numpy.unique(tot17_outW, return_counts=True)
#     totals17outW_dict = {labels_tot17_outW[i]: counts_tot17_outW[i] for i in
#                          range(len(labels_tot17_outW))}
#     labels_tot18_outW, counts_tot18_outW = numpy.unique(tot18_outW, return_counts=True)
#     totals18outW_dict = {labels_tot18_outW[i]: counts_tot18_outW[i] for i in
#                          range(len(labels_tot18_outW))}
#     labels_tot19_outW, counts_tot19_outW = numpy.unique(tot19_outW, return_counts=True)
#     totals19outW_dict = {labels_tot19_outW[i]: counts_tot19_outW[i] for i in
#                          range(len(labels_tot19_outW))}
#     labels_tot20_outW, counts_tot20_outW = numpy.unique(tot20_outW, return_counts=True)
#     totals20outW_dict = {labels_tot20_outW[i]: counts_tot20_outW[i] for i in
#                          range(len(labels_tot20_outW))}
#
#     labels_natshrub17_inW, counts_natshrub17_inW = numpy.unique(natshrub17inW, return_counts=True)
#     natshrub17inW_dict = {labels_natshrub17_inW[i]: counts_natshrub17_inW[i] for i in
#                           range(len(labels_natshrub17_inW))}
#     labels_natshrub18_inW, counts_natshrub18_inW = numpy.unique(natshrub18inW, return_counts=True)
#     natshrub18inW_dict = {labels_natshrub18_inW[i]: counts_natshrub18_inW[i] for i in
#                           range(len(labels_natshrub18_inW))}
#     labels_natshrub19_inW, counts_natshrub19_inW = numpy.unique(natshrub19inW, return_counts=True)
#     natshrub19inW_dict = {labels_natshrub19_inW[i]: counts_natshrub19_inW[i] for i in
#                           range(len(labels_natshrub19_inW))}
#     labels_natshrub20_inW, counts_natshrub20_inW = numpy.unique(natshrub20inW, return_counts=True)
#     natshrub20inW_dict = {labels_natshrub20_inW[i]: counts_natshrub20_inW[i] for i in
#                           range(len(labels_natshrub20_inW))}
#     labels_natshrub17_outW, counts_natshrub17_outW = numpy.unique(natshrub17outW, return_counts=True)
#     natshrub17outW_dict = {labels_natshrub17_outW[i]: counts_natshrub17_outW[i] for i in
#                            range(len(labels_natshrub17_outW))}
#     labels_natshrub18_outW, counts_natshrub18_outW = numpy.unique(natshrub18outW, return_counts=True)
#     natshrub18outW_dict = {labels_natshrub18_outW[i]: counts_natshrub18_outW[i] for i in
#                            range(len(labels_natshrub18_outW))}
#     labels_natshrub19_outW, counts_natshrub19_outW = numpy.unique(natshrub19outW, return_counts=True)
#     natshrub19outW_dict = {labels_natshrub19_outW[i]: counts_natshrub19_outW[i] for i in
#                            range(len(labels_natshrub19_outW))}
#     labels_natshrub20_outW, counts_natshrub20_outW = numpy.unique(natshrub20outW, return_counts=True)
#     natshrub20outW_dict = {labels_natshrub20_outW[i]: counts_natshrub20_outW[i] for i in
#                            range(len(labels_natshrub20_outW))}
#
#     labels_natsubshrub17_inW, counts_natsubshrub17_inW = numpy.unique(natsubshrub17inW, return_counts=True)
#     natsubshrub17inW_dict = {labels_natsubshrub17_inW[i]: counts_natsubshrub17_inW[i] for i in
#                              range(len(labels_natsubshrub17_inW))}
#     labels_natsubshrub18_inW, counts_natsubshrub18_inW = numpy.unique(natsubshrub18inW, return_counts=True)
#     natsubshrub18inW_dict = {labels_natsubshrub18_inW[i]: counts_natsubshrub18_inW[i] for i in
#                              range(len(labels_natsubshrub18_inW))}
#     labels_natsubshrub19_inW, counts_natsubshrub19_inW = numpy.unique(natsubshrub19inW, return_counts=True)
#     natsubshrub19inW_dict = {labels_natsubshrub19_inW[i]: counts_natsubshrub19_inW[i] for i in
#                              range(len(labels_natsubshrub19_inW))}
#     labels_natsubshrub20_inW, counts_natsubshrub20_inW = numpy.unique(natsubshrub20inW, return_counts=True)
#     natsubshrub20inW_dict = {labels_natsubshrub20_inW[i]: counts_natsubshrub20_inW[i] for i in
#                              range(len(labels_natsubshrub20_inW))}
#     labels_natsubshrub17_outW, counts_natsubshrub17_outW = numpy.unique(natsubshrub17outW, return_counts=True)
#     natsubshrub17outW_dict = {labels_natsubshrub17_outW[i]: counts_natsubshrub17_outW[i] for i in
#                               range(len(labels_natsubshrub17_outW))}
#     labels_natsubshrub18_outW, counts_natsubshrub18_outW = numpy.unique(natsubshrub18outW, return_counts=True)
#     natsubshrub18outW_dict = {labels_natsubshrub18_outW[i]: counts_natsubshrub18_outW[i] for i in
#                               range(len(labels_natsubshrub18_outW))}
#     labels_natsubshrub19_outW, counts_natsubshrub19_outW = numpy.unique(natsubshrub19outW, return_counts=True)
#     natsubshrub19outW_dict = {labels_natsubshrub19_outW[i]: counts_natsubshrub19_outW[i] for i in
#                               range(len(labels_natsubshrub19_outW))}
#     labels_natsubshrub20_outW, counts_natsubshrub20_outW = numpy.unique(natsubshrub20outW, return_counts=True)
#     natsubshrub20outW_dict = {labels_natsubshrub20_outW[i]: counts_natsubshrub20_outW[i] for i in
#                               range(len(labels_natsubshrub20_outW))}
#
#     labels_natperherb17_inW, counts_natperherb17_inW = numpy.unique(natperherb17inW, return_counts=True)
#     natperherb17inW_dict = {labels_natperherb17_inW[i]: counts_natperherb17_inW[i] for i in
#                             range(len(labels_natperherb17_inW))}
#     labels_natperherb18_inW, counts_natperherb18_inW = numpy.unique(natperherb18inW, return_counts=True)
#     natperherb18inW_dict = {labels_natperherb18_inW[i]: counts_natperherb18_inW[i] for i in
#                             range(len(labels_natperherb18_inW))}
#     labels_natperherb19_inW, counts_natperherb19_inW = numpy.unique(natperherb19inW, return_counts=True)
#     natperherb19inW_dict = {labels_natperherb19_inW[i]: counts_natperherb19_inW[i] for i in
#                             range(len(labels_natperherb19_inW))}
#     labels_natperherb20_inW, counts_natperherb20_inW = numpy.unique(natperherb20inW, return_counts=True)
#     natperherb20inW_dict = {labels_natperherb20_inW[i]: counts_natperherb20_inW[i] for i in
#                             range(len(labels_natperherb20_inW))}
#     labels_natperherb17_outW, counts_natperherb17_outW = numpy.unique(natperherb17outW, return_counts=True)
#     natperherb17outW_dict = {labels_natperherb17_outW[i]: counts_natperherb17_outW[i] for i in
#                              range(len(labels_natperherb17_outW))}
#     labels_natperherb18_outW, counts_natperherb18_outW = numpy.unique(natperherb18outW, return_counts=True)
#     natperherb18outW_dict = {labels_natperherb18_outW[i]: counts_natperherb18_outW[i] for i in
#                              range(len(labels_natperherb18_outW))}
#     labels_natperherb19_outW, counts_natperherb19_outW = numpy.unique(natperherb19outW, return_counts=True)
#     natperherb19outW_dict = {labels_natperherb19_outW[i]: counts_natperherb19_outW[i] for i in
#                              range(len(labels_natperherb19_outW))}
#     labels_natperherb20_outW, counts_natperherb20_outW = numpy.unique(natperherb20outW, return_counts=True)
#     natperherb20outW_dict = {labels_natperherb20_outW[i]: counts_natperherb20_outW[i] for i in
#                              range(len(labels_natperherb20_outW))}
#
#     labels_natannherb17_inW, counts_natannherb17_inW = numpy.unique(natannherb17inW, return_counts=True)
#     natannherb17inW_dict = {labels_natannherb17_inW[i]: counts_natannherb17_inW[i] for i in
#                             range(len(labels_natannherb17_inW))}
#     labels_natannherb18_inW, counts_natannherb18_inW = numpy.unique(natannherb18inW, return_counts=True)
#     natannherb18inW_dict = {labels_natannherb18_inW[i]: counts_natannherb18_inW[i] for i in
#                             range(len(labels_natannherb18_inW))}
#     labels_natannherb19_inW, counts_natannherb19_inW = numpy.unique(natannherb19inW, return_counts=True)
#     natannherb19inW_dict = {labels_natannherb19_inW[i]: counts_natannherb19_inW[i] for i in
#                             range(len(labels_natannherb19_inW))}
#     labels_natannherb20_inW, counts_natannherb20_inW = numpy.unique(natannherb20inW, return_counts=True)
#     natannherb20inW_dict = {labels_natannherb20_inW[i]: counts_natannherb20_inW[i] for i in
#                             range(len(labels_natannherb20_inW))}
#     labels_natannherb17_outW, counts_natannherb17_outW = numpy.unique(natannherb17outW, return_counts=True)
#     natannherb17outW_dict = {labels_natannherb17_outW[i]: counts_natannherb17_outW[i] for i in
#                              range(len(labels_natannherb17_outW))}
#     labels_natannherb18_outW, counts_natannherb18_outW = numpy.unique(natannherb18outW, return_counts=True)
#     natannherb18outW_dict = {labels_natannherb18_outW[i]: counts_natannherb18_outW[i] for i in
#                              range(len(labels_natannherb18_outW))}
#     labels_natannherb19_outW, counts_natannherb19_outW = numpy.unique(natannherb19outW, return_counts=True)
#     natannherb19outW_dict = {labels_natannherb19_outW[i]: counts_natannherb19_outW[i] for i in
#                              range(len(labels_natannherb19_outW))}
#     labels_natannherb20_outW, counts_natannherb20_outW = numpy.unique(natannherb20outW, return_counts=True)
#     natannherb20outW_dict = {labels_natannherb20_outW[i]: counts_natannherb20_outW[i] for i in
#                              range(len(labels_natannherb20_outW))}
#
#     labels_natpergrass17_inW, counts_natpergrass17_inW = numpy.unique(natpergrass17inW, return_counts=True)
#     natpergrass17inW_dict = {labels_natpergrass17_inW[i]: counts_natpergrass17_inW[i] for i in
#                              range(len(labels_natpergrass17_inW))}
#     labels_natpergrass18_inW, counts_natpergrass18_inW = numpy.unique(natpergrass18inW, return_counts=True)
#     natpergrass18inW_dict = {labels_natpergrass18_inW[i]: counts_natpergrass18_inW[i] for i in
#                              range(len(labels_natpergrass18_inW))}
#     labels_natpergrass19_inW, counts_natpergrass19_inW = numpy.unique(natpergrass19inW, return_counts=True)
#     natpergrass19inW_dict = {labels_natpergrass19_inW[i]: counts_natpergrass19_inW[i] for i in
#                              range(len(labels_natpergrass19_inW))}
#     labels_natpergrass20_inW, counts_natpergrass20_inW = numpy.unique(natpergrass20inW, return_counts=True)
#     natpergrass20inW_dict = {labels_natpergrass20_inW[i]: counts_natpergrass20_inW[i] for i in
#                              range(len(labels_natpergrass20_inW))}
#     labels_natpergrass17_outW, counts_natpergrass17_outW = numpy.unique(natpergrass17outW, return_counts=True)
#     natpergrass17outW_dict = {labels_natpergrass17_outW[i]: counts_natpergrass17_outW[i] for i in
#                               range(len(labels_natpergrass17_outW))}
#     labels_natpergrass18_outW, counts_natpergrass18_outW = numpy.unique(natpergrass18outW, return_counts=True)
#     natpergrass18outW_dict = {labels_natpergrass18_outW[i]: counts_natpergrass18_outW[i] for i in
#                               range(len(labels_natpergrass18_outW))}
#     labels_natpergrass19_outW, counts_natpergrass19_outW = numpy.unique(natpergrass19outW, return_counts=True)
#     natpergrass19outW_dict = {labels_natpergrass19_outW[i]: counts_natpergrass19_outW[i] for i in
#                               range(len(labels_natpergrass19_outW))}
#     labels_natpergrass20_outW, counts_natpergrass20_outW = numpy.unique(natpergrass20outW, return_counts=True)
#     natpergrass20outW_dict = {labels_natpergrass20_outW[i]: counts_natpergrass20_outW[i] for i in
#                               range(len(labels_natpergrass20_outW))}
#
#     labels_nnatannherb17_inW, counts_nnatannherb17_inW = numpy.unique(nnatannherb17inW, return_counts=True)
#     nnatannherb17inW_dict = {labels_nnatannherb17_inW[i]: counts_nnatannherb17_inW[i] for i in
#                              range(len(labels_nnatannherb17_inW))}
#     labels_nnatannherb18_inW, counts_nnatannherb18_inW = numpy.unique(nnatannherb18inW, return_counts=True)
#     nnatannherb18inW_dict = {labels_nnatannherb18_inW[i]: counts_nnatannherb18_inW[i] for i in
#                              range(len(labels_nnatannherb18_inW))}
#     labels_nnatannherb19_inW, counts_nnatannherb19_inW = numpy.unique(nnatannherb19inW, return_counts=True)
#     nnatannherb19inW_dict = {labels_nnatannherb19_inW[i]: counts_nnatannherb19_inW[i] for i in
#                              range(len(labels_nnatannherb19_inW))}
#     labels_nnatannherb20_inW, counts_nnatannherb20_inW = numpy.unique(nnatannherb20inW, return_counts=True)
#     nnatannherb20inW_dict = {labels_nnatannherb20_inW[i]: counts_nnatannherb20_inW[i] for i in
#                              range(len(labels_nnatannherb20_inW))}
#     labels_nnatannherb17_outW, counts_nnatannherb17_outW = numpy.unique(nnatannherb17outW, return_counts=True)
#     nnatannherb17outW_dict = {labels_nnatannherb17_outW[i]: counts_nnatannherb17_outW[i] for i in
#                               range(len(labels_nnatannherb17_outW))}
#     labels_nnatannherb18_outW, counts_nnatannherb18_outW = numpy.unique(nnatannherb18outW, return_counts=True)
#     nnatannherb18outW_dict = {labels_nnatannherb18_outW[i]: counts_nnatannherb18_outW[i] for i in
#                               range(len(labels_nnatannherb18_outW))}
#     labels_nnatannherb19_outW, counts_nnatannherb19_outW = numpy.unique(nnatannherb19outW, return_counts=True)
#     nnatannherb19outW_dict = {labels_nnatannherb19_outW[i]: counts_nnatannherb19_outW[i] for i in
#                               range(len(labels_nnatannherb19_outW))}
#     labels_nnatannherb20_outW, counts_nnatannherb20_outW = numpy.unique(nnatannherb20outW, return_counts=True)
#     nnatannherb20outW_dict = {labels_nnatannherb20_outW[i]: counts_nnatannherb20_outW[i] for i in
#                               range(len(labels_nnatannherb20_outW))}
#
#     labels_nnatanngrass17_inW, counts_nnatanngrass17_inW = numpy.unique(nnatanngrass17inW, return_counts=True)
#     nnatanngrass17inW_dict = {labels_nnatanngrass17_inW[i]: counts_nnatanngrass17_inW[i] for i in
#                               range(len(labels_nnatanngrass17_inW))}
#     labels_nnatanngrass18_inW, counts_nnatanngrass18_inW = numpy.unique(nnatanngrass18inW, return_counts=True)
#     nnatanngrass18inW_dict = {labels_nnatanngrass18_inW[i]: counts_nnatanngrass18_inW[i] for i in
#                               range(len(labels_nnatanngrass18_inW))}
#     labels_nnatanngrass19_inW, counts_nnatanngrass19_inW = numpy.unique(nnatanngrass19inW, return_counts=True)
#     nnatanngrass19inW_dict = {labels_nnatanngrass19_inW[i]: counts_nnatanngrass19_inW[i] for i in
#                               range(len(labels_nnatanngrass19_inW))}
#     labels_nnatanngrass20_inW, counts_nnatanngrass20_inW = numpy.unique(nnatanngrass20inW, return_counts=True)
#     nnatanngrass20inW_dict = {labels_nnatanngrass20_inW[i]: counts_nnatanngrass20_inW[i] for i in
#                               range(len(labels_nnatanngrass20_inW))}
#     labels_nnatanngrass17_outW, counts_nnatanngrass17_outW = numpy.unique(nnatanngrass17outW, return_counts=True)
#     nnatanngrass17outW_dict = {labels_nnatanngrass17_outW[i]: counts_nnatanngrass17_outW[i] for i in
#                                range(len(labels_nnatanngrass17_outW))}
#     labels_nnatanngrass18_outW, counts_nnatanngrass18_outW = numpy.unique(nnatanngrass18outW, return_counts=True)
#     nnatanngrass18outW_dict = {labels_nnatanngrass18_outW[i]: counts_nnatanngrass18_outW[i] for i in
#                                range(len(labels_nnatanngrass18_outW))}
#     labels_nnatanngrass19_outW, counts_nnatanngrass19_outW = numpy.unique(nnatanngrass19outW, return_counts=True)
#     nnatanngrass19outW_dict = {labels_nnatanngrass19_outW[i]: counts_nnatanngrass19_outW[i] for i in
#                                range(len(labels_nnatanngrass19_outW))}
#     labels_nnatanngrass20_outW, counts_nnatanngrass20_outW = numpy.unique(nnatanngrass20outW, return_counts=True)
#     nnatanngrass20outW_dict = {labels_nnatanngrass20_outW[i]: counts_nnatanngrass20_outW[i] for i in
#                                range(len(labels_nnatanngrass20_outW))}
#
#     print("Sitecode: [[0]Total obs, [1]Nat Shrub, [2]Nat Subshrub, [3]Nat Per. Herbaceous, [4]Nat Ann. Herbaceous, [5]Nat Per. Grass, [6]Nonnat Ann. Herbaceous, [7]Nonnat Ann. Grass")
#     fulldict17_inW = {}
#     for key in set(list(totals17inW_dict.keys()) + list(natshrub17inW_dict.keys())):
#         try:
#             fulldict17_inW.setdefault(key, []).append(totals17inW_dict[key])
#         except KeyError:
#             fulldict17_inW.setdefault(key, []).append(0)
#         try:
#             fulldict17_inW.setdefault(key, []).append(natshrub17inW_dict[key])
#         except KeyError:
#             fulldict17_inW.setdefault(key, []).append(0)
#         try:
#             fulldict17_inW.setdefault(key, []).append(natsubshrub17inW_dict[key])
#         except KeyError:
#             fulldict17_inW.setdefault(key, []).append(0)
#         try:
#             fulldict17_inW.setdefault(key, []).append(natperherb17inW_dict[key])
#         except KeyError:
#             fulldict17_inW.setdefault(key, []).append(0)
#         try:
#             fulldict17_inW.setdefault(key, []).append(natannherb17inW_dict[key])
#         except KeyError:
#             fulldict17_inW.setdefault(key, []).append(0)
#         try:
#             fulldict17_inW.setdefault(key, []).append(natpergrass17inW_dict[key])
#         except KeyError:
#             fulldict17_inW.setdefault(key, []).append(0)
#         try:
#             fulldict17_inW.setdefault(key, []).append(nnatannherb17inW_dict[key])
#         except KeyError:
#             fulldict17_inW.setdefault(key, []).append(0)
#         try:
#             fulldict17_inW.setdefault(key, []).append(nnatanngrass17inW_dict[key])
#         except KeyError:
#             fulldict17_inW.setdefault(key, []).append(0)
#     fulldict17_outW = {}
#     for key in set(list(totals17outW_dict.keys()) + list(natshrub17outW_dict.keys())):
#         try:
#             fulldict17_outW.setdefault(key, []).append(totals17outW_dict[key])
#         except KeyError:
#             fulldict17_outW.setdefault(key, []).append(0)
#         try:
#             fulldict17_outW.setdefault(key, []).append(natshrub17outW_dict[key])
#         except KeyError:
#             fulldict17_outW.setdefault(key, []).append(0)
#         try:
#             fulldict17_outW.setdefault(key, []).append(natsubshrub17outW_dict[key])
#         except KeyError:
#             fulldict17_outW.setdefault(key, []).append(0)
#         try:
#             fulldict17_outW.setdefault(key, []).append(natperherb17outW_dict[key])
#         except KeyError:
#             fulldict17_outW.setdefault(key, []).append(0)
#         try:
#             fulldict17_outW.setdefault(key, []).append(natannherb17outW_dict[key])
#         except KeyError:
#             fulldict17_outW.setdefault(key, []).append(0)
#         try:
#             fulldict17_outW.setdefault(key, []).append(natpergrass17outW_dict[key])
#         except KeyError:
#             fulldict17_outW.setdefault(key, []).append(0)
#         try:
#             fulldict17_outW.setdefault(key, []).append(nnatannherb17outW_dict[key])
#         except KeyError:
#             fulldict17_outW.setdefault(key, []).append(0)
#         try:
#             fulldict17_outW.setdefault(key, []).append(nnatanngrass17outW_dict[key])
#         except KeyError:
#             fulldict17_outW.setdefault(key, []).append(0)
#
#     fulldict18_inW = {}
#     for key in set(list(totals18inW_dict.keys()) + list(natshrub18inW_dict.keys())):
#         try:
#             fulldict18_inW.setdefault(key, []).append(totals18inW_dict[key])
#         except KeyError:
#             fulldict18_inW.setdefault(key, []).append(0)
#         try:
#             fulldict18_inW.setdefault(key, []).append(natshrub18inW_dict[key])
#         except KeyError:
#             fulldict18_inW.setdefault(key, []).append(0)
#         try:
#             fulldict18_inW.setdefault(key, []).append(natsubshrub18inW_dict[key])
#         except KeyError:
#             fulldict18_inW.setdefault(key, []).append(0)
#         try:
#             fulldict18_inW.setdefault(key, []).append(natperherb18inW_dict[key])
#         except KeyError:
#             fulldict18_inW.setdefault(key, []).append(0)
#         try:
#             fulldict18_inW.setdefault(key, []).append(natannherb18inW_dict[key])
#         except KeyError:
#             fulldict18_inW.setdefault(key, []).append(0)
#         try:
#             fulldict18_inW.setdefault(key, []).append(natpergrass18inW_dict[key])
#         except KeyError:
#             fulldict18_inW.setdefault(key, []).append(0)
#         try:
#             fulldict18_inW.setdefault(key, []).append(nnatannherb18inW_dict[key])
#         except KeyError:
#             fulldict18_inW.setdefault(key, []).append(0)
#         try:
#             fulldict18_inW.setdefault(key, []).append(nnatanngrass18inW_dict[key])
#         except KeyError:
#             fulldict18_inW.setdefault(key, []).append(0)
#     fulldict18_outW = {}
#     for key in set(list(totals18outW_dict.keys()) + list(natshrub18outW_dict.keys())):
#         try:
#             fulldict18_outW.setdefault(key, []).append(totals18outW_dict[key])
#         except KeyError:
#             fulldict18_outW.setdefault(key, []).append(0)
#         try:
#             fulldict18_outW.setdefault(key, []).append(natshrub18outW_dict[key])
#         except KeyError:
#             fulldict18_outW.setdefault(key, []).append(0)
#         try:
#             fulldict18_outW.setdefault(key, []).append(natsubshrub18outW_dict[key])
#         except KeyError:
#             fulldict18_outW.setdefault(key, []).append(0)
#         try:
#             fulldict18_outW.setdefault(key, []).append(natperherb18outW_dict[key])
#         except KeyError:
#             fulldict18_outW.setdefault(key, []).append(0)
#         try:
#             fulldict18_outW.setdefault(key, []).append(natannherb18outW_dict[key])
#         except KeyError:
#             fulldict18_outW.setdefault(key, []).append(0)
#         try:
#             fulldict18_outW.setdefault(key, []).append(natpergrass18outW_dict[key])
#         except KeyError:
#             fulldict18_outW.setdefault(key, []).append(0)
#         try:
#             fulldict18_outW.setdefault(key, []).append(nnatannherb18outW_dict[key])
#         except KeyError:
#             fulldict18_outW.setdefault(key, []).append(0)
#         try:
#             fulldict18_outW.setdefault(key, []).append(nnatanngrass18outW_dict[key])
#         except KeyError:
#             fulldict18_outW.setdefault(key, []).append(0)
#
#     fulldict19_inW = {}
#     for key in set(list(totals19inW_dict.keys()) + list(natshrub19inW_dict.keys())):
#         try:
#             fulldict19_inW.setdefault(key, []).append(totals19inW_dict[key])
#         except KeyError:
#             fulldict19_inW.setdefault(key, []).append(0)
#         try:
#             fulldict19_inW.setdefault(key, []).append(natshrub19inW_dict[key])
#         except KeyError:
#             fulldict19_inW.setdefault(key, []).append(0)
#         try:
#             fulldict19_inW.setdefault(key, []).append(natsubshrub19inW_dict[key])
#         except KeyError:
#             fulldict19_inW.setdefault(key, []).append(0)
#         try:
#             fulldict19_inW.setdefault(key, []).append(natperherb19inW_dict[key])
#         except KeyError:
#             fulldict19_inW.setdefault(key, []).append(0)
#         try:
#             fulldict19_inW.setdefault(key, []).append(natannherb19inW_dict[key])
#         except KeyError:
#             fulldict19_inW.setdefault(key, []).append(0)
#         try:
#             fulldict19_inW.setdefault(key, []).append(natpergrass19inW_dict[key])
#         except KeyError:
#             fulldict19_inW.setdefault(key, []).append(0)
#         try:
#             fulldict19_inW.setdefault(key, []).append(nnatannherb19inW_dict[key])
#         except KeyError:
#             fulldict19_inW.setdefault(key, []).append(0)
#         try:
#             fulldict19_inW.setdefault(key, []).append(nnatanngrass19inW_dict[key])
#         except KeyError:
#             fulldict19_inW.setdefault(key, []).append(0)
#     fulldict19_outW = {}
#     for key in set(list(totals19outW_dict.keys()) + list(natshrub19outW_dict.keys())):
#         try:
#             fulldict19_outW.setdefault(key, []).append(totals19outW_dict[key])
#         except KeyError:
#             fulldict19_outW.setdefault(key, []).append(0)
#         try:
#             fulldict19_outW.setdefault(key, []).append(natshrub19outW_dict[key])
#         except KeyError:
#             fulldict19_outW.setdefault(key, []).append(0)
#         try:
#             fulldict19_outW.setdefault(key, []).append(natsubshrub19outW_dict[key])
#         except KeyError:
#             fulldict19_outW.setdefault(key, []).append(0)
#         try:
#             fulldict19_outW.setdefault(key, []).append(natperherb19outW_dict[key])
#         except KeyError:
#             fulldict19_outW.setdefault(key, []).append(0)
#         try:
#             fulldict19_outW.setdefault(key, []).append(natannherb19outW_dict[key])
#         except KeyError:
#             fulldict19_outW.setdefault(key, []).append(0)
#         try:
#             fulldict19_outW.setdefault(key, []).append(natpergrass19outW_dict[key])
#         except KeyError:
#             fulldict19_outW.setdefault(key, []).append(0)
#         try:
#             fulldict19_outW.setdefault(key, []).append(nnatannherb19outW_dict[key])
#         except KeyError:
#             fulldict19_outW.setdefault(key, []).append(0)
#         try:
#             fulldict19_outW.setdefault(key, []).append(nnatanngrass19outW_dict[key])
#         except KeyError:
#             fulldict19_outW.setdefault(key, []).append(0)
#
#     fulldict20_inW = {}
#     for key in set(list(totals20inW_dict.keys()) + list(natshrub20inW_dict.keys())):
#         try:
#             fulldict20_inW.setdefault(key, []).append(totals20inW_dict[key])
#         except KeyError:
#             fulldict20_inW.setdefault(key, []).append(0)
#         try:
#             fulldict20_inW.setdefault(key, []).append(natshrub20inW_dict[key])
#         except KeyError:
#             fulldict20_inW.setdefault(key, []).append(0)
#         try:
#             fulldict20_inW.setdefault(key, []).append(natsubshrub20inW_dict[key])
#         except KeyError:
#             fulldict20_inW.setdefault(key, []).append(0)
#         try:
#             fulldict20_inW.setdefault(key, []).append(natperherb20inW_dict[key])
#         except KeyError:
#             fulldict20_inW.setdefault(key, []).append(0)
#         try:
#             fulldict20_inW.setdefault(key, []).append(natannherb20inW_dict[key])
#         except KeyError:
#             fulldict20_inW.setdefault(key, []).append(0)
#         try:
#             fulldict20_inW.setdefault(key, []).append(natpergrass20inW_dict[key])
#         except KeyError:
#             fulldict20_inW.setdefault(key, []).append(0)
#         try:
#             fulldict20_inW.setdefault(key, []).append(nnatannherb20inW_dict[key])
#         except KeyError:
#             fulldict20_inW.setdefault(key, []).append(0)
#         try:
#             fulldict20_inW.setdefault(key, []).append(nnatanngrass20inW_dict[key])
#         except KeyError:
#             fulldict20_inW.setdefault(key, []).append(0)
#     fulldict20_outW = {}
#     for key in set(list(totals20outW_dict.keys()) + list(natshrub20outW_dict.keys())):
#         try:
#             fulldict20_outW.setdefault(key, []).append(totals20outW_dict[key])
#         except KeyError:
#             fulldict20_outW.setdefault(key, []).append(0)
#         try:
#             fulldict20_outW.setdefault(key, []).append(natshrub20outW_dict[key])
#         except KeyError:
#             fulldict20_outW.setdefault(key, []).append(0)
#         try:
#             fulldict20_outW.setdefault(key, []).append(natsubshrub20outW_dict[key])
#         except KeyError:
#             fulldict20_outW.setdefault(key, []).append(0)
#         try:
#             fulldict20_outW.setdefault(key, []).append(natperherb20outW_dict[key])
#         except KeyError:
#             fulldict20_outW.setdefault(key, []).append(0)
#         try:
#             fulldict20_outW.setdefault(key, []).append(natannherb20outW_dict[key])
#         except KeyError:
#             fulldict20_outW.setdefault(key, []).append(0)
#         try:
#             fulldict20_outW.setdefault(key, []).append(natpergrass20outW_dict[key])
#         except KeyError:
#             fulldict20_outW.setdefault(key, []).append(0)
#         try:
#             fulldict20_outW.setdefault(key, []).append(nnatannherb20outW_dict[key])
#         except KeyError:
#             fulldict20_outW.setdefault(key, []).append(0)
#         try:
#             fulldict20_outW.setdefault(key, []).append(nnatanngrass20outW_dict[key])
#         except KeyError:
#             fulldict20_outW.setdefault(key, []).append(0)
#     print('Dict in Woolsey 2020')
#     for key, value in fulldict20_inW.items():
#         print(key, value)
#     print('Dict in Woolsey 2019')
#     for key, value in fulldict19_inW.items():
#         print(key, value)
#
#     ### Separate dictionaries into separate arrays
#     values17inW = numpy.array(list(fulldict17_inW.values()))
#     values_totals17inW = [sublist[0] for sublist in values17inW]
#     values_nshrub17inW = [sublist[1] for sublist in values17inW]
#     values_nsubshrub17inW = [sublist[2] for sublist in values17inW]
#     values_nperherb17inW = [sublist[3] for sublist in values17inW]
#     values_nannherb17inW = [sublist[4] for sublist in values17inW]
#     values_npergrass17inW = [sublist[5] for sublist in values17inW]
#     values_nnannherb17inW = [sublist[6] for sublist in values17inW]
#     values_nnanngrass17inW = [sublist[7] for sublist in values17inW]
#     values17outW = numpy.array(list(fulldict17_outW.values()))
#     values_totals17outW = [sublist[0] for sublist in values17outW]
#     values_nshrub17outW = [sublist[1] for sublist in values17outW]
#     values_nsubshrub17outW = [sublist[2] for sublist in values17outW]
#     values_nperherb17outW = [sublist[3] for sublist in values17outW]
#     values_nannherb17outW = [sublist[4] for sublist in values17outW]
#     values_npergrass17outW = [sublist[5] for sublist in values17outW]
#     values_nnannherb17outW = [sublist[6] for sublist in values17outW]
#     values_nnanngrass17outW = [sublist[7] for sublist in values17outW]
#
#     values18inW = numpy.array(list(fulldict18_inW.values()))
#     values_totals18inW = [sublist[0] for sublist in values18inW]
#     values_nshrub18inW = [sublist[1] for sublist in values18inW]
#     values_nsubshrub18inW = [sublist[2] for sublist in values18inW]
#     values_nperherb18inW = [sublist[3] for sublist in values18inW]
#     values_nannherb18inW = [sublist[4] for sublist in values18inW]
#     values_npergrass18inW = [sublist[5] for sublist in values18inW]
#     values_nnannherb18inW = [sublist[6] for sublist in values18inW]
#     values_nnanngrass18inW = [sublist[7] for sublist in values18inW]
#     values18outW = numpy.array(list(fulldict18_outW.values()))
#     values_totals18outW = [sublist[0] for sublist in values18outW]
#     values_nshrub18outW = [sublist[1] for sublist in values18outW]
#     values_nsubshrub18outW = [sublist[2] for sublist in values18outW]
#     values_nperherb18outW = [sublist[3] for sublist in values18outW]
#     values_nannherb18outW = [sublist[4] for sublist in values18outW]
#     values_npergrass18outW = [sublist[5] for sublist in values18outW]
#     values_nnannherb18outW = [sublist[6] for sublist in values18outW]
#     values_nnanngrass18outW = [sublist[7] for sublist in values18outW]
#
#     values19inW = numpy.array(list(fulldict19_inW.values()))
#     values_totals19inW = [sublist[0] for sublist in values19inW]
#     values_nshrub19inW = [sublist[1] for sublist in values19inW]
#     values_nsubshrub19inW = [sublist[2] for sublist in values19inW]
#     values_nperherb19inW = [sublist[3] for sublist in values19inW]
#     values_nannherb19inW = [sublist[4] for sublist in values19inW]
#     values_npergrass19inW = [sublist[5] for sublist in values19inW]
#     values_nnannherb19inW = [sublist[6] for sublist in values19inW]
#     values_nnanngrass19inW = [sublist[7] for sublist in values19inW]
#     values19outW = numpy.array(list(fulldict19_outW.values()))
#     values_totals19outW = [sublist[0] for sublist in values19outW]
#     values_nshrub19outW = [sublist[1] for sublist in values19outW]
#     values_nsubshrub19outW = [sublist[2] for sublist in values19outW]
#     values_nperherb19outW = [sublist[3] for sublist in values19outW]
#     values_nannherb19outW = [sublist[4] for sublist in values19outW]
#     values_npergrass19outW = [sublist[5] for sublist in values19outW]
#     values_nnannherb19outW = [sublist[6] for sublist in values19outW]
#     values_nnanngrass19outW = [sublist[7] for sublist in values19outW]
#
#     values20inW = numpy.array(list(fulldict20_inW.values()))
#     values_totals20inW = [sublist[0] for sublist in values20inW]
#     values_nshrub20inW = [sublist[1] for sublist in values20inW]
#     values_nsubshrub20inW = [sublist[2] for sublist in values20inW]
#     values_nperherb20inW = [sublist[3] for sublist in values20inW]
#     values_nannherb20inW = [sublist[4] for sublist in values20inW]
#     values_npergrass20inW = [sublist[5] for sublist in values20inW]
#     values_nnannherb20inW = [sublist[6] for sublist in values20inW]
#     values_nnanngrass20inW = [sublist[7] for sublist in values20inW]
#     values20outW = numpy.array(list(fulldict20_outW.values()))
#     values_totals20outW = [sublist[0] for sublist in values20outW]
#     values_nshrub20outW = [sublist[1] for sublist in values20outW]
#     values_nsubshrub20outW = [sublist[2] for sublist in values20outW]
#     values_nperherb20outW = [sublist[3] for sublist in values20outW]
#     values_nannherb20outW = [sublist[4] for sublist in values20outW]
#     values_npergrass20outW = [sublist[5] for sublist in values20outW]
#     values_nnannherb20outW = [sublist[6] for sublist in values20outW]
#     values_nnanngrass20outW = [sublist[7] for sublist in values20outW]
#
#     ### Turn arrays into numpy arrays to run numpy arithmatic
#     totals17inW = numpy.asarray(values_totals17inW)
#     nshrub17inW = numpy.asarray(values_nshrub17inW)
#     nsubshrub17inW = numpy.asarray(values_nsubshrub17inW)
#     nperherb17inW = numpy.asarray(values_nperherb17inW)
#     nannherb17inW = numpy.asarray(values_nannherb17inW)
#     npergrass17inW = numpy.asarray(values_npergrass17inW)
#     nnannherb17inW = numpy.asarray(values_nnannherb17inW)
#     nnanngrass17inW = numpy.asarray(values_nnanngrass17inW)
#     totals17outW = numpy.asarray(values_totals17outW)
#     nshrub17outW = numpy.asarray(values_nshrub17outW)
#     nsubshrub17outW = numpy.asarray(values_nsubshrub17outW)
#     nperherb17outW = numpy.asarray(values_nperherb17outW)
#     nannherb17outW = numpy.asarray(values_nannherb17outW)
#     npergrass17outW = numpy.asarray(values_npergrass17outW)
#     nnannherb17outW = numpy.asarray(values_nnannherb17outW)
#     nnanngrass17outW = numpy.asarray(values_nnanngrass17outW)
#
#     totals18inW = numpy.asarray(values_totals18inW)
#     nshrub18inW = numpy.asarray(values_nshrub18inW)
#     nsubshrub18inW = numpy.asarray(values_nsubshrub18inW)
#     nperherb18inW = numpy.asarray(values_nperherb18inW)
#     nannherb18inW = numpy.asarray(values_nannherb18inW)
#     npergrass18inW = numpy.asarray(values_npergrass18inW)
#     nnannherb18inW = numpy.asarray(values_nnannherb18inW)
#     nnanngrass18inW = numpy.asarray(values_nnanngrass18inW)
#     totals18outW = numpy.asarray(values_totals18outW)
#     nshrub18outW = numpy.asarray(values_nshrub18outW)
#     nsubshrub18outW = numpy.asarray(values_nsubshrub18outW)
#     nperherb18outW = numpy.asarray(values_nperherb18outW)
#     nannherb18outW = numpy.asarray(values_nannherb18outW)
#     npergrass18outW = numpy.asarray(values_npergrass18outW)
#     nnannherb18outW = numpy.asarray(values_nnannherb18outW)
#     nnanngrass18outW = numpy.asarray(values_nnanngrass18outW)
#
#     totals19inW = numpy.asarray(values_totals19inW)
#     nshrub19inW = numpy.asarray(values_nshrub19inW)
#     nsubshrub19inW = numpy.asarray(values_nsubshrub19inW)
#     nperherb19inW = numpy.asarray(values_nperherb19inW)
#     nannherb19inW = numpy.asarray(values_nannherb19inW)
#     npergrass19inW = numpy.asarray(values_npergrass19inW)
#     nnannherb19inW = numpy.asarray(values_nnannherb19inW)
#     nnanngrass19inW = numpy.asarray(values_nnanngrass19inW)
#     totals19outW = numpy.asarray(values_totals19outW)
#     nshrub19outW = numpy.asarray(values_nshrub19outW)
#     nsubshrub19outW = numpy.asarray(values_nsubshrub19outW)
#     nperherb19outW = numpy.asarray(values_nperherb19outW)
#     nannherb19outW = numpy.asarray(values_nannherb19outW)
#     npergrass19outW = numpy.asarray(values_npergrass19outW)
#     nnannherb19outW = numpy.asarray(values_nnannherb19outW)
#     nnanngrass19outW = numpy.asarray(values_nnanngrass19outW)
#
#     totals20inW = numpy.asarray(values_totals20inW)
#     nshrub20inW = numpy.asarray(values_nshrub20inW)
#     nsubshrub20inW = numpy.asarray(values_nsubshrub20inW)
#     nperherb20inW = numpy.asarray(values_nperherb20inW)
#     nannherb20inW = numpy.asarray(values_nannherb20inW)
#     npergrass20inW = numpy.asarray(values_npergrass20inW)
#     nnannherb20inW = numpy.asarray(values_nnannherb20inW)
#     nnanngrass20inW = numpy.asarray(values_nnanngrass20inW)
#     totals20outW = numpy.asarray(values_totals20outW)
#     nshrub20outW = numpy.asarray(values_nshrub20outW)
#     nsubshrub20outW = numpy.asarray(values_nsubshrub20outW)
#     nperherb20outW = numpy.asarray(values_nperherb20outW)
#     nannherb20outW = numpy.asarray(values_nannherb20outW)
#     npergrass20outW = numpy.asarray(values_npergrass20outW)
#     nnannherb20outW = numpy.asarray(values_nnannherb20outW)
#     nnanngrass20outW = numpy.asarray(values_nnanngrass20outW)
#
#     ### Find percentage of each functional group within the transects
#     percent_nshrub17inW = numpy.divide(nshrub17inW, totals17inW)
#     percent_nsubshrub17inW = numpy.divide(nsubshrub17inW, totals17inW)
#     percent_nperherb17inW = numpy.divide(nperherb17inW, totals17inW)
#     percent_nannherb17inW = numpy.divide(nannherb17inW, totals17inW)
#     percent_npergrass17inW = numpy.divide(npergrass17inW, totals17inW)
#     percent_nnannherb17inW = numpy.divide(nnannherb17inW, totals17inW)
#     percent_nnanngrass17inW = numpy.divide(nnanngrass17inW, totals17inW)
#     percent_nshrub17outW = numpy.divide(nshrub17outW, totals17outW)
#     percent_nsubshrub17outW = numpy.divide(nsubshrub17outW, totals17outW)
#     percent_nperherb17outW = numpy.divide(nperherb17outW, totals17outW)
#     percent_nannherb17outW = numpy.divide(nannherb17outW, totals17outW)
#     percent_npergrass17outW = numpy.divide(npergrass17outW, totals17outW)
#     percent_nnannherb17outW = numpy.divide(nnannherb17outW, totals17outW)
#     percent_nnanngrass17outW = numpy.divide(nnanngrass17outW, totals17outW)
#
#     percent_nshrub18inW = numpy.divide(nshrub18inW, totals18inW)
#     percent_nsubshrub18inW = numpy.divide(nsubshrub18inW, totals18inW)
#     percent_nperherb18inW = numpy.divide(nperherb18inW, totals18inW)
#     percent_nannherb18inW = numpy.divide(nannherb18inW, totals18inW)
#     percent_npergrass18inW = numpy.divide(npergrass18inW, totals18inW)
#     percent_nnannherb18inW = numpy.divide(nnannherb18inW, totals18inW)
#     percent_nnanngrass18inW = numpy.divide(nnanngrass18inW, totals18inW)
#     percent_nshrub18outW = numpy.divide(nshrub18outW, totals18outW)
#     percent_nsubshrub18outW = numpy.divide(nsubshrub18outW, totals18outW)
#     percent_nperherb18outW = numpy.divide(nperherb18outW, totals18outW)
#     percent_nannherb18outW = numpy.divide(nannherb18outW, totals18outW)
#     percent_npergrass18outW = numpy.divide(npergrass18outW, totals18outW)
#     percent_nnannherb18outW = numpy.divide(nnannherb18outW, totals18outW)
#     percent_nnanngrass18outW = numpy.divide(nnanngrass18outW, totals18outW)
#
#     percent_nshrub19inW = numpy.divide(nshrub19inW, totals19inW)
#     percent_nsubshrub19inW = numpy.divide(nsubshrub19inW, totals19inW)
#     percent_nperherb19inW = numpy.divide(nperherb19inW, totals19inW)
#     percent_nannherb19inW = numpy.divide(nannherb19inW, totals19inW)
#     percent_npergrass19inW = numpy.divide(npergrass19inW, totals19inW)
#     percent_nnannherb19inW = numpy.divide(nnannherb19inW, totals19inW)
#     percent_nnanngrass19inW = numpy.divide(nnanngrass19inW, totals19inW)
#     percent_nshrub19outW = numpy.divide(nshrub19outW, totals19outW)
#     percent_nsubshrub19outW = numpy.divide(nsubshrub19outW, totals19outW)
#     percent_nperherb19outW = numpy.divide(nperherb19outW, totals19outW)
#     percent_nannherb19outW = numpy.divide(nannherb19outW, totals19outW)
#     percent_npergrass19outW = numpy.divide(npergrass19outW, totals19outW)
#     percent_nnannherb19outW = numpy.divide(nnannherb19outW, totals19outW)
#     percent_nnanngrass19outW = numpy.divide(nnanngrass19outW, totals19outW)
#
#     percent_nshrub20inW = numpy.divide(nshrub20inW, totals20inW)
#     percent_nsubshrub20inW = numpy.divide(nsubshrub20inW, totals20inW)
#     percent_nperherb20inW = numpy.divide(nperherb20inW, totals20inW)
#     percent_nannherb20inW = numpy.divide(nannherb20inW, totals20inW)
#     percent_npergrass20inW = numpy.divide(npergrass20inW, totals20inW)
#     percent_nnannherb20inW = numpy.divide(nnannherb20inW, totals20inW)
#     percent_nnanngrass20inW = numpy.divide(nnanngrass20inW, totals20inW)
#     percent_nshrub20outW = numpy.divide(nshrub20outW, totals20outW)
#     percent_nsubshrub20outW = numpy.divide(nsubshrub20outW, totals20outW)
#     percent_nperherb20outW = numpy.divide(nperherb20outW, totals20outW)
#     percent_nannherb20outW = numpy.divide(nannherb20outW, totals20outW)
#     percent_npergrass20outW = numpy.divide(npergrass20outW, totals20outW)
#     percent_nnannherb20outW = numpy.divide(nnannherb20outW, totals20outW)
#     percent_nnanngrass20outW = numpy.divide(nnanngrass20outW, totals20outW)
#
#     ### Find mean of each functional groups from all transects
#     mean_nshrub17inW = numpy.around(numpy.mean(percent_nshrub17inW) * 100)
#     mean_nsubshrub17inW = numpy.around(numpy.mean(percent_nsubshrub17inW) * 100)
#     mean_nperherb17inW = numpy.around(numpy.mean(percent_nperherb17inW) * 100)
#     mean_nannherb17inW = numpy.around(numpy.mean(percent_nannherb17inW) * 100)
#     mean_npergrass17inW = numpy.around(numpy.mean(percent_npergrass17inW) * 100)
#     mean_nnannherb17inW = numpy.around(numpy.mean(percent_nnannherb17inW) * 100)
#     mean_nnanngrass17inW = numpy.around(numpy.mean(percent_nnanngrass17inW) * 100)
#     mean_nshrub17outW = numpy.around(numpy.mean(percent_nshrub17outW) * 100)
#     mean_nsubshrub17outW = numpy.around(numpy.mean(percent_nsubshrub17outW) * 100)
#     mean_nperherb17outW = numpy.around(numpy.mean(percent_nperherb17outW) * 100)
#     mean_nannherb17outW = numpy.around(numpy.mean(percent_nannherb17outW) * 100)
#     mean_npergrass17outW = numpy.around(numpy.mean(percent_npergrass17outW) * 100)
#     mean_nnannherb17outW = numpy.around(numpy.mean(percent_nnannherb17outW) * 100)
#     mean_nnanngrass17outW = numpy.around(numpy.mean(percent_nnanngrass17outW) * 100)
#
#     mean_nshrub18inW = numpy.around(numpy.mean(percent_nshrub18inW) * 100)
#     mean_nsubshrub18inW = numpy.around(numpy.mean(percent_nsubshrub18inW) * 100)
#     mean_nperherb18inW = numpy.around(numpy.mean(percent_nperherb18inW) * 100)
#     mean_nannherb18inW = numpy.around(numpy.mean(percent_nannherb18inW) * 100)
#     mean_npergrass18inW = numpy.around(numpy.mean(percent_npergrass18inW) * 100)
#     mean_nnannherb18inW = numpy.around(numpy.mean(percent_nnannherb18inW) * 100)
#     mean_nnanngrass18inW = numpy.around(numpy.mean(percent_nnanngrass18inW) * 100)
#     mean_nshrub18outW = numpy.around(numpy.mean(percent_nshrub18outW) * 100)
#     mean_nsubshrub18outW = numpy.around(numpy.mean(percent_nsubshrub18outW) * 100)
#     mean_nperherb18outW = numpy.around(numpy.mean(percent_nperherb18outW) * 100)
#     mean_nannherb18outW = numpy.around(numpy.mean(percent_nannherb18outW) * 100)
#     mean_npergrass18outW = numpy.around(numpy.mean(percent_npergrass18outW) * 100)
#     mean_nnannherb18outW = numpy.around(numpy.mean(percent_nnannherb18outW) * 100)
#     mean_nnanngrass18outW = numpy.around(numpy.mean(percent_nnanngrass18outW) * 100)
#
#     mean_nshrub19inW = numpy.around(numpy.mean(percent_nshrub19inW) * 100)
#     mean_nsubshrub19inW = numpy.around(numpy.mean(percent_nsubshrub19inW) * 100)
#     mean_nperherb19inW = numpy.around(numpy.mean(percent_nperherb19inW) * 100)
#     mean_nannherb19inW = numpy.around(numpy.mean(percent_nannherb19inW) * 100)
#     mean_npergrass19inW = numpy.around(numpy.mean(percent_npergrass19inW) * 100)
#     mean_nnannherb19inW = numpy.around(numpy.mean(percent_nnannherb19inW) * 100)
#     mean_nnanngrass19inW = numpy.around(numpy.mean(percent_nnanngrass19inW) * 100)
#     mean_nshrub19outW = numpy.around(numpy.mean(percent_nshrub19outW) * 100)
#     mean_nsubshrub19outW = numpy.around(numpy.mean(percent_nsubshrub19outW) * 100)
#     mean_nperherb19outW = numpy.around(numpy.mean(percent_nperherb19outW) * 100)
#     mean_nannherb19outW = numpy.around(numpy.mean(percent_nannherb19outW) * 100)
#     mean_npergrass19outW = numpy.around(numpy.mean(percent_npergrass19outW) * 100)
#     mean_nnannherb19outW = numpy.around(numpy.mean(percent_nnannherb19outW) * 100)
#     mean_nnanngrass19outW = numpy.around(numpy.mean(percent_nnanngrass19outW) * 100)
#
#     mean_nshrub20inW = numpy.around(numpy.mean(percent_nshrub20inW) * 100)
#     mean_nsubshrub20inW = numpy.around(numpy.mean(percent_nsubshrub20inW) * 100)
#     mean_nperherb20inW = numpy.around(numpy.mean(percent_nperherb20inW) * 100)
#     mean_nannherb20inW = numpy.around(numpy.mean(percent_nannherb20inW) * 100)
#     mean_npergrass20inW = numpy.around(numpy.mean(percent_npergrass20inW) * 100)
#     mean_nnannherb20inW = numpy.around(numpy.mean(percent_nnannherb20inW) * 100)
#     mean_nnanngrass20inW = numpy.around(numpy.mean(percent_nnanngrass20inW) * 100)
#     mean_nshrub20outW = numpy.around(numpy.mean(percent_nshrub20outW) * 100)
#     mean_nsubshrub20outW = numpy.around(numpy.mean(percent_nsubshrub20outW) * 100)
#     mean_nperherb20outW = numpy.around(numpy.mean(percent_nperherb20outW) * 100)
#     mean_nannherb20outW = numpy.around(numpy.mean(percent_nannherb20outW) * 100)
#     mean_npergrass20outW = numpy.around(numpy.mean(percent_npergrass20outW) * 100)
#     mean_nnannherb20outW = numpy.around(numpy.mean(percent_nnannherb20outW) * 100)
#     mean_nnanngrass20outW = numpy.around(numpy.mean(percent_nnanngrass20outW) * 100)
#
#     # ### Find standard deviation of the mean
#     # sd_nshrub17inW = numpy.around(numpy.std(percent_nshrub17inW) * 100, 2)
#     # sd_nsubshrub17inW = numpy.around(numpy.std(percent_nsubshrub17inW) * 100, 2)
#     # sd_nperherb17inW = numpy.around(numpy.std(percent_nperherb17inW) * 100, 2)
#     # sd_nannherb17inW = numpy.around(numpy.std(percent_nannherb17inW) * 100, 2)
#     # sd_npergrass17inW = numpy.around(numpy.std(percent_npergrass17inW) * 100, 2)
#     # sd_nnannherb17inW = numpy.around(numpy.std(percent_nnannherb17inW) * 100, 2)
#     # sd_nnanngrass17inW = numpy.around(numpy.std(percent_nnanngrass17inW) * 100, 2)
#     # sd_nshrub17outW = numpy.around(numpy.std(percent_nshrub17outW) * 100, 2)
#     # sd_nsubshrub17outW = numpy.around(numpy.std(percent_nsubshrub17outW) * 100, 2)
#     # sd_nperherb17outW = numpy.around(numpy.std(percent_nperherb17outW) * 100, 2)
#     # sd_nannherb17outW = numpy.around(numpy.std(percent_nannherb17outW) * 100, 2)
#     # sd_npergrass17outW = numpy.around(numpy.std(percent_npergrass17outW) * 100, 2)
#     # sd_nnannherb17outW = numpy.around(numpy.std(percent_nnannherb17outW) * 100, 2)
#     # sd_nnanngrass17outW = numpy.around(numpy.std(percent_nnanngrass17outW) * 100, 2)
#     #
#     # sd_nshrub18inW = numpy.around(numpy.std(percent_nshrub18inW) * 100, 2)
#     # sd_nsubshrub18inW = numpy.around(numpy.std(percent_nsubshrub18inW) * 100, 2)
#     # sd_nperherb18inW = numpy.around(numpy.std(percent_nperherb18inW) * 100, 2)
#     # sd_nannherb18inW = numpy.around(numpy.std(percent_nannherb18inW) * 100, 2)
#     # sd_npergrass18inW = numpy.around(numpy.std(percent_npergrass18inW) * 100, 2)
#     # sd_nnannherb18inW = numpy.around(numpy.std(percent_nnannherb18inW) * 100, 2)
#     # sd_nnanngrass18inW = numpy.around(numpy.std(percent_nnanngrass18inW) * 100, 2)
#     # sd_nshrub18outW = numpy.around(numpy.std(percent_nshrub18outW) * 100, 2)
#     # sd_nsubshrub18outW = numpy.around(numpy.std(percent_nsubshrub18outW) * 100, 2)
#     # sd_nperherb18outW = numpy.around(numpy.std(percent_nperherb18outW) * 100, 2)
#     # sd_nannherb18outW = numpy.around(numpy.std(percent_nannherb18outW) * 100, 2)
#     # sd_npergrass18outW = numpy.around(numpy.std(percent_npergrass18outW) * 100, 2)
#     # sd_nnannherb18outW = numpy.around(numpy.std(percent_nnannherb18outW) * 100, 2)
#     # sd_nnanngrass18outW = numpy.around(numpy.std(percent_nnanngrass18outW) * 100, 2)
#     #
#     # sd_nshrub19inW = numpy.around(numpy.std(percent_nshrub19inW) * 100, 2)
#     # sd_nsubshrub19inW = numpy.around(numpy.std(percent_nsubshrub19inW) * 100, 2)
#     # sd_nperherb19inW = numpy.around(numpy.std(percent_nperherb19inW) * 100, 2)
#     # sd_nannherb19inW = numpy.around(numpy.std(percent_nannherb19inW) * 100, 2)
#     # sd_npergrass19inW = numpy.around(numpy.std(percent_npergrass19inW) * 100, 2)
#     # sd_nnannherb19inW = numpy.around(numpy.std(percent_nnannherb19inW) * 100, 2)
#     # sd_nnanngrass19inW = numpy.around(numpy.std(percent_nnanngrass19inW) * 100, 2)
#     # sd_nshrub19outW = numpy.around(numpy.std(percent_nshrub19outW) * 100, 2)
#     # sd_nsubshrub19outW = numpy.around(numpy.std(percent_nsubshrub19outW) * 100, 2)
#     # sd_nperherb19outW = numpy.around(numpy.std(percent_nperherb19outW) * 100, 2)
#     # sd_nannherb19outW = numpy.around(numpy.std(percent_nannherb19outW) * 100, 2)
#     # sd_npergrass19outW = numpy.around(numpy.std(percent_npergrass19outW) * 100, 2)
#     # sd_nnannherb19outW = numpy.around(numpy.std(percent_nnannherb19outW) * 100, 2)
#     # sd_nnanngrass19outW = numpy.around(numpy.std(percent_nnanngrass19outW) * 100, 2)
#     #
#     # sd_nshrub20inW = numpy.around(numpy.std(percent_nshrub20inW) * 100, 2)
#     # sd_nsubshrub20inW = numpy.around(numpy.std(percent_nsubshrub20inW) * 100, 2)
#     # sd_nperherb20inW = numpy.around(numpy.std(percent_nperherb20inW) * 100, 2)
#     # sd_nannherb20inW = numpy.around(numpy.std(percent_nannherb20inW) * 100, 2)
#     # sd_npergrass20inW = numpy.around(numpy.std(percent_npergrass20inW) * 100, 2)
#     # sd_nnannherb20inW = numpy.around(numpy.std(percent_nnannherb20inW) * 100, 2)
#     # sd_nnanngrass20inW = numpy.around(numpy.std(percent_nnanngrass20inW) * 100, 2)
#     # sd_nshrub20outW = numpy.around(numpy.std(percent_nshrub20outW) * 100, 2)
#     # sd_nsubshrub20outW = numpy.around(numpy.std(percent_nsubshrub20outW) * 100, 2)
#     # sd_nperherb20outW = numpy.around(numpy.std(percent_nperherb20outW) * 100, 2)
#     # sd_nannherb20outW = numpy.around(numpy.std(percent_nannherb20outW) * 100, 2)
#     # sd_npergrass20outW = numpy.around(numpy.std(percent_npergrass20outW) * 100, 2)
#     # sd_nnannherb20outW = numpy.around(numpy.std(percent_nnannherb20outW) * 100, 2)
#     # sd_nnanngrass20outW = numpy.around(numpy.std(percent_nnanngrass20outW) * 100, 2)
#     #
#     # ### Find stadard error of the mean
#     # sem_nshrub17inW = numpy.around(sem(percent_nshrub17inW) * 100, 2)
#     # sem_nsubshrub17inW = numpy.around(sem(percent_nsubshrub17inW) * 100, 2)
#     # sem_nperherb17inW = numpy.around(sem(percent_nperherb17inW) * 100, 2)
#     # sem_nannherb17inW = numpy.around(sem(percent_nannherb17inW) * 100, 2)
#     # sem_npergrass17inW = numpy.around(sem(percent_npergrass17inW) * 100, 2)
#     # sem_nnannherb17inW = numpy.around(sem(percent_nnannherb17inW) * 100, 2)
#     # sem_nnanngrass17inW = numpy.around(sem(percent_nnanngrass17inW) * 100, 2)
#     # sem_nshrub17outW = numpy.around(sem(percent_nshrub17outW) * 100, 2)
#     # sem_nsubshrub17outW = numpy.around(sem(percent_nsubshrub17outW) * 100, 2)
#     # sem_nperherb17outW = numpy.around(sem(percent_nperherb17outW) * 100, 2)
#     # sem_nannherb17outW = numpy.around(sem(percent_nannherb17outW) * 100, 2)
#     # sem_npergrass17outW = numpy.around(sem(percent_npergrass17outW) * 100, 2)
#     # sem_nnannherb17outW = numpy.around(sem(percent_nnannherb17outW) * 100, 2)
#     # sem_nnanngrass17outW = numpy.around(sem(percent_nnanngrass17outW) * 100, 2)
#     #
#     # sem_nshrub18inW = numpy.around(sem(percent_nshrub18inW) * 100, 2)
#     # sem_nsubshrub18inW = numpy.around(sem(percent_nsubshrub18inW) * 100, 2)
#     # sem_nperherb18inW = numpy.around(sem(percent_nperherb18inW) * 100, 2)
#     # sem_nannherb18inW = numpy.around(sem(percent_nannherb18inW) * 100, 2)
#     # sem_npergrass18inW = numpy.around(sem(percent_npergrass18inW) * 100, 2)
#     # sem_nnannherb18inW = numpy.around(sem(percent_nnannherb18inW) * 100, 2)
#     # sem_nnanngrass18inW = numpy.around(sem(percent_nnanngrass18inW) * 100, 2)
#     # sem_nshrub18outW = numpy.around(sem(percent_nshrub18outW) * 100, 2)
#     # sem_nsubshrub18outW = numpy.around(sem(percent_nsubshrub18outW) * 100, 2)
#     # sem_nperherb18outW = numpy.around(sem(percent_nperherb18outW) * 100, 2)
#     # sem_nannherb18outW = numpy.around(sem(percent_nannherb18outW) * 100, 2)
#     # sem_npergrass18outW = numpy.around(sem(percent_npergrass18outW) * 100, 2)
#     # sem_nnannherb18outW = numpy.around(sem(percent_nnannherb18outW) * 100, 2)
#     # sem_nnanngrass18outW = numpy.around(sem(percent_nnanngrass18outW) * 100, 2)
#     #
#     # sem_nshrub19inW = numpy.around(sem(percent_nshrub19inW) * 100, 2)
#     # sem_nsubshrub19inW = numpy.around(sem(percent_nsubshrub19inW) * 100, 2)
#     # sem_nperherb19inW = numpy.around(sem(percent_nperherb19inW) * 100, 2)
#     # sem_nannherb19inW = numpy.around(sem(percent_nannherb19inW) * 100, 2)
#     # sem_npergrass19inW = numpy.around(sem(percent_npergrass19inW) * 100, 2)
#     # sem_nnannherb19inW = numpy.around(sem(percent_nnannherb19inW) * 100, 2)
#     # sem_nnanngrass19inW = numpy.around(sem(percent_nnanngrass19inW) * 100, 2)
#     # sem_nshrub19outW = numpy.around(sem(percent_nshrub19outW) * 100, 2)
#     # sem_nsubshrub19outW = numpy.around(sem(percent_nsubshrub19outW) * 100, 2)
#     # sem_nperherb19outW = numpy.around(sem(percent_nperherb19outW) * 100, 2)
#     # sem_nannherb19outW = numpy.around(sem(percent_nannherb19outW) * 100, 2)
#     # sem_npergrass19outW = numpy.around(sem(percent_npergrass19outW) * 100, 2)
#     # sem_nnannherb19outW = numpy.around(sem(percent_nnannherb19outW) * 100, 2)
#     # sem_nnanngrass19outW = numpy.around(sem(percent_nnanngrass19outW) * 100, 2)
#     #
#     # sem_nshrub20inW = numpy.around(sem(percent_nshrub20inW) * 100, 2)
#     # sem_nsubshrub20inW = numpy.around(sem(percent_nsubshrub20inW) * 100, 2)
#     # sem_nperherb20inW = numpy.around(sem(percent_nperherb20inW) * 100, 2)
#     # sem_nannherb20inW = numpy.around(sem(percent_nannherb20inW) * 100, 2)
#     # sem_npergrass20inW = numpy.around(sem(percent_npergrass20inW) * 100, 2)
#     # sem_nnannherb20inW = numpy.around(sem(percent_nnannherb20inW) * 100, 2)
#     # sem_nnanngrass20inW = numpy.around(sem(percent_nnanngrass20inW) * 100, 2)
#     # sem_nshrub20outW = numpy.around(sem(percent_nshrub20outW) * 100, 2)
#     # sem_nsubshrub20outW = numpy.around(sem(percent_nsubshrub20outW) * 100, 2)
#     # sem_nperherb20outW = numpy.around(sem(percent_nperherb20outW) * 100, 2)
#     # sem_nannherb20outW = numpy.around(sem(percent_nannherb20outW) * 100, 2)
#     # sem_npergrass20outW = numpy.around(sem(percent_npergrass20outW) * 100, 2)
#     # sem_nnannherb20outW = numpy.around(sem(percent_nnannherb20outW) * 100, 2)
#     # sem_nnanngrass20outW = numpy.around(sem(percent_nnanngrass20outW) * 100, 2)
#
#     allin = [[mean_nshrub17inW, mean_nshrub18inW, mean_nshrub19inW, mean_nshrub20inW],
#              [mean_nsubshrub17inW, mean_nsubshrub18inW, mean_nsubshrub19inW, mean_nsubshrub20inW],
#              [mean_nperherb17inW, mean_nperherb18inW, mean_nperherb19inW, mean_nperherb20inW],
#              [mean_nannherb17inW, mean_nannherb18inW, mean_nannherb19inW, mean_nannherb20inW],
#              [mean_npergrass17inW, mean_npergrass18inW, mean_npergrass19inW, mean_npergrass20inW],
#              [mean_nnannherb17inW, mean_nnannherb18inW, mean_nnannherb19inW, mean_nnannherb20inW],
#              [mean_nnanngrass17inW, mean_nnanngrass18inW, mean_nnanngrass19inW, mean_nnanngrass20inW]
#              ]
#
#     allout = [[mean_nshrub17outW, mean_nshrub18outW, mean_nshrub19outW, mean_nshrub20outW],
#               [mean_nsubshrub17outW, mean_nsubshrub18outW, mean_nsubshrub19outW, mean_nsubshrub20outW],
#               [mean_nperherb17outW, mean_nperherb18outW, mean_nperherb19outW, mean_nperherb20outW],
#               [mean_nannherb17outW, mean_nannherb18outW, mean_nannherb19outW, mean_nannherb20outW],
#               [mean_npergrass17outW, mean_npergrass18outW, mean_npergrass19outW, mean_npergrass20outW],
#               [mean_nnannherb17outW, mean_nnannherb18outW, mean_nnannherb19outW, mean_nnannherb20outW],
#               [mean_nnanngrass17outW, mean_nnanngrass18outW, mean_nnanngrass19outW, mean_nnanngrass20outW]
#               ]
#
#     fig, ax = plt.subplots(2, figsize=(6, 6))
#     array = (numpy.arange(4))
#
#     ax[0].bar(array + 0.0, allout[0], color=darkestblue, width=0.1, label='Native shrub')
#     ax[0].bar(array + 0.1, allout[1], color=darkblue, width=0.1, label='Native sub-shrub')
#     ax[0].bar(array + 0.2, allout[2], color=blue, width=0.1, label='Native perennial herbaceous')
#     ax[0].bar(array + 0.3, allout[3], color=lightblue, width=0.1, label='Native annual herbaceous')
#     ax[0].bar(array + 0.4, allout[4], color=lightestblue, width=0.1, label='Native perennial grasses')
#     ax[0].bar(array + 0.5, allout[5], color=gold, width=0.1, label='Non-native annual herbaceous')
#     ax[0].bar(array + 0.6, allout[6], color=rust, width=0.1, label='Non-native annual grasses')
#     ax[0].text(0, 36, 'a)', fontsize=ea_panelletter)  # Outside the Woolsey Fire boundary
#     ax[0].set_xticks(array + 0.3)
#     ax[0].set_xticklabels(['2017', '2018\n(pre-fire)', '2019', '2020'], fontsize=ea_axisnum)
#     ax[0].set_ylabel('Cover (%)', fontsize=ea_axislabel)
#     ax[0].set_yticks([0, 10, 20, 30, 40])
#     ax[0].set_yticklabels(['0', '10', '20', '30', '40'], fontsize=ea_axisnum)
#
#     ax[1].bar(array + 0.0, allin[0], color=darkestblue, width=0.1, label='Native shrub')
#     ax[1].bar(array + 0.1, allin[1], color=darkblue, width=0.1, label='Native sub-shrub')
#     ax[1].bar(array + 0.2, allin[2], color=blue, width=0.1, label='Native perennial herbaceous')
#     ax[1].bar(array + 0.3, allin[3], color=lightblue, width=0.1, label='Native annual herbaceous')
#     ax[1].bar(array + 0.4, allin[4], color=lightestblue, width=0.1, label='Native perennial grasses')
#     ax[1].bar(array + 0.5, allin[5], color=gold, width=0.1, label='Nonnative annual herbaceous')
#     ax[1].bar(array + 0.6, allin[6], color=rust, width=0.1, label='Nonnative annual grasses')
#     ax[1].text(0, 55, 'b)', fontsize=ea_panelletter)  # Within the Woolsey Fire boundary
#     ax[1].set_xticks(array + 0.3)
#     ax[1].set_xticklabels(['2017', '2018\n(pre-fire)', '2019', '2020'], fontsize=ea_axisnum)
#     ax[1].set_ylabel('Cover (%)', fontsize=ea_axislabel)
#     ax[1].set_yticks([0, 10, 20, 30, 40, 50, 60])
#     ax[1].set_yticklabels(['0', '10', '20', '30', '40', '50', '60'], fontsize=ea_axisnum)
#     ax[1].legend(fontsize=ea_legend)
#
#     plt.xlabel('Year', fontsize=ea_axislabel)
#     plt.tight_layout()
#     plt.savefig(
#         "/Users/nicolettastork/Documents/Geography/SMMNRA_Invasive_Species/NPSData/VegData_FromSMMNRA_Modified/Fig_FxnGroupBarchart.png",
#         dpi=600)
#     # plt.show()
#
#     ##### RAW NUMBER TABLE #####
#     table_fxngroupinW = [['Inside Woolsey'],
#         ['years',
#          2017, 2018, 2019, 2020],
#         ['native shrubs:',
#         mean_nshrub17inW, mean_nshrub18inW, mean_nshrub19inW, mean_nshrub20inW],
#         ['native sub-shrubs:',
#         mean_nsubshrub17inW, mean_nsubshrub18inW, mean_nsubshrub19inW, mean_nsubshrub20inW],
#         ['native perennial herbaceous:',
#         mean_nperherb17inW, mean_nperherb18inW, mean_nperherb19inW, mean_nperherb20inW],
#         ['native annual herbaceous:',
#         mean_nannherb17inW, mean_nannherb18inW, mean_nannherb19inW, mean_nannherb20inW],
#         ['native perennial grasses:',
#         mean_npergrass17inW, mean_npergrass18inW, mean_npergrass19inW, mean_npergrass20inW],
#         ['non-native annual herbaceous:',
#         mean_nnannherb17inW, mean_nnannherb18inW, mean_nnannherb19inW, mean_nnannherb20inW],
#         ['non-native annual grasses:',
#         mean_nnanngrass17inW, mean_nnanngrass18inW, mean_nnanngrass19inW, mean_nnanngrass20inW]
#         ]
#     table_fxngroupoutW = [['Outside Woolsey'],
#         ['years',
#          2017, 2018, 2019, 2020],
#         ['native shrubs:',
#         mean_nshrub17outW, mean_nshrub18outW, mean_nshrub19outW, mean_nshrub20outW],
#         ['native sub-shrubs:',
#         mean_nsubshrub17outW, mean_nsubshrub18outW, mean_nsubshrub19outW, mean_nsubshrub20outW],
#         ['native perennial herbaceous:',
#         mean_nperherb17outW, mean_nperherb18outW, mean_nperherb19outW, mean_nperherb20outW],
#         ['native annual herbaceous:',
#         mean_nannherb17outW, mean_nannherb18outW, mean_nannherb19outW, mean_nannherb20outW],
#         ['native perennial grasses:',
#         mean_npergrass17outW, mean_npergrass18outW, mean_npergrass19outW, mean_npergrass20outW],
#         ['non-native annual herbaceous:',
#         mean_nnannherb17outW, mean_nnannherb18outW, mean_nnannherb19outW, mean_nnannherb20outW],
#         ['non-native annual grasses:',
#         mean_nnanngrass17outW, mean_nnanngrass18outW, mean_nnanngrass19outW, mean_nnanngrass20outW]
#         ]
#     print(tabulate(table_fxngroupinW))
#     print(tabulate(table_fxngroupoutW))
#
# fxndict(sitecode_inW_totals_2017, sitecode_inW_totals_2018, sitecode_inW_totals_2019, sitecode_inW_totals_2020,
#         natshrub_2017_inW, natshrub_2018_inW, natshrub_2019_inW, natshrub_2020_inW,
#         natsubshrub_2017_inW, natsubshrub_2018_inW, natsubshrub_2019_inW, natsubshrub_2020_inW,
#         natperherb_2017_inW, natperherb_2018_inW, natperherb_2019_inW, natperherb_2020_inW,
#         natannherb_2017_inW, natannherb_2018_inW, natannherb_2019_inW, natannherb_2020_inW,
#         natpergrass_2017_inW, natpergrass_2018_inW, natpergrass_2019_inW, natpergrass_2020_inW,
#         nnatannherb_2017_inW, nnatannherb_2018_inW, nnatannherb_2019_inW, nnatannherb_2020_inW,
#         nnatanngrass_2017_inW, nnatanngrass_2018_inW, nnatanngrass_2019_inW, nnatanngrass_2020_inW,
#         sitecode_outW_totals_2017, sitecode_outW_totals_2018, sitecode_outW_totals_2019, sitecode_outW_totals_2020,
#         natshrub_2017_outW, natshrub_2018_outW, natshrub_2019_outW, natshrub_2020_outW,
#         natsubshrub_2017_outW, natsubshrub_2018_outW, natsubshrub_2019_outW, natsubshrub_2020_outW,
#         natperherb_2017_outW, natperherb_2018_outW, natperherb_2019_outW, natperherb_2020_outW,
#         natannherb_2017_outW, natannherb_2018_outW, natannherb_2019_outW, natannherb_2020_outW,
#         natpergrass_2017_outW, natpergrass_2018_outW, natpergrass_2019_outW, natpergrass_2020_outW,
#         nnatannherb_2017_outW, nnatannherb_2018_outW, nnatannherb_2019_outW, nnatannherb_2020_outW,
#         nnatanngrass_2017_outW, nnatanngrass_2018_outW, nnatanngrass_2019_outW, nnatanngrass_2020_outW)


########################################################################
##### FIGURE 5: Local Morans: Nonnative hotspot map and scatterplot
########################################################################
# longitude = numpy.array([])
# latitude = numpy.array([])
# longitude17 = [obj.decimalLongitude for name, obj in monitoringpoints.items() if 2017 in obj.year]
# latitude17 = [obj.decimalLatitude for name, obj in monitoringpoints.items() if 2017 in obj.year]
# longitude18 = [obj.decimalLongitude for name, obj in monitoringpoints.items() if 2018 in obj.year]
# latitude18 = [obj.decimalLatitude for name, obj in monitoringpoints.items() if 2018 in obj.year]
# longitude19 = [obj.decimalLongitude for name, obj in monitoringpoints.items() if 2019 in obj.year]
# latitude19 = [obj.decimalLatitude for name, obj in monitoringpoints.items() if 2019 in obj.year]
# longitude20 = [obj.decimalLongitude for name, obj in monitoringpoints.items() if 2020 in obj.year]
# latitude20 = [obj.decimalLatitude for name, obj in monitoringpoints.items() if 2020 in obj.year]
#
# fraction_nonnative17 = numpy.array([])
# fraction_nonnative18 = numpy.array([])
# fraction_nonnative19 = numpy.array([])
# fraction_nonnative20 = numpy.array([])
#
# for name, obj in monitoringpoints.items():
#     longitude = numpy.append(longitude, obj.decimalLongitude)
#     latitude = numpy.append(latitude, obj.decimalLatitude)
#
#     test_indices17 = numpy.where((obj.year == 2017) & (obj.native_status == "Nonnative"))
#     try:
#         frac17 = len(obj.native_status[test_indices17])/len(numpy.where(obj.year == 2017)[0])
#         fraction_nonnative17 = numpy.append(fraction_nonnative17, frac17)
#     except ZeroDivisionError:
#         fraction_nonnative17 = numpy.append(fraction_nonnative17, numpy.nan)
#
#     test_indices18 = numpy.where((obj.year == 2018) & (obj.native_status == "Nonnative"))
#     try:
#         frac18 = len(obj.native_status[test_indices18])/len(numpy.where(obj.year == 2018)[0])
#         fraction_nonnative18 = numpy.append(fraction_nonnative18, frac18)
#     except ZeroDivisionError:
#         fraction_nonnative18 = numpy.append(fraction_nonnative18, numpy.nan)
#
#     test_indices19 = numpy.where((obj.year == 2019) & (obj.native_status == "Nonnative"))
#     try:
#         frac19 = len(obj.native_status[test_indices19])/len(numpy.where(obj.year == 2019)[0])
#         fraction_nonnative19 = numpy.append(fraction_nonnative19, frac19)
#     except ZeroDivisionError:
#         fraction_nonnative19 = numpy.append(fraction_nonnative19, numpy.nan)
#
#     test_indices20 = numpy.where((obj.year == 2020) & (obj.native_status == "Nonnative"))
#     try:
#         frac20 = len(obj.native_status[test_indices20])/len(numpy.where(obj.year == 2020)[0])
#         fraction_nonnative20 = numpy.append(fraction_nonnative20, frac20)
#     except ZeroDivisionError:
#         fraction_nonnative20 = numpy.append(fraction_nonnative20, numpy.nan)
#
# collection17 = plt.hexbin(longitude, latitude, C=fraction_nonnative17, gridsize=(28, 9))
# nncounts17 = collection17.get_array()
# hex_polys17 = collection17.get_paths()[0].vertices
# hex_array17 = []
# for xs, ys in collection17.get_offsets():
#     hex_x = numpy.add(hex_polys17[:, 0],  xs)
#     hex_y = numpy.add(hex_polys17[:, 1],  ys)
#     hex_array17.append(Polygon(numpy.vstack([hex_x, hex_y]).T))
# hex_grid_nonnative17 = gpd.GeoDataFrame({"counts": nncounts17, 'geometry':hex_array17}, crs="EPSG:2955")
# plt.close('all')
# wqnn17 = lps.weights.distance.KNN.from_dataframe(hex_grid_nonnative17, k=5)
# wqnn17.transform = "r"
# ynn17 = hex_grid_nonnative17["counts"]
#
# collection18 = plt.hexbin(longitude, latitude, C=fraction_nonnative18, gridsize=(28, 9))
# nncounts18 = collection18.get_array()
# hex_polys18 = collection18.get_paths()[0].vertices
# hex_array18 = []
# for xs, ys in collection18.get_offsets():
#     hex_x = numpy.add(hex_polys18[:, 0],  xs)
#     hex_y = numpy.add(hex_polys18[:, 1],  ys)
#     hex_array18.append(Polygon(numpy.vstack([hex_x, hex_y]).T))
# hex_grid_nonnative18 = gpd.GeoDataFrame({"counts": nncounts18, 'geometry':hex_array18}, crs="EPSG:2955")
# plt.close('all')
# wqnn18 = lps.weights.distance.KNN.from_dataframe(hex_grid_nonnative18, k=5)
# wqnn18.transform = "r"
# ynn18 = hex_grid_nonnative18["counts"]
#
# collection19 = plt.hexbin(longitude, latitude, C=fraction_nonnative19, gridsize=(28, 9))
# nncounts19 = collection19.get_array()
# hex_polys19 = collection19.get_paths()[0].vertices
# hex_array19 = []
# for xs, ys in collection19.get_offsets():
#     hex_x = numpy.add(hex_polys19[:, 0],  xs)
#     hex_y = numpy.add(hex_polys19[:, 1],  ys)
#     hex_array19.append(Polygon(numpy.vstack([hex_x, hex_y]).T))
# hex_grid_nonnative19 = gpd.GeoDataFrame({"counts": nncounts19, 'geometry':hex_array19}, crs="EPSG:2955")
# plt.close('all')
# wqnn19 = lps.weights.distance.KNN.from_dataframe(hex_grid_nonnative19, k=5)
# wqnn19.transform = "r"
# ynn19 = hex_grid_nonnative19["counts"]
#
# collection = plt.hexbin(longitude, latitude, C=fraction_nonnative20, gridsize=(28, 9))
# nncounts = collection.get_array()
# hex_polys = collection.get_paths()[0].vertices
# hex_array = []
# for xs, ys in collection.get_offsets():
#     hex_x = numpy.add(hex_polys[:, 0],  xs)
#     hex_y = numpy.add(hex_polys[:, 1],  ys)
#     hex_array.append(Polygon(numpy.vstack([hex_x, hex_y]).T))
# hex_grid_nonnative20 = gpd.GeoDataFrame({"counts": nncounts, 'geometry': hex_array}, crs="EPSG:2955")
# plt.close('all')
# wqnn = lps.weights.distance.KNN.from_dataframe(hex_grid_nonnative20, k=5)
# wqnn.transform = "r"
# ynn = hex_grid_nonnative20["counts"]
#
# ### Set up the local moran to be used in the Lisa Cluster map.
# li17 = Moran_Local(ynn17, wqnn17)
# li18 = Moran_Local(ynn18, wqnn18)
# li19 = Moran_Local(ynn19, wqnn19)
# li = Moran_Local(ynn, wqnn)
#
# ### More info on this code can be found: https://pysal.org/esda/notebooks/spatialautocorrelation.html
# sig17 = 1 * (li17.p_sim < 0.05)
# hotspot17 = 1 * (sig17 * li17.q==1)
# nn_in_n17 = 2 * (sig17 * li17.q==2)
# coldspot17 = 3 * (sig17 * li17.q==3)
# n_in_nn17 = 4 * (sig17 * li17.q==4)
# spots17 = hotspot17 + nn_in_n17 + coldspot17 + n_in_nn17
# spot_labels17 = ['1 Random', '2 Hotspot', '3 Non-native in native', '4 Coldspot', '5 Native in non-native']
# labels17 = [spot_labels17[i] for i in spots17]
# hmap17 = colors.ListedColormap(['#c0c0c0', rust, gold, darkblue, lightblue])
#
# sig18 = 1 * (li18.p_sim < 0.05)
# hotspot18 = 1 * (sig18 * li18.q==1)
# nn_in_n18 = 2 * (sig18 * li18.q==2)
# coldspot18 = 3 * (sig18 * li18.q==3)
# # n_in_nn18 = 4 * (sig18 * li18.q==4)
# spots18 = hotspot18 + nn_in_n18 + coldspot18# + n_in_nn18
# spot_labels18 = ['1 Random', '2 Hotspot', '3 Non-native in native', '4 Coldspot']#, '5 Native in non-native']
# labels18 = [spot_labels18[i] for i in spots18]
# hmap18 = colors.ListedColormap(['#c0c0c0', rust, gold, darkblue])#, lightblue])
#
# sig19 = 1 * (li19.p_sim < 0.05)
# hotspot19 = 1 * (sig19 * li19.q==1)
# nn_in_n19 = 2 * (sig19 * li19.q==2)
# coldspot19 = 3 * (sig19 * li19.q==3)
# n_in_nn19 = 4 * (sig19 * li19.q==4)
# spots19 = hotspot19 + nn_in_n19 + coldspot19 + n_in_nn19
# spot_labels19 = ['1 Random', '2 Hotspot', '3 Non-native in native', '4 Coldspot', '5 Native in non-native']
# labels19 = [spot_labels19[i] for i in spots19]
# hmap19 = colors.ListedColormap(['#c0c0c0', rust, gold, darkblue, lightblue])
#
# sig = 1 * (li.p_sim < 0.05)
# hotspot = 1 * (sig * li.q==1)
# nn_in_n = 2 * (sig * li.q==2)
# coldspot = 3 * (sig * li.q==3)
# n_in_nn = 4 * (sig * li.q==4)
# spots = hotspot + nn_in_n + coldspot + n_in_nn
# spot_labels = ['1 Random', '2 Hotspot', '3 Non-native in native', '4 Coldspot', '5 Native in non-native']
# labels = [spot_labels[i] for i in spots]
# hmap = colors.ListedColormap(['#c0c0c0', rust, gold, darkblue, lightblue])
#
# ##### Hotspot map 2017-2020 ###########################################################################################
# # plt.rc('legend', fontsize=7)
# # fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
# # SMMNRAboundary.plot(ax=ax[0,0], color='green', alpha=0.1)
# # hex_grid_nonnative17.assign(cl=labels17).plot(column='cl', categorical=True, cmap=hmap17, linewidth=0.1, ax=ax[0,0], edgecolor='white', legend=True)
# # WoolseyBoundary.plot(ax=ax[0,0], color='none', edgecolor='black', linewidth=0.5, alpha=0.4)
# # WoolseyBoundary.plot(ax=ax[0,0], color='#c54f19', edgecolor='black', linewidth=0.5, alpha=0.2)
# # # Freeway101.plot(ax=ax[0,0], color='blue', linewidth=1, alpha=0.4)
# # Freeway405.plot(ax=ax[0,0], color='blue', linewidth=1, alpha=0.4)
# # Coastline.plot(ax=ax[0,0], color='lightblue', linewidth=1, alpha=0.6)
# # ax[0,0].scatter(longitude17, latitude17, marker='.', s=2, color='black')
# # # ax[0,0].text(-119.05, 34.22, '101 FWY', rotation=-5, fontsize=6)
# # # ax[0,0].text(-117.55, 34.175, '101 FWY', rotation=-5, fontsize=6)
# # # ax[0,0].text(-117.477, 34.08, '405 FWY', rotation=-80, fontsize=6)
# # ax[0,0].text(-119.05, 34.285, 'a)', fontsize=ea_panelletter)
# # ax[0,0].tick_params(labelsize=ea_axisnum)
# # ax[0,0].set_xlim(-119.09, -118.43)
# # ax[0,0].set_ylim(33.99, 34.31)
# # ax[0,0].set_ylabel("Latitude (degrees)", fontsize=ea_axislabel)
# # # ax[0,0].set_xlabel("Longitude (degrees)", fontsize=ea_axislabel)
# #
# # SMMNRAboundary.plot(ax=ax[0,1], color='green', alpha=0.1)
# # hex_grid_nonnative18.assign(cl=labels18).plot(column='cl', categorical=True, cmap=hmap18, linewidth=0.1, ax=ax[0,1], edgecolor='white', legend=True)
# # WoolseyBoundary.plot(ax=ax[0,1], color='none', edgecolor='black', linewidth=0.5, alpha=0.4)
# # WoolseyBoundary.plot(ax=ax[0,1], color='#c54f19', edgecolor='black', linewidth=0.5, alpha=0.2)
# # Freeway101.plot(ax=ax[0,1], color='blue', linewidth=1, alpha=0.4)
# # Freeway405.plot(ax=ax[0,1], color='blue', linewidth=1, alpha=0.4)
# # Coastline.plot(ax=ax[0,1], color='lightblue', linewidth=1, alpha=0.6)
# # ax[0,1].scatter(longitude18, latitude18, marker='.', s=2, color='black')
# # # ax[0,1].text(-119.05, 34.22, '101 FWY', rotation=-5, fontsize=6)
# # # ax[0,1].text(-118.55, 34.175, '101 FWY', rotation=-5, fontsize=6)
# # # ax[0,1].text(-118.477, 34.08, '405 FWY', rotation=-80, fontsize=6)
# # ax[0,1].text(-119.05, 34.285, 'b)', fontsize=ea_panelletter)
# # ax[0,1].tick_params(labelsize=ea_axisnum)
# # ax[0,1].set_xlim(-119.09, -118.43)
# # ax[0,1].set_ylim(33.99, 34.31)
# # # ax[0,1].set_ylabel("Latitude (degrees)", fontsize=ea_axislabel)
# # # ax[0,1].set_xlabel("Longitude (degrees)", fontsize=ea_axislabel)
# #
# # SMMNRAboundary.plot(ax=ax[1,0], color='green', alpha=0.1)
# # hex_grid_nonnative19.assign(cl=labels19).plot(column='cl', categorical=True, cmap=hmap19, linewidth=0.1, ax=ax[1,0], edgecolor='white', legend=True)
# # WoolseyBoundary.plot(ax=ax[1,0], color='none', edgecolor='black', linewidth=0.5, alpha=0.4)
# # WoolseyBoundary.plot(ax=ax[1,0], color='#c54f19', edgecolor='black', linewidth=0.5, alpha=0.2)
# # Freeway101.plot(ax=ax[1,0], color='blue', linewidth=1, alpha=0.4)
# # Freeway405.plot(ax=ax[1,0], color='blue', linewidth=1, alpha=0.4)
# # Coastline.plot(ax=ax[1,0], color='lightblue', linewidth=1, alpha=0.6)
# # ax[1,0].scatter(longitude19, latitude19, marker='.', s=2, color='black')
# # # ax[1,0].text(-119.05, 34.22, '101 FWY', rotation=-5, fontsize=6)
# # # ax[1,0].text(-119.55, 34.175, '101 FWY', rotation=-5, fontsize=6)
# # # ax[1,0].text(-119.477, 34.08, '405 FWY', rotation=-80, fontsize=6)
# # ax[1,0].text(-119.05, 34.285, 'c)', fontsize=ea_panelletter)
# # ax[1,0].tick_params(labelsize=ea_axisnum)
# # ax[1,0].set_xlim(-119.09, -118.43)
# # ax[1,0].set_ylim(33.99, 34.31)
# # ax[1,0].set_ylabel("Latitude (degrees)", fontsize=ea_axislabel)
# # ax[1,0].set_xlabel("Longitude (degrees)", fontsize=ea_axislabel)
# #
# # SMMNRAboundary.plot(ax=ax[1,1], color='green', alpha=0.1)
# # hex_grid_nonnative20.assign(cl=labels).plot(column='cl', categorical=True, cmap=hmap, linewidth=0.1, ax=ax[1,1], edgecolor='white', legend=True)
# # WoolseyBoundary.plot(ax=ax[1,1], color='none', edgecolor='black', linewidth=0.5, alpha=0.4)
# # WoolseyBoundary.plot(ax=ax[1,1], color='#c54f19', edgecolor='black', linewidth=0.5, alpha=0.2)
# # Freeway101.plot(ax=ax[1,1], color='blue', linewidth=1, alpha=0.4)
# # Freeway405.plot(ax=ax[1,1], color='blue', linewidth=1, alpha=0.4)
# # Coastline.plot(ax=ax[1,1], color='lightblue', linewidth=1, alpha=0.6)
# # ax[1,1].scatter(longitude20, latitude20, marker='.', s=2, color='black')
# # # ax[1,1].text(-119.05, 34.22, '101 FWY', rotation=-5, fontsize=6)
# # # ax[1,1].text(-118.55, 34.175, '101 FWY', rotation=-5, fontsize=6)
# # # ax[1,1].text(-118.477, 34.08, '405 FWY', rotation=-80, fontsize=6)
# # ax[1,1].text(-119.05, 34.285, 'd)',  fontsize=ea_panelletter)
# # ax[1,1].tick_params(labelsize=ea_axisnum)
# # ax[1,1].set_xlim(-119.09, -118.43)
# # ax[1,1].set_ylim(33.99, 34.31)
# # # ax[1,1].set_ylabel("Latitude (degrees)", fontsize=ea_axislabel)
# # ax[1,1].set_xlabel("Longitude (degrees)", fontsize=ea_axislabel)
# #
# # plt.tight_layout()
# # plt.savefig("/Users/nicolettastork/Documents/Geography/SMMNRA_Invasive_Species/NPSData/VegData_FromSMMNRA_Modified/Fig_HotspotMap_17to20.png", dpi=600)
# # # plt.show()
# #######################################################################################################################
#
# ##### Hotspot map 2020 only ###########################################################################################
# plt.rc('legend', fontsize=7)
# fig, ax = plt.subplots(1, figsize=(6, 4), sharex=True, sharey=True)
# SMMNRAboundary.plot(ax=ax, color='green', alpha=0.1)
# hex_grid_nonnative20.assign(cl=labels).plot(column='cl', categorical=True, cmap=hmap, linewidth=0.1, ax=ax, edgecolor='white', legend=True)
# WoolseyBoundary.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5, alpha=0.4)
# WoolseyBoundary.plot(ax=ax, color='#c54f19', edgecolor='black', linewidth=0.5, alpha=0.2)
# Freeway101.plot(ax=ax, color='blue', linewidth=1, alpha=0.4)
# Freeway405.plot(ax=ax, color='blue', linewidth=1, alpha=0.4)
# Coastline.plot(ax=ax, color='lightblue', linewidth=1, alpha=0.6)
# ax.scatter(longitude20, latitude20, marker='.', s=2, color='black')
# # ax.text(-119.05, 34.22, '101 FWY', rotation=-5, fontsize=6)
# # ax.text(-118.55, 34.175, '101 FWY', rotation=-5, fontsize=6)
# # ax.text(-118.477, 34.08, '405 FWY', rotation=-80, fontsize=6)
# ax.tick_params(labelsize=ea_axisnum)
# ax.set_xlim(-119.09, -118.43)
# ax.set_ylim(33.99, 34.31)
# ax.set_ylabel("Latitude (degrees)", fontsize=ea_axislabel)
# ax.set_xlabel("Longitude (degrees)", fontsize=ea_axislabel)
# plt.tight_layout()
# plt.savefig("/Users/nicolettastork/Documents/Geography/SMMNRA_Invasive_Species/NPSData/VegData_FromSMMNRA_Modified/Fig_HotspotMap_20.png", dpi=600)
# # plt.show()
# ######################################################################################################################
#
# ### Add a column 'hotspot' and signify the hotspot cells with a 1.
# hex_grid_nonnative20 = gpd.GeoDataFrame({"counts": nncounts, 'geometry': hex_array, "hot": hotspot}, crs="EPSG:2955")
# ### Make a new geodataframe with only the hotspot cells.
# nnhotspot_poly = gpd.GeoDataFrame(hex_grid_nonnative20[hex_grid_nonnative20['hot'] == 1], crs="EPSG:2955")
#
# nnhotspot_poly.to_file('/Users/nicolettastork/Documents/Geography/SMMNRA_Invasive_Species/GIS_KMZ/NonnativeHotspot2020.shp', crs="EPSG:2955")
#
###########################################################################
##### FIGURE 6 & 7: Rarefaction curve in hotspot & Top species in hotspot
###########################################################################
# index2017 = 0
# index2018 = 0
# index2019 = 0
# index2020 = 0
#
# sitecodes_inHotspot = numpy.array([])
#
# sitecodes17 = numpy.array([])
# sitecodes18 = numpy.array([])
# sitecodes19 = numpy.array([])
# sitecodes20 = numpy.array([])
# sitecodes_nat17 = numpy.array([])
# sitecodes_nat18 = numpy.array([])
# sitecodes_nat19 = numpy.array([])
# sitecodes_nat20 = numpy.array([])
# sitecodes_nnat17 = numpy.array([])
# sitecodes_nnat18 = numpy.array([])
# sitecodes_nnat19 = numpy.array([])
# sitecodes_nnat20 = numpy.array([])
# sitecodes_inW_17 = numpy.array([])
# sitecodes_inW_18 = numpy.array([])
# sitecodes_inW_19 = numpy.array([])
# sitecodes_inW_20 = numpy.array([])
# sitecodes_outW_17 = numpy.array([])
# sitecodes_outW_18 = numpy.array([])
# sitecodes_outW_19 = numpy.array([])
# sitecodes_outW_20 = numpy.array([])
#
# nspecies = numpy.array([])
# nnspecies = numpy.array([])
# nspecies2017 = numpy.array([])
# nnspecies2017 = numpy.array([])
# nspecies2018 = numpy.array([])
# nnspecies2018 = numpy.array([])
# nspecies2019 = numpy.array([])
# nnspecies2019 = numpy.array([])
# nspecies2020 = numpy.array([])
# nnspecies2020 = numpy.array([])
#
# speciestotals17 = numpy.array([])
# speciestotals18 = numpy.array([])
# speciestotals19 = numpy.array([])
# speciestotals20 = numpy.array([])
#
# n_specieswithinhotspot = numpy.array([])
# nn_specieswithinhotspot = numpy.array([])
# n_specieswithinhotspot2017 = numpy.array([])
# nn_specieswithinhotspot2017 = numpy.array([])
# n_specieswithinhotspot2018 = numpy.array([])
# nn_specieswithinhotspot2018 = numpy.array([])
# n_specieswithinhotspot2019 = numpy.array([])
# nn_specieswithinhotspot2019 = numpy.array([])
# n_specieswithinhotspot2020 = numpy.array([])
# nn_specieswithinhotspot2020 = numpy.array([])
#
# n_fxngroup17 = numpy.array([])
# n_fxngroup18 = numpy.array([])
# n_fxngroup19 = numpy.array([])
# n_fxngroup20 = numpy.array([])
#
# nativemean17 = numpy.array([])
# nativemean18 = numpy.array([])
# nativemean19 = numpy.array([])
# nativemean20 = numpy.array([])
# nonnativemean17 = numpy.array([])
# nonnativemean18 = numpy.array([])
# nonnativemean19 = numpy.array([])
# nonnativemean20 = numpy.array([])
#
#
#
# for name, obj in monitoringpoints.items():
#     point = Point(obj.decimalLongitude, obj.decimalLatitude) # for each key and value in monitoring points
#     for index, row in nnhotspot_poly.iterrows():
#         if point.within(row['geometry']) == True:
#             sitecodes_inHotspot = numpy.append(sitecodes_inHotspot, numpy.unique(obj.sitecode))
#             ### 2017 ###
#             test_indices17 = numpy.where(obj.year == 2017)
#             test_indices_inW17 = numpy.where((obj.year == 2017) & (obj.inWoolsey == True))
#             test_indices_outW17 = numpy.where((obj.year == 2017) & (obj.inWoolsey == False))
#             test_indices_nat17 = numpy.where((obj.year == 2017) & (obj.native_status == 'Native'))
#             test_indices_nnat17 = numpy.where((obj.year == 2017) & (obj.native_status == 'Nonnative'))
#             index2017 += len(test_indices17[0])
#             speciestotals17 = numpy.append(speciestotals17, obj.species_code[test_indices17])
#             sitecodes17 = numpy.append(sitecodes17, obj.sitecode[test_indices17])
#             sitecodes_nat17 = numpy.append(sitecodes_nat17, obj.sitecode[test_indices_nat17])
#             sitecodes_nnat17 = numpy.append(sitecodes_nnat17, obj.sitecode[test_indices_nnat17])
#             sitecodes_inW_17 = numpy.append(sitecodes_inW_17, obj.sitecode[test_indices_inW17])
#             sitecodes_outW_17 = numpy.append(sitecodes_outW_17, obj.sitecode[test_indices_outW17])
#             nspecies2017 = numpy.append(nspecies2017, obj.species_code[test_indices_nat17])
#             nnspecies2017 = numpy.append(nnspecies2017, obj.species_code[test_indices_nnat17])
#             n_specieswithinhotspot2017 = [x for x in nspecies2017 if x != ""]
#             nn_specieswithinhotspot2017 = [x for x in nnspecies2017 if x != ""]
#             n_fxngroup17 = numpy.append(n_fxngroup17, obj.fxngroup[test_indices_nat17])
#             nspfxn17 = {n_specieswithinhotspot2017[i]: n_fxngroup17[i] for i in range(len(n_specieswithinhotspot2017))}
#
#             label_totals17, count_totals17 = numpy.unique(sitecodes17, return_counts=True)
#             totals17 = {label_totals17[i]: count_totals17[i] for i in range(len(label_totals17))}
#             label_nat17, count_nat17 = numpy.unique(sitecodes_nat17, return_counts=True)
#             natives17 = {label_nat17[i]: count_nat17[i] for i in range(len(label_nat17))}
#             label_nnat17, count_nnat17 = numpy.unique(sitecodes_nnat17, return_counts=True)
#             nonnatives17 = {label_nnat17[i]: count_nnat17[i] for i in range(len(label_nnat17))}
#             fulldict17 = {}
#             for key in set(list(totals17.keys()) + list(natives17) + list(nonnatives17)):
#                 try:
#                     fulldict17.setdefault(key, []).append(totals17[key])
#                 except KeyError:
#                     fulldict17.setdefault(key, []).append(0)
#                 try:
#                     fulldict17.setdefault(key, []).append(natives17[key])
#                 except KeyError:
#                     fulldict17.setdefault(key, []).append(0)
#                 try:
#                     fulldict17.setdefault(key, []).append(nonnatives17[key])
#                 except KeyError:
#                     fulldict17.setdefault(key, []).append(0)
#             values17 = numpy.array(list(fulldict17.values()))
#             valuestotals17 = [sublist[0] for sublist in values17]
#             valuesnatives17 = [sublist[1] for sublist in values17]
#             valuesnonnatives17 = [sublist[-1] for sublist in values17]
#             nptotals17 = numpy.asarray(valuestotals17)
#             npnatives17 = numpy.asarray(valuesnatives17)
#             npnonnatives17 = numpy.asarray(valuesnonnatives17)
#             nativepercentage17 = numpy.divide(npnatives17, nptotals17)
#             nonnativepercentage17 = numpy.divide(npnonnatives17, nptotals17)
#             nativemean17 = numpy.around(numpy.mean(nativepercentage17) * 100)
#             nativesd17 = numpy.around(numpy.std(nativepercentage17) * 100, 2)
#             nativesem17 = numpy.around(sem(nativepercentage17) * 100, 2)
#             nonnativemean17 = numpy.around(numpy.mean(nonnativepercentage17) * 100)
#             nonnativesd17 = numpy.around(numpy.std(nonnativepercentage17) * 100, 2)
#             nonnativesem17 = numpy.around(sem(nonnativepercentage17) * 100, 2)
#
#             ### 2018 ###
#             test_indices18 = numpy.where(obj.year == 2018)
#             test_indices_inW18 = numpy.where((obj.year == 2018) & (obj.inWoolsey == True))
#             test_indices_outW18 = numpy.where((obj.year == 2018) & (obj.inWoolsey == False))
#             test_indices_nat18 = numpy.where((obj.year == 2018) & (obj.native_status == 'Native'))
#             test_indices_nnat18 = numpy.where((obj.year == 2018) & (obj.native_status == 'Nonnative'))
#             index2018 += len(test_indices18[0])
#             speciestotals18 = numpy.append(speciestotals18, obj.species_code[test_indices18])
#             sitecodes18 = numpy.append(sitecodes18, obj.sitecode[test_indices18])
#             sitecodes_nat18 = numpy.append(sitecodes_nat18, obj.sitecode[test_indices_nat18])
#             sitecodes_nnat18 = numpy.append(sitecodes_nnat18, obj.sitecode[test_indices_nnat18])
#             sitecodes_inW_18 = numpy.append(sitecodes_inW_18, obj.sitecode[test_indices_inW18])
#             sitecodes_outW_18 = numpy.append(sitecodes_outW_18, obj.sitecode[test_indices_outW18])
#             nspecies2018 = numpy.append(nspecies2018, obj.species_code[test_indices_nat18])
#             nnspecies2018 = numpy.append(nnspecies2018, obj.species_code[test_indices_nnat18])
#             n_specieswithinhotspot2018 = [x for x in nspecies2018 if x != ""]
#             nn_specieswithinhotspot2018 = [x for x in nnspecies2018 if x != ""]
#             n_fxngroup18 = numpy.append(n_fxngroup18, obj.fxngroup[test_indices_nat18])
#             nspfxn18 = {n_specieswithinhotspot2018[i]: n_fxngroup18[i] for i in range(len(n_specieswithinhotspot2018))}
#
#             label_totals18, count_totals18 = numpy.unique(sitecodes18, return_counts=True)
#             totals18 = {label_totals18[i]: count_totals18[i] for i in range(len(label_totals18))}
#             label_nat18, count_nat18 = numpy.unique(sitecodes_nat18, return_counts=True)
#             natives18 = {label_nat18[i]: count_nat18[i] for i in range(len(label_nat18))}
#             label_nnat18, count_nnat18 = numpy.unique(sitecodes_nnat18, return_counts=True)
#             nonnatives18 = {label_nnat18[i]: count_nnat18[i] for i in range(len(label_nnat18))}
#             fulldict18 = {}
#             for key in set(list(totals18.keys()) + list(natives18) + list(nonnatives18)):
#                 try:
#                     fulldict18.setdefault(key, []).append(totals18[key])
#                 except KeyError:
#                     fulldict18.setdefault(key, []).append(0)
#                 try:
#                     fulldict18.setdefault(key, []).append(natives18[key])
#                 except KeyError:
#                     fulldict18.setdefault(key, []).append(0)
#                 try:
#                     fulldict18.setdefault(key, []).append(nonnatives18[key])
#                 except KeyError:
#                     fulldict18.setdefault(key, []).append(0)
#             values18 = numpy.array(list(fulldict18.values()))
#             valuestotals18 = [sublist[0] for sublist in values18]
#             valuesnatives18 = [sublist[1] for sublist in values18]
#             valuesnonnatives18 = [sublist[-1] for sublist in values18]
#             nptotals18 = numpy.asarray(valuestotals18)
#             npnatives18 = numpy.asarray(valuesnatives18)
#             npnonnatives18 = numpy.asarray(valuesnonnatives18)
#             nativepercentage18 = numpy.divide(npnatives18, nptotals18)
#             nonnativepercentage18 = numpy.divide(npnonnatives18, nptotals18)
#             nativemean18 = numpy.around(numpy.mean(nativepercentage18) * 100)
#             nativesd18 = numpy.around(numpy.std(nativepercentage18) * 100, 2)
#             nativesem18 = numpy.around(sem(nativepercentage18) * 100, 2)
#             nonnativemean18 = numpy.around(numpy.mean(nonnativepercentage18) * 100)
#             nonnativesd18 = numpy.around(numpy.std(nonnativepercentage18) * 100, 2)
#             nonnativesem18 = numpy.around(sem(nonnativepercentage18) * 100, 2)
#
#             ### 2019 ###
#             test_indices19 = numpy.where(obj.year == 2019)
#             test_indices_inW19 = numpy.where((obj.year == 2019) & (obj.inWoolsey == True))
#             test_indices_outW19 = numpy.where((obj.year == 2019) & (obj.inWoolsey == False))
#             test_indices_nat19 = numpy.where((obj.year == 2019) & (obj.native_status == 'Native'))
#             test_indices_nnat19 = numpy.where((obj.year == 2019) & (obj.native_status == 'Nonnative'))
#             index2019 += len(test_indices19[0])
#             speciestotals19 = numpy.append(speciestotals19, obj.species_code[test_indices19])
#             sitecodes19 = numpy.append(sitecodes19, obj.sitecode[test_indices19])
#             sitecodes_nat19 = numpy.append(sitecodes_nat19, obj.sitecode[test_indices_nat19])
#             sitecodes_nnat19 = numpy.append(sitecodes_nnat19, obj.sitecode[test_indices_nnat19])
#             sitecodes_inW_19 = numpy.append(sitecodes_inW_19, obj.sitecode[test_indices_inW19])
#             sitecodes_outW_19 = numpy.append(sitecodes_outW_19, obj.sitecode[test_indices_outW19])
#             nspecies2019 = numpy.append(nspecies2019, obj.species_code[test_indices_nat19])
#             nnspecies2019 = numpy.append(nnspecies2019, obj.species_code[test_indices_nnat19])
#             n_specieswithinhotspot2019 = [x for x in nspecies2019 if x != ""]
#             nn_specieswithinhotspot2019 = [x for x in nnspecies2019 if x != ""]
#             n_fxngroup19 = numpy.append(n_fxngroup19, obj.fxngroup[test_indices_nat19])
#             nspfxn19 = {n_specieswithinhotspot2019[i]: n_fxngroup19[i] for i in range(len(n_specieswithinhotspot2019))}
#
#             label_totals19, count_totals19 = numpy.unique(sitecodes19, return_counts=True)
#             totals19 = {label_totals19[i]: count_totals19[i] for i in range(len(label_totals19))}
#             label_nat19, count_nat19 = numpy.unique(sitecodes_nat19, return_counts=True)
#             natives19 = {label_nat19[i]: count_nat19[i] for i in range(len(label_nat19))}
#             label_nnat19, count_nnat19 = numpy.unique(sitecodes_nnat19, return_counts=True)
#             nonnatives19 = {label_nnat19[i]: count_nnat19[i] for i in range(len(label_nnat19))}
#             fulldict19 = {}
#             for key in set(list(totals19.keys()) + list(natives19) + list(nonnatives19)):
#                 try:
#                     fulldict19.setdefault(key, []).append(totals19[key])
#                 except KeyError:
#                     fulldict19.setdefault(key, []).append(0)
#                 try:
#                     fulldict19.setdefault(key, []).append(natives19[key])
#                 except KeyError:
#                     fulldict19.setdefault(key, []).append(0)
#                 try:
#                     fulldict19.setdefault(key, []).append(nonnatives19[key])
#                 except KeyError:
#                     fulldict19.setdefault(key, []).append(0)
#             values19 = numpy.array(list(fulldict19.values()))
#             valuestotals19 = [sublist[0] for sublist in values19]
#             valuesnatives19 = [sublist[1] for sublist in values19]
#             valuesnonnatives19 = [sublist[-1] for sublist in values19]
#             nptotals19 = numpy.asarray(valuestotals19)
#             npnatives19 = numpy.asarray(valuesnatives19)
#             npnonnatives19 = numpy.asarray(valuesnonnatives19)
#             nativepercentage19 = numpy.divide(npnatives19, nptotals19)
#             nonnativepercentage19 = numpy.divide(npnonnatives19, nptotals19)
#             nativemean19 = numpy.around(numpy.mean(nativepercentage19) * 100)
#             nativesd19 = numpy.around(numpy.std(nativepercentage19) * 100, 2)
#             nativesem19 = numpy.around(sem(nativepercentage19) * 100, 2)
#             nonnativemean19 = numpy.around(numpy.mean(nonnativepercentage19) * 100)
#             nonnativesd19 = numpy.around(numpy.std(nonnativepercentage19) * 100, 2)
#             nonnativesem19 = numpy.around(sem(nonnativepercentage19) * 100, 2)
#
#             ### 2020 ###
#             test_indices20 = numpy.where(obj.year == 2020)
#             test_indices_inW20 = numpy.where((obj.year == 2020) & (obj.inWoolsey == True))
#             test_indices_outW20 = numpy.where((obj.year == 2020) & (obj.inWoolsey == False))
#             test_indices_nat20 = numpy.where((obj.year == 2020) & (obj.native_status == 'Native'))
#             test_indices_nnat20 = numpy.where((obj.year == 2020) & (obj.native_status == 'Nonnative'))
#             index2020 += len(test_indices20[0])
#             speciestotals20 = numpy.append(speciestotals20, obj.species_code[test_indices20])
#             sitecodes20 = numpy.append(sitecodes20, obj.sitecode[test_indices20])
#             sitecodes_nat20 = numpy.append(sitecodes_nat20, obj.sitecode[test_indices_nat20])
#             sitecodes_nnat20 = numpy.append(sitecodes_nnat20, obj.sitecode[test_indices_nnat20])
#             sitecodes_inW_20 = numpy.append(sitecodes_inW_20, obj.sitecode[test_indices_inW20])
#             sitecodes_outW_20 = numpy.append(sitecodes_outW_20, obj.sitecode[test_indices_outW20])
#             nspecies2020 = numpy.append(nspecies2020, obj.species_code[test_indices_nat20])
#             nnspecies2020 = numpy.append(nnspecies2020, obj.species_code[test_indices_nnat20])
#             n_specieswithinhotspot2020 = [x for x in nspecies2020 if x != ""]
#             nn_specieswithinhotspot2020 = [x for x in nnspecies2020 if x != ""]
#             n_fxngroup20 = numpy.append(n_fxngroup20, obj.fxngroup[test_indices_nat20])
#             nspfxn20 = {n_specieswithinhotspot2020[i]: n_fxngroup20[i] for i in range(len(n_specieswithinhotspot2020))}
#
#             label_totals20, count_totals20 = numpy.unique(sitecodes20, return_counts=True)
#             totals20 = {label_totals20[i]: count_totals20[i] for i in range(len(label_totals20))}
#             label_nat20, count_nat20 = numpy.unique(sitecodes_nat20, return_counts=True)
#             natives20 = {label_nat20[i]: count_nat20[i] for i in range(len(label_nat20))}
#             label_nnat20, count_nnat20 = numpy.unique(sitecodes_nnat20, return_counts=True)
#             nonnatives20 = {label_nnat20[i]: count_nnat20[i] for i in range(len(label_nnat20))}
#             fulldict20 = {}
#             for key in set(list(totals20.keys()) + list(natives20) + list(nonnatives20)):
#                 try:
#                     fulldict20.setdefault(key, []).append(totals20[key])
#                 except KeyError:
#                     fulldict20.setdefault(key, []).append(0)
#                 try:
#                     fulldict20.setdefault(key, []).append(natives20[key])
#                 except KeyError:
#                     fulldict20.setdefault(key, []).append(0)
#                 try:
#                     fulldict20.setdefault(key, []).append(nonnatives20[key])
#                 except KeyError:
#                     fulldict20.setdefault(key, []).append(0)
#             values20 = numpy.array(list(fulldict20.values()))
#             valuestotals20 = [sublist[0] for sublist in values20]
#             valuesnatives20 = [sublist[1] for sublist in values20]
#             valuesnonnatives20 = [sublist[-1] for sublist in values20]
#             nptotals20 = numpy.asarray(valuestotals20)
#             npnatives20 = numpy.asarray(valuesnatives20)
#             npnonnatives20 = numpy.asarray(valuesnonnatives20)
#             nativepercentage20 = numpy.divide(npnatives20, nptotals20)
#             nonnativepercentage20 = numpy.divide(npnonnatives20, nptotals20)
#             nativemean20 = numpy.around(numpy.mean(nativepercentage20) * 100)
#             nativesd20 = numpy.around(numpy.std(nativepercentage20) * 100, 2)
#             nativesem20 = numpy.around(sem(nativepercentage20) * 100, 2)
#             nonnativemean20 = numpy.around(numpy.mean(nonnativepercentage20) * 100)
#             nonnativesd20 = numpy.around(numpy.std(nonnativepercentage20) * 100, 2)
#             nonnativesem20 = numpy.around(sem(nonnativepercentage20) * 100, 2)
#
#             ### All ###
#             n_index = numpy.where((obj.native_status == 'Native'))
#             nn_index = numpy.where((obj.native_status == 'Nonnative'))
#             nspecies = numpy.append(nspecies, obj.species_code[n_index])
#             nnspecies = numpy.append(nnspecies, obj.species_code[nn_index])
#             n_specieswithinhotspot = [x for x in nspecies if x !=""]
#             nn_specieswithinhotspot = [x for x in nnspecies if x !=""]
#             numpy.printoptions(numpy.inf)
#
#
# hotspot_monitoringsite_table = [['Year', 2017, 2018, 2019, 2020],
#                                 ['Hotspot', len(numpy.unique(sitecodes17)),
#                                  len(numpy.unique(sitecodes18)),
#                                  len(numpy.unique(sitecodes19)),
#                                  len(numpy.unique(sitecodes20))],
#                                 ['In Woolsey', len(numpy.unique(sitecodes_inW_17)),
#                                  len(numpy.unique(sitecodes_inW_18)),
#                                  len(numpy.unique(sitecodes_inW_19)),
#                                  len(numpy.unique(sitecodes_inW_20))],
#                                 ['Out Woolsey', len(numpy.unique(sitecodes_outW_17)),
#                                  len(numpy.unique(sitecodes_outW_18)),
#                                  len(numpy.unique(sitecodes_outW_19)),
#                                  len(numpy.unique(sitecodes_outW_20))],
#                                 ['Native cover in hotspot (%)', nativemean17,
#                                  nativemean18, nativemean19, nativemean20],
#                                 ['Nonnative cover in hotspot (%)', nonnativemean17, nonnativemean18,
#                                  nonnativemean19, nonnativemean20]]
#
# print(tabulate(hotspot_monitoringsite_table))
#
#
# hotspot_monitoringsite_latex = [['Year', 2017, 2018, 2019, 2020],
#                                 ['Monitoring sites surveyed within hotspot',
#                                  len(numpy.unique(sitecodes17)),
#                                  len(numpy.unique(sitecodes18)),
#                                  len(numpy.unique(sitecodes19)),
#                                  len(numpy.unique(sitecodes20))],
#                                 ['Monitoring sites surveyed within Woolsey Fire boundary in the hotspot',
#                                  len(numpy.unique(sitecodes_inW_17)),
#                                  len(numpy.unique(sitecodes_inW_18)),
#                                  len(numpy.unique(sitecodes_inW_19)),
#                                  len(numpy.unique(sitecodes_inW_20))],
#                                 ['Monitoring sites surveyed outside Woolsey Fire boundary in the hotspot',
#                                  len(numpy.unique(sitecodes_outW_17)),
#                                  len(numpy.unique(sitecodes_outW_18)),
#                                  len(numpy.unique(sitecodes_outW_19)),
#                                  len(numpy.unique(sitecodes_outW_20))],
#                                 ['Native cover in hotspot (\%)',
#                                  nativemean17, nativemean18, nativemean19, nativemean20],
#                                 ['Nonnative cover in hotspot (\%)',
#                                  nonnativemean17, nonnativemean18, nonnativemean19, nonnativemean20]]
#
# #### Print for latex #####
# print('\\begin{tabular}{lllllll} \\toprule')
# for row in hotspot_monitoringsite_latex:
#     for item in row:
#         y = ' & '.join([str(item) for item in row])
#     print(y + ' \\''\\')
# print('\\end{tabular}')
# ##########################

###########################################################################
# ##### FIGURE 6: Rarefaction Curve: Hotspot #####
###########################################################################
#
# ## Figure 6 numbers ###
# # table_htsptrarefaction = [['years', 2017, 2018, 2019, 2020],
# #                      ['Sitecodes within the hotspot',
# #                       len(numpy.unique(sitecodes17)),
# #                       len(numpy.unique(sitecodes18)),
# #                       len(numpy.unique(sitecodes19)),
# #                       len(numpy.unique(sitecodes20))],
# #                      ['Sitecodes within the Woolsey boundary, in hotspot',
# #                       len(numpy.unique(sitecodes_inW_17)),
# #                       len(numpy.unique(sitecodes_inW_18)),
# #                       len(numpy.unique(sitecodes_inW_19)),
# #                       len(numpy.unique(sitecodes_inW_20))],
# #                      ['Total observations in hotspot',
# #                       index2017,
# #                       index2018,
# #                       index2019,
# #                       index2020],
# #                      ['Native species in hotspot',
# #                       len(numpy.unique(n_specieswithinhotspot2017)),
# #                       len(numpy.unique(n_specieswithinhotspot2018)),
# #                       len(numpy.unique(n_specieswithinhotspot2019)),
# #                       len(numpy.unique(n_specieswithinhotspot2020))],
# #                      ['Native cover percentage in hotspot',
# #                       numpy.around((len(n_specieswithinhotspot2017)/index2017)*100),
# #                       numpy.around((len(n_specieswithinhotspot2018)/index2018)*100),
# #                       numpy.around((len(n_specieswithinhotspot2019)/index2019)*100),
# #                       numpy.around((len(n_specieswithinhotspot2020)/index2020)*100)],
# #                      ['Nonnative species in hotspot',
# #                       len(numpy.unique(nn_specieswithinhotspot2017)),
# #                       len(numpy.unique(nn_specieswithinhotspot2018)),
# #                       len(numpy.unique(nn_specieswithinhotspot2019)),
# #                       len(numpy.unique(nn_specieswithinhotspot2020))],
# #                      ['Nonnative cover percentage in hotspot',
# #                       numpy.around((len(nn_specieswithinhotspot2017) / index2017) * 100),
# #                       numpy.around((len(nn_specieswithinhotspot2018) / index2018) * 100),
# #                       numpy.around((len(nn_specieswithinhotspot2019) / index2019) * 100),
# #                       numpy.around((len(nn_specieswithinhotspot2020) / index2020) * 100)]]
# #
# # print(tabulate(table_htsptrarefaction))
#
# # table_htsptrarefaction_latex = [['Year', 2017, 2018, 2019, 2020],
# #                           ['Counted native species',
# #                           len(numpy.unique(n_specieswithinhotspot2017)),
# #                           len(numpy.unique(n_specieswithinhotspot2018)),
# #                           len(numpy.unique(n_specieswithinhotspot2019)),
# #                           len(numpy.unique(n_specieswithinhotspot2020))],
# #                           ['Estimated native species', 31, 26.3, 73.9, 71.8],
# #                           ['Lower 95\% confidence of native species', 31, 26.3, 73.9, 71.8],
# #                           ['Upper 95\% confidence of native species', 36.5, 31, 75.4, 73.6],
# #                           ['Counted non-native species',
# #                           len(numpy.unique(nn_specieswithinhotspot2017)),
# #                           len(numpy.unique(nn_specieswithinhotspot2018)),
# #                           len(numpy.unique(nn_specieswithinhotspot2019)),
# #                           len(numpy.unique(nn_specieswithinhotspot2020))],
# #                           ['Estimated non-native species', 16.5, 17.9, 29.5, 25.8],
# #                           ['Lower 95\% confidence of non-native species', 13.5, 14.8, 28.9, 25.1],
# #                           ['Upper 95\% confidence of non-native species', 19.5, 21.1, 30.1, 26.4],
# #                           ]
# #
# # print('\\begin{tabular}{lllll} \\toprule')
# # for row in table_htsptrarefaction_latex:
# #     for item in row:
# #         y = ' & '.join([str(item) for item in row])
# #     print(y + ' \\''\\')
# # print('\\end{tabular}')
#
# def rarefactionHotspot(native17, native18, native19, native20, nonnative17, nonnative18, nonnative19, nonnative20, species17, species18, species19, species20):
#     trials = 10
#     nat_species17 = numpy.empty((0, trials), int)
#     nat_samples17 = numpy.arange(0, len(species17), 100)
#     for trial in range(0, trials):
#         for n in nat_samples17:
#             nat_index17 = numpy.random.choice(native17, n)
#             nat_species17 = numpy.append([nat_species17], [len(numpy.unique(nat_index17))])
#
#     nat_species18 = numpy.empty((0, trials), int)
#     nat_samples18 = numpy.arange(0, len(species18), 100)
#     for trial in range(0, trials):
#         for n in nat_samples18:
#             nat_index18 = numpy.random.choice(native18, n)
#             nat_species18 = numpy.append([nat_species18], [len(numpy.unique(nat_index18))])
#
#     nat_species19 = numpy.empty((0, trials), int)
#     nat_samples19 = numpy.arange(0, len(species19), 100)
#     for trial in range(0, trials):
#         for n in nat_samples19:
#             nat_index19 = numpy.random.choice(native19, n)
#             nat_species19 = numpy.append([nat_species19], [len(numpy.unique(nat_index19))])
#
#     nat_species20 = numpy.empty((0, trials), int)
#     nat_samples20 = numpy.arange(0, len(species20), 100)
#     for trial in range(0, trials):
#         for n in nat_samples20:
#             nat_index20 = numpy.random.choice(native20, n)
#             nat_species20 = numpy.append([nat_species20], [len(numpy.unique(nat_index20))])
#
#     nnat_species17 = numpy.empty((0, trials), int)
#     nnat_samples17 = numpy.arange(0, len(species17), 100)
#     for trial in range(0, trials):
#         for n in nnat_samples17:
#             nnat_index17 = numpy.random.choice(nonnative17, n)
#             nnat_species17 = numpy.append([nnat_species17], [len(numpy.unique(nnat_index17))])
#
#     nnat_species18 = numpy.empty((0, trials), int)
#     nnat_samples18 = numpy.arange(0, len(species18), 100)
#     for trial in range(0, trials):
#         for n in nnat_samples18:
#             nnat_index18 = numpy.random.choice(nonnative18, n)
#             nnat_species18 = numpy.append([nnat_species18], [len(numpy.unique(nnat_index18))])
#
#     nnat_species19 = numpy.empty((0, trials), int)
#     nnat_samples19 = numpy.arange(0, len(species19), 100)
#     for trial in range(0, trials):
#         for n in nnat_samples19:
#             nnat_index19 = numpy.random.choice(nonnative19, n)
#             nnat_species19 = numpy.append([nnat_species19], [len(numpy.unique(nnat_index19))])
#
#     nnat_species20 = numpy.empty((0, trials), int)
#     nnat_samples20 = numpy.arange(0, len(species20), 100)
#     for trial in range(0, trials):
#         for n in nnat_samples20:
#             nnat_index20 = numpy.random.choice(nonnative20, n)
#             nnat_species20 = numpy.append([nnat_species20], [len(numpy.unique(nnat_index20))])
#
#     nat_array17 = numpy.split(nat_species17, trials)
#     nat_array18 = numpy.split(nat_species18, trials)
#     nat_array19 = numpy.split(nat_species19, trials)
#     nat_array20 = numpy.split(nat_species20, trials)
#     nnat_array17 = numpy.split(nnat_species17, trials)
#     nnat_array18 = numpy.split(nnat_species18, trials)
#     nnat_array19 = numpy.split(nnat_species19, trials)
#     nnat_array20 = numpy.split(nnat_species20, trials)
#
#     # numpyarray = numpy.array(list(zip(*newarray)))
#     nat_mean17 = numpy.array([sum(x)/len(x) for x in zip(*nat_array17)])
#     nat_stdeviation17 = numpy.std(nat_array17, axis=0)
#     nat_mean18 = numpy.array([sum(x)/len(x) for x in zip(*nat_array18)])
#     nat_stdeviation18 = numpy.std(nat_array18, axis=0)
#     nat_mean19 = numpy.array([sum(x)/len(x) for x in zip(*nat_array19)])
#     nat_stdeviation19 = numpy.std(nat_array19, axis=0)
#     nat_mean20 = numpy.array([sum(x)/len(x) for x in zip(*nat_array20)])
#     nat_stdeviation20 = numpy.std(nat_array20, axis=0)
#
#     nnat_mean17 = numpy.array([sum(x)/len(x) for x in zip(*nnat_array17)])
#     nnat_stdeviation17 = numpy.std(nnat_array17, axis=0)
#     nnat_mean18 = numpy.array([sum(x)/len(x) for x in zip(*nnat_array18)])
#     nnat_stdeviation18 = numpy.std(nnat_array18, axis=0)
#     nnat_mean19 = numpy.array([sum(x)/len(x) for x in zip(*nnat_array19)])
#     nnat_stdeviation19 = numpy.std(nnat_array19, axis=0)
#     nnat_mean20 = numpy.array([sum(x)/len(x) for x in zip(*nnat_array20)])
#     nnat_stdeviation20 = numpy.std(nnat_array20, axis=0)
#
#     ##### Print for JMP analysis #####
#     # print('nat_mean17')
#     # for x in nat_mean17:
#     #     print(x)
#     # print('nat_mean18')
#     # for x in nat_mean18:
#     #     print(x)
#     # print('nat_mean19')
#     # for x in nat_mean19:
#     #     print(x)
#     # print('nat_mean20')
#     # for x in nat_mean20:
#     #     print(x)
#     #
#     # print('nnat_mean17')
#     # for x in nnat_mean17:
#     #     print(x)
#     # print('nnat_mean18')
#     # for x in nnat_mean18:
#     #     print(x)
#     # print('nnat_mean19')
#     # for x in nnat_mean19:
#     #     print(x)
#     # print('nnat_mean20')
#     # for x in nnat_mean20:
#     #     print(x)
#     ##################################
#
#
# #     fig, ax = plt.subplots(1, figsize=(6, 6))
# #     ax.plot(nat_samples17, nat_mean17, color=lightblue, zorder=2)
# #     ax.errorbar(nat_samples17, nat_mean17, nat_stdeviation17, color=lightblue, zorder=1, capsize=1, elinewidth=1, markeredgewidth=1, label='Native 2017')
# #     ax.plot(nat_samples18, nat_mean18, color=blue, zorder=2)
# #     ax.errorbar(nat_samples18, nat_mean18, nat_stdeviation18, color=blue, zorder=1, capsize=1, elinewidth=1, markeredgewidth=1, label='Native 2018 (pre-fire)')
# #     ax.plot(nat_samples19, nat_mean19, color=darkblue, zorder=2)
# #     ax.errorbar(nat_samples19, nat_mean19, nat_stdeviation19, color=darkblue, zorder=1, capsize=1, elinewidth=1, markeredgewidth=1, label='Native 2019')
# #     ax.plot(nat_samples20, nat_mean20, color=darkestblue, zorder=2)
# #     ax.errorbar(nat_samples20, nat_mean20, nat_stdeviation20, color=darkestblue, zorder=1, capsize=1, elinewidth=1, markeredgewidth=1, label='Native 2020')
# #
# #     ax.plot(nnat_samples17, nnat_mean17, color=yellow, zorder=2)
# #     ax.errorbar(nnat_samples17, nnat_mean17, nnat_stdeviation17, color=yellow, zorder=1, capsize=1, elinewidth=1, markeredgewidth=1, label='Non-native 2017')
# #     ax.plot(nnat_samples18, nnat_mean18, color=gold, zorder=2)
# #     ax.errorbar(nnat_samples18, nnat_mean18, nnat_stdeviation18, color=gold, zorder=1, capsize=1, elinewidth=1, markeredgewidth=1, label='Non-native 2018 (pre-fire')
# #     ax.plot(nnat_samples19, nnat_mean19, color=rust, zorder=2)
# #     ax.errorbar(nnat_samples19, nnat_mean19, nnat_stdeviation19, color=rust, zorder=1, capsize=1, elinewidth=1, markeredgewidth=1, label='Non-native 2019')
# #     ax.plot(nnat_samples20, nnat_mean20, color=darkred, zorder=2)
# #     ax.errorbar(nnat_samples20, nnat_mean20, nnat_stdeviation20, color=darkred, zorder=1, capsize=1, elinewidth=1, markeredgewidth=1, label='Non-native 2020')
# #
# #     ax.tick_params(labelsize=ea_axisnum)
# #     ### plt.text(10, Hotspot18=22.25; OutW19=136; OutW20=116; InW20=190; InW19=183; InW18=108
# #     plt.legend(prop={"size":ea_legend}) #, loc='center right')
# #     plt.ylabel('Number of species', fontsize=ea_axislabel)
# #     plt.xlabel('Samples', fontsize=ea_axislabel)
# #     plt.tight_layout()
# #     plt.savefig("/Users/nicolettastork/Documents/Geography/SMMNRA_Invasive_Species/NPSData/VegData_FromSMMNRA_Modified/Fig_Rarefaction_Hotspot.png", dpi=600)
# #
# # rarefactionHotspot(n_specieswithinhotspot2017, n_specieswithinhotspot2018, n_specieswithinhotspot2019, n_specieswithinhotspot2020,
# #                    nn_specieswithinhotspot2017, nn_specieswithinhotspot2018, nn_specieswithinhotspot2019, nn_specieswithinhotspot2020,
# #                    speciestotals17, speciestotals18, speciestotals19, speciestotals20)

###########################################################################
# ##### FIGURE 7: TOP SPECIES WITHIN HOTSPOT #####
###########################################################################
# nat_ann_herb = numpy.array([])
# nnat_ann_herb = numpy.array([])
# nat_per_herb = numpy.array([])
# nnat_per_herb = numpy.array([])
#
# nat_ann_grass = numpy.array([])
# nnat_ann_grass = numpy.array([])
# nat_per_grass = numpy.array([])
# nnat_per_grass = numpy.array([])
#
# nat_per_shrub = numpy.array([])
# nnat_per_shrub = numpy.array([])
#
# nat_per_subshrub = numpy.array([])
# nat_per_tree = numpy.array([])
#
# for name, obj in monitoringpoints.items():
#     test = numpy.where((obj.native_status == "Native") & (obj.ann_per == "Annual") & (obj.fxngroup == "Herbaceous"))
#     nat_ann_herb = numpy.append(nat_ann_herb, obj.species_code[test])
#     test = numpy.where((obj.native_status == "Nonnative") & (obj.ann_per == "Annual") & (obj.fxngroup == "Herbaceous"))
#     nnat_ann_herb = numpy.append(nnat_ann_herb, obj.species_code[test])
#     test = numpy.where((obj.native_status == "Native") & (obj.ann_per == "Perennial") & (obj.fxngroup == "Herbaceous"))
#     nat_per_herb = numpy.append(nat_per_herb, obj.species_code[test])
#     test = numpy.where((obj.native_status == "Nonnative") & (obj.ann_per == "Perennial") & (obj.fxngroup == "Herbaceous"))
#     nnat_per_herb = numpy.append(nnat_per_herb, obj.species_code[test])
#
#     test = numpy.where((obj.native_status == "Native") & (obj.ann_per == "Annual") & (obj.fxngroup == "Grass"))
#     nat_ann_grass = numpy.append(nat_ann_grass, obj.species_code[test])
#     test = numpy.where((obj.native_status == "Nonnative") & (obj.ann_per == "Annual") & (obj.fxngroup == "Grass"))
#     nnat_ann_grass = numpy.append(nnat_ann_grass, obj.species_code[test])
#     test = numpy.where((obj.native_status == "Native") & (obj.ann_per == "Perennial") & (obj.fxngroup == "Grass"))
#     nat_per_grass = numpy.append(nat_per_grass, obj.species_code[test])
#     test = numpy.where((obj.native_status == "Nonnative") & (obj.ann_per == "Perennial") & (obj.fxngroup == "Grass"))
#     nnat_per_grass = numpy.append(nnat_per_grass, obj.species_code[test])
#
#     test = numpy.where((obj.native_status == "Native") & (obj.ann_per == "Perennial") & (obj.fxngroup == "Shrub"))
#     nat_per_shrub = numpy.append(nat_per_shrub, obj.species_code[test])
#     test = numpy.where((obj.native_status == "Nonnative") & (obj.ann_per == "Perennial") & (obj.fxngroup == "Shrub"))
#     nnat_per_shrub = numpy.append(nnat_per_shrub, obj.species_code[test])
#
#     test = numpy.where((obj.native_status == "Native") & (obj.ann_per == "Perennial") & (obj.fxngroup == "Sub-shrub"))
#     nat_per_subshrub = numpy.append(nat_per_subshrub, obj.species_code[test])
#
#     test = numpy.where((obj.native_status == "Native") & (obj.ann_per == "Perennial") & (obj.fxngroup == "Tree"))
#     nat_per_tree = numpy.append(nat_per_tree, obj.species_code[test])
#
#
# fxngroups = {"ann_grass": numpy.unique(numpy.append(nnat_ann_grass, nat_ann_grass)),
#              "per_grass": numpy.unique(numpy.append(nnat_per_grass, nat_per_grass)),
#              "ann_herb": numpy.unique(numpy.append(nnat_ann_herb, nat_ann_herb)),
#              "per_herb": numpy.unique(numpy.append(nnat_per_herb, nat_per_herb)),
#              "per_shrub": numpy.unique(numpy.append(nnat_per_shrub, nat_per_shrub)),
#              "per_subshrub": numpy.unique(nat_per_subshrub),
#              "per_tree": numpy.unique(nat_per_tree)}
#
#
# nlabels, ncounts = numpy.unique(n_specieswithinhotspot, return_counts=True)
# nnlabels, nncounts = numpy.unique(nn_specieswithinhotspot, return_counts=True)
# # print(numpy.unique(nlabels), numpy.unique(nnlabels))
#
# # ### Native 2017 ###
# nlabels17, ncounts17 = numpy.unique(n_specieswithinhotspot2017, return_counts=True)
# nfrac2017 = (ncounts17/index2017)*100
# nfractions17 = numpy.round(nfrac2017)
#
# # ### Native 2018 ###
# nlabels18, ncounts18 = numpy.unique(n_specieswithinhotspot2018, return_counts=True)
# nfrac2018 = (ncounts18/index2018)*100
# nfractions18 = numpy.round(nfrac2018)
#
# # ### Native 2019 ###
# nlabels19, ncounts19 = numpy.unique(n_specieswithinhotspot2019, return_counts=True)
# nfrac2019 = (ncounts19/index2019)*100
# nfractions19 = numpy.round(nfrac2019)
#
# # ### Native 2020 ###
# nlabels20, ncounts20 = numpy.unique(n_specieswithinhotspot2020, return_counts=True)
# nfrac2020 = (ncounts20/index2020)*100
# nfractions20 = numpy.round(nfrac2020)
#
# # ### Nonnative 2017 ###
# nnlabels17, nncounts17 = numpy.unique(nn_specieswithinhotspot2017, return_counts=True)
# nnfrac2017 = (nncounts17/index2017)*100
# nnfractions17 = numpy.round(nnfrac2017)
#
# # ### Nonnative 2018 ###
# nnlabels18, nncounts18 = numpy.unique(nn_specieswithinhotspot2018, return_counts=True)
# nnfrac2018 = (nncounts18/index2018)*100
# nnfractions18 = numpy.round(nnfrac2018)
#
# # ### Nonnative 2019 ###
# nnlabels19, nncounts19 = numpy.unique(nn_specieswithinhotspot2019, return_counts=True)
# nnfrac2019 = (nncounts19/index2019)*100
# nnfractions19 = numpy.round(nnfrac2019)
#
# # ### Nonnative 2020 ###
# nnlabels20, nncounts20 = numpy.unique(nn_specieswithinhotspot2020, return_counts=True)
# nnfrac2020 = (nncounts20/index2020)*100
# nnfractions20 = numpy.round(nnfrac2020)
#
# class Species(object):
#     def __init__(self):
#         self.percentage = numpy.array([])
#         self.fxngroup = ""
#
# ### Make dictionary:
# ### Keys = species name (label)
# ### Values = percentage of species from total observations in the area per year
#
# ########## NATIVE DICTIONARY ##########
# native17 = {nlabels17[i]: nfractions17[i] for i in range(len(nlabels17))}
# topnative17 = dict(heapq.nlargest(10, native17.items(), key=itemgetter(1)))
# native18 = {nlabels18[i]: nfractions18[i] for i in range(len(nlabels18))}
# topnative18 = dict(heapq.nlargest(10, native18.items(), key=itemgetter(1)))
# native19 = {nlabels19[i]: nfractions19[i] for i in range(len(nlabels19))}
# topnative19 = dict(heapq.nlargest(10, native19.items(), key=itemgetter(1)))
# native20 = {nlabels20[i]: nfractions20[i] for i in range(len(nlabels20))}
# topnative20 = dict(heapq.nlargest(10, native20.items(), key=itemgetter(1)))
#
# topnative = {}
# for key in set(list(topnative17.keys()) + list(topnative18.keys()) + list(topnative19.keys()) + list(topnative20.keys())):
#     obj = Species()
#     try:
#         obj.percentage = numpy.append(obj.percentage, topnative17[key])
#     except KeyError:
#         obj.percentage = numpy.append(obj.percentage, 0)
#     try:
#         obj.percentage = numpy.append(obj.percentage, topnative18[key])
#     except KeyError:
#         obj.percentage = numpy.append(obj.percentage, 0)
#     try:
#         obj.percentage = numpy.append(obj.percentage, topnative19[key])
#     except KeyError:
#         obj.percentage = numpy.append(obj.percentage, 0)
#     try:
#         obj.percentage = numpy.append(obj.percentage, topnative20[key])
#     except KeyError:
#         obj.percentage = numpy.append(obj.percentage, 0)
#     for group, species in fxngroups.items():
#         test = numpy.where(species == key)
#         if len(test[0]) == 1:
#             obj.fxngroup = group
#     topnative[key] = obj
#
# ###### NONNATIVE DICTIONARY ######
# nonnative17 = {nnlabels17[i]: nnfractions17[i] for i in range(len(nnlabels17))}
# topnonnative17 = dict(heapq.nlargest(10, nonnative17.items(), key=itemgetter(1)))
# nonnative18 = {nnlabels18[i]: nnfractions18[i] for i in range(len(nnlabels18))}
# topnonnative18 = dict(heapq.nlargest(10, nonnative18.items(), key=itemgetter(1)))
# nonnative19 = {nnlabels19[i]: nnfractions19[i] for i in range(len(nnlabels19))}
# topnonnative19 = dict(heapq.nlargest(10, nonnative19.items(), key=itemgetter(1)))
# nonnative20 = {nnlabels20[i]: nnfractions20[i] for i in range(len(nnlabels20))}
# topnonnative20 = dict(heapq.nlargest(10, nonnative20.items(), key=itemgetter(1)))
#
# topnonnative = {}
# # print(set(list(topnonnative17.keys()) + list(topnonnative18.keys()) + list(topnonnative19.keys()) + list(topnonnative20.keys())))
# for key in set(list(topnonnative17.keys()) + list(topnonnative18.keys()) + list(topnonnative19.keys()) + list(topnonnative20.keys())):
#     obj = Species()
#     try:
#         obj.percentage = numpy.append(obj.percentage, topnonnative17[key])
#     except KeyError:
#         obj.percentage = numpy.append(obj.percentage, 0)
#     try:
#         obj.percentage = numpy.append(obj.percentage, topnonnative18[key])
#     except KeyError:
#         obj.percentage = numpy.append(obj.percentage, 0)
#     try:
#         obj.percentage = numpy.append(obj.percentage, topnonnative19[key])
#     except KeyError:
#         obj.percentage = numpy.append(obj.percentage, 0)
#     try:
#         obj.percentage = numpy.append(obj.percentage, topnonnative20[key])
#     except KeyError:
#         obj.percentage = numpy.append(obj.percentage, 0)
#     for group, species in fxngroups.items():
#         test = numpy.where(species == key)
#         if len(test[0]) == 1:
#             obj.fxngroup = group
#     topnonnative[key] = obj
#
# for name, obj in topnative.items():
#     print(name, obj.percentage)
#
# for name, obj in topnonnative.items():
#     print(name, obj.percentage)
#
# fig, ax = plt.subplots(2, 1, figsize=(6, 8))
# X = (numpy.arange(4))
# w = 0.2
# n_colors = [lightestblue, lightblue, blue, darkestblue]
# nn_colors = [yellow, gold, rust, darkred]
# years = ['2017', '2018 (pre-fire)', '2019', '2020']
# for i in range(0, 4):
#     ############# Plot 1: Non-native ##############
#
#     nn_ann_grass = [obj.percentage[i] for name, obj in topnonnative.items() if obj.fxngroup == "ann_grass"]
#     nn_ann_grass_names = [name for name, obj in topnonnative.items() if obj.fxngroup == "ann_grass"]
#     nn_xpos_anngrass = [j + w*i for j, _ in enumerate(nn_ann_grass_names)]
#     ax[0].bar(nn_xpos_anngrass, nn_ann_grass, width=w, color=nn_colors[i], label=years[i], align='center')
#
#     nn_per_grass = [obj.percentage[i] for name, obj in topnonnative.items() if obj.fxngroup == "per_grass"]
#     nn_per_grass_names = [name for name, obj in topnonnative.items() if obj.fxngroup == "per_grass"]
#     nn_xpos_pergrass = [j + w*i + len(nn_xpos_anngrass) for j, _ in enumerate(nn_per_grass_names)]
#     ax[0].bar(nn_xpos_pergrass, nn_per_grass, width=w, color=nn_colors[i])
#
#     nn_ann_herb = [obj.percentage[i] for name, obj in topnonnative.items() if obj.fxngroup == "ann_herb"]
#     nn_ann_herb_names = [name for name, obj in topnonnative.items() if obj.fxngroup == "ann_herb"]
#     nn_xpos_annherb = [j + w*i + len(nn_xpos_anngrass + nn_xpos_pergrass) for j, _ in enumerate(nn_ann_herb_names)]
#     ax[0].bar(nn_xpos_annherb, nn_ann_herb, width=w, color=nn_colors[i])
#
#     nn_per_herb = [obj.percentage[i] for name, obj in topnonnative.items() if obj.fxngroup == "per_herb"]
#     nn_per_herb_names = [name for name, obj in topnonnative.items() if obj.fxngroup == "per_herb"]
#     nn_xpos_perherb = [j + w*i + len(nn_xpos_anngrass + nn_xpos_pergrass + nn_xpos_annherb) for j, _ in enumerate(nn_per_herb_names)]
#     ax[0].bar(nn_xpos_perherb, nn_per_herb, width=w, color=nn_colors[i])
#
#     ########## Plot 2: Native ###############
#
#     # retrieve the percentages from the topnative dictionary if the functional group is perennial shrub
#     n_per_shrub = [obj.percentage[i] for name, obj in topnative.items() if obj.fxngroup == "per_shrub"]
#     # retrieve the names from the topnative dictionary within perennial shrub dictionary
#     n_per_shrub_names = [name for name, obj in topnative.items() if obj.fxngroup == "per_shrub"]
#     # for each bar (j), plot it then plot the next one a distance of 0.2 away (w*i), do that i times (3 times).
#     n_xpos_pershrub = [j + w*i for j, _ in enumerate(n_per_shrub_names)]
#     ax[1].bar(n_xpos_pershrub, n_per_shrub, width=w, color=n_colors[i], label=years[i], align='center')
#
#     n_per_subshrub = [obj.percentage[i] for name, obj in topnative.items() if obj.fxngroup == "per_subshrub"]
#     n_per_subshrub_names = [name for name, obj in topnative.items() if obj.fxngroup == "per_subshrub"]
#     n_xpos_persubshrub = [j + w*i + len(n_xpos_pershrub) for j, _ in enumerate(n_per_subshrub_names)]
#     ax[1].bar(n_xpos_persubshrub, n_per_subshrub, width=w, color=n_colors[i])
#
#     n_ann_herb = [obj.percentage[i] for name, obj in topnative.items() if obj.fxngroup == "ann_herb"]
#     n_ann_herb_names = [name for name, obj in topnative.items() if obj.fxngroup == "ann_herb"]
#     n_xpos_annherb = [j + w*i + len(n_xpos_pershrub + n_xpos_persubshrub) for j, _ in enumerate(n_ann_herb_names)]
#     ax[1].bar(n_xpos_annherb, n_ann_herb, width=w, color=n_colors[i])
#
#     n_per_herb = [obj.percentage[i] for name, obj in topnative.items() if obj.fxngroup == "per_herb"]
#     n_per_herb_names = [name for name, obj in topnative.items() if obj.fxngroup == "per_herb"]
#     n_xpos_perherb = [j + w*i + len(n_xpos_pershrub + n_xpos_persubshrub + n_xpos_annherb) for j, _ in enumerate(n_per_herb_names)]
#     ax[1].bar(n_xpos_perherb, n_per_herb, width=w, color=n_colors[i])
#
#     n_per_tree = [obj.percentage[i] for name, obj in topnative.items() if obj.fxngroup == "per_tree"]
#     n_per_tree_names = [name for name, obj in topnative.items() if obj.fxngroup == "per_tree"]
#     n_xpos_pertree = [j + w*i + len(n_xpos_pershrub + n_xpos_persubshrub + n_xpos_annherb + n_xpos_perherb) for j, _ in enumerate(n_per_tree_names)]
#     ax[1].bar(n_xpos_pertree, n_per_tree, width=w, color=n_colors[i])
#
#     n_per_grass = [obj.percentage[i] for name, obj in topnative.items() if obj.fxngroup == "per_grass"]
#     n_per_grass_names = [name for name, obj in topnative.items() if obj.fxngroup == "per_grass"]
#     n_xpos_pergrass = [j + w*i + len(n_xpos_pershrub + n_xpos_persubshrub + n_xpos_annherb + n_xpos_perherb + n_xpos_pertree) for j, _ in enumerate(n_per_grass_names)]
#     ax[1].bar(n_xpos_pergrass, n_per_grass, width=w, color=n_colors[i])
#
# print('native perennial shrubs:', n_per_shrub_names)
# print('native perennial subshrubs:', n_per_subshrub_names)
# print('native annual herbaceous:', n_ann_herb_names)
# print('native perennial herbaceous:', n_per_herb_names)
# print('native perennial tree:', n_per_tree_names)
# print('native perennial grass:', n_per_grass_names)
#
# print('nonnative annual grasses:', nn_ann_grass_names)
# print('nonnative perennial grass:', nn_per_grass_names)
# print('nonnative annual herbaceous:', nn_ann_herb_names)
# print('nonnative perennial herbaceous:', nn_per_herb_names)
#
# nn_x = numpy.arange(len(nn_xpos_anngrass + nn_xpos_pergrass + nn_xpos_annherb + nn_xpos_perherb))
# ax[0].legend(fontsize=ea_axislabel)
# ax[0].set_ylabel('Percentage', fontsize=ea_axislabel)
# ax[0].set_ylim(0, 29)
# ax[0].text(0, 27, 'a)', fontsize=ea_panelletter)
# ax[0].set_xticks(nn_x + 0.3) # This allows you to adjust the placement of the tick marks.
# ax[0].set_xticklabels([i[0:6] for i in nn_ann_grass_names + nn_per_grass_names + nn_ann_herb_names + nn_per_herb_names], fontsize=8, rotation=90, ha='center')
#
# n_x = numpy.arange(len(n_xpos_pershrub + n_xpos_persubshrub + n_xpos_annherb + n_xpos_perherb + n_xpos_pertree + n_xpos_pergrass))
# ax[1].legend(fontsize=7)
# ax[1].set_ylabel('Percentage', fontsize=ea_axislabel)
# ax[1].set_xlabel('Species', fontsize=ea_axislabel)
# ax[1].set_ylim(0, 9.75)
# ax[1].text(0, 9, 'b)', fontsize=ea_panelletter)
# ax[1].set_xticks(n_x + 0.3) # This allows you to adjust the placement of the tick marks.
# ax[1].set_xticklabels([i[0:6] for i in n_per_shrub_names + n_per_subshrub_names + n_ann_herb_names + n_per_herb_names + n_per_tree_names + n_per_grass_names], fontsize=8, rotation=90, ha='center')
#
# plt.tight_layout()
# plt.savefig("/Users/nicolettastork/Documents/Geography/SMMNRA_Invasive_Species/NPSData/VegData_FromSMMNRA_Modified/Fig_HotspotTopSpecies.png", dpi=600)
