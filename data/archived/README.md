# Archived inputs

These three files are the inputs that cannot be regenerated identically, so they are versioned
here rather than fetched at run time. The gzip files decompress to the exact JSON the analysis
read; the CSV is already in its working form.

    chicago_roads.geojson.gz     4.9 MB   decompresses to 42.6 MB   md5 9705cdb96656afb41534b5570e573971
    london_roads.geojson.gz     21.0 MB   decompresses to 135.2 MB  md5 97d35133e4dbf06345c999c373ffe191
    chicago_2024_violent.csv     2.5 MB   73,488 rows              md5 32f83e888560e341e6e2101711cbf23c

To use them, decompress into the data directory the scripts read:

    gunzip -c chicago_roads.geojson.gz > ../raw/chicago_roads.geojson
    gunzip -c london_roads.geojson.gz  > ../raw/london_roads.geojson
    mv chicago_2024_violent.csv ../raw/

Licences: the two road extracts are Derivative Databases of OpenStreetMap and are distributed
under the Open Data Commons Open Database License (ODbL) v1.0, copyright OpenStreetMap
contributors. The Chicago extract is City of Chicago open data (dataset ijzp-q8t2). Both road
extracts post-date the 2024 incidents by one to two years; the manuscript reports that mismatch
rather than correcting it.
