# Reference data

Small public files that the code reads as they are. No farm-level data.

| File | What it is | Source and licence |
| --- | --- | --- |
| `rwanda_districts.geojson` | Outlines of Rwanda's 30 districts (level 2), simplified. Each feature has `district` (the name, spelled as in the CIAT and NISR tables) and `shapeID`. | geoBoundaries gbOpen, RWA ADM2, boundaryID `RWA-ADM2-39286606` (build of 12 Dec 2023), simplified version [1]. Original source: Open Data Rwanda, NISR geodata, 2012 district boundaries. Licence: CC BY 4.0, so the credit stays wherever the outlines are shown. |

Used by `src/recommendation/map_model.py` to find the district of a clicked point, and drawn on the dashboard map page, which shows the credit under the map.

Why this file and not the Kaggle one mentioned on 28 Oct: it has the same NISR district boundaries, a clear licence and a stable ID, and the four pilot district names match the CIAT and NISR tables without any renaming.

How it was made (29 Oct 2026): downloaded the simplified geoBoundaries GeoJSON for RWA ADM2, renamed `shapeName` to `district`, kept only `district` and `shapeID`, rounded the coordinates to 5 decimals (about 1 m) to make the file smaller, and added a `source` note at the top. The vertices are the same as in the download.

[1] D. Runfola et al., "geoBoundaries: A global database of political administrative boundaries," PLoS ONE, vol. 15, no. 4, e0231866, 2020.
