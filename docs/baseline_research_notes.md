# Expert-rules baseline: research notes

Status: DRAFT, 22 Oct 2026, Thierry (Part B). These are literature notes, not rules yet. Nothing here is agronomic advice. Every number below is copied from the source named next to it. Numbers marked GAP were not found and must not be filled from memory.

Scope: the four CIAT labels (beans, irish_potato, maize, rice). Cassava is not in the CIAT labels and was not researched.

## 1. What the sources give (by crop)

| Crop | Variable | Value | Source |
| --- | --- | --- | --- |
| Maize | Temperature, optimal / absolute | 18 to 33 C / 10 to 47 C | [1] |
| Maize | Annual rainfall, optimal / absolute | 600 to 1200 mm / 400 to 1800 mm | [1] |
| Maize | Soil pH, optimal / absolute | 5 to 7 / 4.5 to 8.5 | [1] |
| Maize | Mean temperature during growth | above 15 C; 10 to 15 C suits forage only; maturity is delayed 10 to 20 days for each 0.5 C below 20 C | [2] |
| Maize | Water need per crop cycle | 500 to 800 mm | [2] |
| Maize | Altitude | absolute maximum 4000 m, no minimum | [1] |
| Beans (dry, common) | Altitude | 600 to 1950 m | [3] |
| Beans | Temperature | grows within 17.5 to 27 C; flower buds fall above 30 C; seed rarely forms above 35 C | [3] |
| Beans | Rainfall per crop cycle | 300 to 400 mm, well distributed; dry weather at harvest | [3] |
| Beans | Growth habit | bush types predominate in Africa; climbing types suit high-rainfall and land-limited areas | [3] |
| Beans | Soil pH | "near-neutral", no number given. GAP | [3] |
| Irish potato | Altitude | grown "mainly above 1800 m" in Rwanda | [4] |
| Irish potato | Temperature | 16 to 20 C cited nationally; station means 21.2 C (Kayonza, 1581 m), 18.2 C (Nyamagabe, 1910 m), 16.1 C (Nyabihu, 2500 m) | [4] |
| Irish potato | Annual rainfall | 1000 to 1600 mm cited nationally; station means 929 mm (Kayonza), 1324 mm (Nyamagabe), 1273 mm (Nyabihu) | [4] |
| Irish potato | Seasons | B: February to June; A: September to December | [4] |
| Irish potato | Soil pH | GAP | none retrieved |
| Rice | Where grown | almost all rice area is marshland; districts listed include Nyagatare and Ruhango; Musanze and Nyabihu are not listed | [5] |
| Rice | Altitude, temperature, rainfall | no numbers in the national strategy. GAP | [5] |
| Rice | Cold risk | minimum below 10 C in early growth and below 14 C at reproduction damaged rice in Madagascar at 1500 m; Rwanda's Ruhengeri area is about 2000 m with cold-season lows near 10 C | [6] |
| Rice | Seasons | A: July to December; B: January to June | [5] |
| Rice | Soil pH | GAP | none retrieved |

Rwanda context (not crop specific):

| Item | Value | Source |
| --- | --- | --- |
| Rainfall seasons | short rains Sep to Dec; short dry Jan to Feb; long rains Mar to May; long dry Jun to Aug | [7] |
| Eastern region | below 700 mm per year, about 20 C | [7] |
| Western region | about 1500 mm per year, below 17 C | [7] |
| Elevation | west often above 2000 m; central plateau about 1500 to 2000 m; east mostly below 1500 m | [7] |

## 2. What this means for the baseline

1. The sources separate the crops mainly by altitude and temperature: potato high and cool, rice low and warm with water access, maize and beans in between. Rainfall and pH separate them little, because the ranges overlap.
2. Rice is decided mostly by water access (marshland or irrigation), and our feature table has no such layer. Without one, a rice rule based on climate alone will be weak. Ask Arsène whether a marshland or irrigation layer can be added, or accept rice as a known limitation and say so in the report.
3. No source gives numeric suitability by district for Rwanda. The baseline will therefore be generic crop requirements plus Rwanda regional climate. It will be coarse, and with only four districts it may behave like a district lookup. That is a fair finding to report, and it is the reason ML must be compared against it.
4. Because every number comes from a general source, each rule must carry its source and must stay status draft until a teammate checks the page.

## 3. Gaps and limits of this research

- The FAO EcoCrop data sheets for bean, potato and rice could not be retrieved. A fetch for the bean sheet returned the maize sheet, so no bean value from EcoCrop is used.
- The CIAT Atlas of Common Bean in Africa could not be opened (blocked to automated access). It may hold Rwanda-specific bean zones and should be read by hand.
- The bean figures in [3] have no stated author or date, and the page does not tie each figure to a source. Treat them as medium confidence.
- The potato paper [4] says part of its ranges come from other studies. Check the PDF before quoting.
- Not found: potato and rice soil pH, rice temperature optimum, and climate figures for Musanze, Nyagatare and Ruhango.
- The paper on maize fertiliser [8] is about fertiliser, not crop suitability.

## 4. References (IEEE)

[1] FAO, "EcoCrop: Zea mays ssp. mays data sheet," Food and Agriculture Organization. [Online]. Available: https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=238663

[2] FAO, "Crop information: Maize," Land and Water Division, FAO. [Online]. Available: https://www.fao.org/land-water/databases-and-software/crop-information/maize/en/

[3] Infonet-Biovision, "Beans (Phaseolus vulgaris L.)." [Online]. Available: https://assets.accessagriculture.org/s3fs-public/Beans_by_infonet_biovision.pdf

[4] A. Nduwayezu, K. D. Kouame, M. Cherif, A. Nduwumuremyi, S. Gaidashova, and K. Daouda, "Agro-climatic characterization of potato production areas in Rwanda: Meteorological data analysis and farmer perceptions," Int. J. Environ. Clim. Change, vol. 13, no. 1, pp. 62-74, Feb. 2023.

[5] Republic of Rwanda, "National Rice Development Strategy (2021 edition)." [Online]. Available: https://riceforafrica.net/wp-content/uploads/2021/09/rwanda_nrds2.pdf

[6] "Weathering the cold," Rice Today, IRRI, Mar. 9, 2010. [Online]. Available: https://ricetoday.irri.org/weathering-the-cold/

[7] Rwanda Environment Management Authority, "Technical report on downscaled climate projections for the National Adaptation Plan in Rwanda," Jun. 2022. [Online]. Available: https://www.rema.gov.rw/fileadmin/user_upload/REMA/Publications/Reports/FINAL_TECHNICAL_REPORT_ON_DOWNSCALED_CLIMATE_PROJECTIONS_-JUNE-2022.pdf

[8] B. T. Mudereri et al., "Smart fertiliser recommendations for maize in Rwanda," CGIAR working paper, Nov. 2025. [Online]. Available: https://cgspace.cgiar.org/bitstreams/ddda75f4-25c5-4d38-8160-6b35f237033f/download
