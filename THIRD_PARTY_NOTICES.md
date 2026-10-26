# Third-party notices

## Lenia life-form data

The `经典-...` presets in `apps/Lenia presets.npz` are decoded from the life-form
catalogue in [Chakazul/Lenia](https://github.com/Chakazul/Lenia) by Bert Chan,
remapped to this project's kernel and re-validated.

MIT License

Copyright (c) 2018 Bert Chan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## ETOPO1 global relief

`apps/Physics/worldelev.npz` is derived from ETOPO1, the 1 arc-minute global
relief model of NOAA NGDC (now NCEI), sampled to a 1-degree grid.

Amante, C. and B. W. Eakins, 2009: ETOPO1 1 Arc-Minute Global Relief Model:
Procedures, Data Sources and Analysis. NOAA Technical Memorandum NESDIS
NGDC-24, 19 pp.

Source: https://www.ngdc.noaa.gov/mgg/global/global.html
Access used: NOAA ERDDAP, https://cwcgom.aoml.noaa.gov/erddap/griddap/etopo180.html

The data are provided free of charge for use and redistribution; NOAA and the
data contributors make no warranty of accuracy or fitness for a particular
purpose.

## NCEP/NCAR Reanalysis 1 wind

`apps/Physics/wind_real.npz` contains monthly mean 10 m wind fields of the
NCEP/NCAR Reanalysis 1 for 2015-2024 (120 months), together with derived
calendar-month climatologies and annual-mean linear trends. The data were
obtained from the NOAA Physical Sciences Laboratory (PSL) through its THREDDS
NetCDF Subset Service.

Kalnay, E., et al., 1996: The NCEP/NCAR 40-Year Reanalysis Project. Bulletin
of the American Meteorological Society, 77, 437-471.

Source: https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html
Service used: https://psl.noaa.gov/thredds/ncss/grid/Datasets/ncep.reanalysis/

The data are provided by NOAA/OAR/ESRL PSL and are not subject to U.S.
copyright; attribution is requested.
