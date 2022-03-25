#!/usr/bin/env python3


from pathlib import Path

IMBALANCE = "Imbalance"

SEED = 1957 #2022


PATH_ROOT_FOLDER = Path(Path(__file__).parent.absolute())
LOG_FOLDER ="Classification_Logs"
CLASSIFICATION_TASKS = {

    "Demand-Programmed-Diesel-Wind-Hydrawlic":['Real_demand', 'Programmed_demand', 'Diesel_Power', 'WindGen_Power',
                                               'Hydrawlic'],
    "Diesel-Wind-Hydrawlic":[ 'Diesel_Power', 'WindGen_Power', 'Hydrawlic'],
    "Diesel": ['Diesel_Power'],
    "Wind": ['WindGen_Power'],
    "Hydrawlic": ['Hydrawlic'],
    "Demand-Programmed":['Real_demand', 'Programmed_demand'],
    "Demand":['Real_demand'],
    "Programmed":['Programmed_demand'],
     "Demand-Diesel":['Real_demand',  'Diesel_Power' ],
    "Demand-Wind":['Real_demand', 'WindGen_Power'],
    "Demand-Hydrawlic":['Real_demand', 'Hydrawlic'],

}