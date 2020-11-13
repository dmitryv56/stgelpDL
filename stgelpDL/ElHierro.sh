#!/bin/bash

./start_predictor.py --mode train --csv_dataset ../dataLaLaguna/ElHiero_24092020_27102020.csv --tsname Imbalance -vvv

./start_predictor.py --mode predict --csv_dataset ../dataLaLaguna/ElHiero_24092020_27102020.csv --tsname Imbalance -vvv


./start_predictor.py --mode train --csv_dataset ../dataLaLaguna/ElHiero_24092020_27102020.csv --tsname Real_demand -vvv

./start_predictor.py --mode predict --csv_dataset ../dataLaLaguna/ElHiero_24092020_27102020.csv --tsname Real_demand -vvv

./start_predictor.py --mode train --csv_dataset ../dataLaLaguna/ElHiero_24092020_27102020.csv --tsname Forecasting -vvv

./start_predictor.py --mode predict --csv_dataset ../dataLaLaguna/ElHiero_24092020_27102020.csv --tsname Forecasting -vvv


./start_predictor.py --mode train --csv_dataset ../dataLaLaguna/ElHiero_24092020_27102020.csv --tsname Programmed_demand -vvv

./start_predictor.py --mode predict --csv_dataset ../dataLaLaguna/ElHiero_24092020_27102020.csv --tsname Programmed_demand -vvv


./start_predictor.py --mode train --csv_dataset ../dataLaLaguna/ElHiero_24092020_27102020.csv --tsname Diesel_Power -vvv

./start_predictor.py --mode predict --csv_dataset ../dataLaLaguna/ElHiero_24092020_27102020.csv --tsname Diesel_Power -vvv


./start_predictor.py --mode train --csv_dataset ../dataLaLaguna/ElHiero_24092020_27102020.csv --tsname WindTurbine_Power -vvv

./start_predictor.py --mode predict --csv_dataset ../dataLaLaguna/ElHiero_24092020_27102020.csv --tsname WindTurbine_Power -vvv


./start_predictor.py --mode train --csv_dataset ../dataLaLaguna/ElHiero_24092020_27102020.csv --tsname Hydrawlic -vvv

./start_predictor.py --mode predict --csv_dataset ../dataLaLaguna/ElHiero_24092020_27102020.csv --tsname Hydrawlic -vvv


./start_predictor.py --mode train --csv_dataset ../dataLaLaguna/ElHiero_24092020_27102020.csv --tsname HydrawlicTurbine_Power -vvv

./start_predictor.py --mode predict --csv_dataset ../dataLaLaguna/ElHiero_24092020_27102020.csv --tsname HydrawlicTurbine_Power -vvv


./start_predictor.py --mode train --csv_dataset ../dataLaLaguna/ElHiero_24092020_27102020.csv --tsname Pump_Power -vvv

./start_predictor.py --mode predict --csv_dataset ../dataLaLaguna/ElHiero_24092020_27102020.csv --tsname Pump_Power -vvv

./start_predictor.py --mode train --csv_dataset ../dataLaLaguna/ElHiero_24092020_27102020.csv --tsname CO2 -vvv

./start_predictor.py --mode predict --csv_dataset ../dataLaLaguna/ElHiero_24092020_27102020.csv --tsname CO2 -vvv




