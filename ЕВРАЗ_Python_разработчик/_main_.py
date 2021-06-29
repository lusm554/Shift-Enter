import pandas as pd
from data_examples import *
from pipeline import MeanDummyRegressor, predict_cao

if __name__ == '__main__':
    print(predict_cao(CHARGE_CHEMISTRY, LIMESTONE_CONSUMPTIONS,
          CHARGE_CONSUMPTIONS, COKE_CONSUMPTIONS, COKE_SIEVING, LIMESTONE_CAO))
