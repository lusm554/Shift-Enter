import numpy as np
import pandas as pd


def features(params_df_with_rolling):
    features = pd.DataFrame()
    features['CAO_charge_rolling_2'] = (
        params_df_with_rolling['CAO_charge_rolling_2']
    )
    features['log(Concentrate_rolling_4)/CAO_charge_rolling_4'] = (
        (np.log(params_df_with_rolling['Concentrate_rolling_4']))
        .divide(params_df_with_rolling['CAO_charge_rolling_4'])
    )
    features['CAO_charge_rolling_2**3*OSN_charge_rolling_4'] = (
        (params_df_with_rolling['CAO_charge_rolling_2'] ** 3)
        .multiply(params_df_with_rolling['OSN_charge_rolling_4'])
    )
    features['CAO_charge/TIO2_charge_rolling_4'] = (
        (params_df_with_rolling['CAO_charge'])
        .divide(params_df_with_rolling['TIO2_charge_rolling_4'])
    )
    features['CAO_charge**3*log(Sieving3mm_fuel_rolling_4)'] = (
        (params_df_with_rolling['CAO_charge'] ** 3)
        .multiply(np.log(params_df_with_rolling['Sieving3mm_fuel_rolling_4']))
    )
    features['sqrt(CAO_limestone_rolling_2)*CAO_charge**3'] = (
        (np.sqrt(params_df_with_rolling['CAO_limestone_rolling_2']))
        .multiply((params_df_with_rolling['CAO_charge'] ** 3))
    )
    return features


def findAllNaN(params):
    return params.fillna(10.0)
