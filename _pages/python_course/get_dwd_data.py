# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021, earthobservations developers.
# Distributed under the MIT License. See LICENSE for more info.
"""
=====
About
=====
Acquire station information from DWD.

"""  # Noqa:D205,D400
import logging
from datetime import datetime
import polars as pl

from wetterdienst.provider.dwd.observation import (
    DwdObservationDataset,
    DwdObservationPeriod,
    DwdObservationRequest,
    DwdObservationResolution,
)

log = logging.getLogger()


def station_example():
    """Retrieve stations_result of DWD that measure temperature."""
    stations = DwdObservationRequest(
        parameter=[DwdObservationDataset.TEMPERATURE_AIR, DwdObservationDataset.TEMPERATURE_EXTREME, DwdObservationDataset.PRECIPITATION, DwdObservationDataset.PRESSURE, DwdObservationDataset.SUN, DwdObservationDataset.WIND, DwdObservationDataset.WIND_EXTREME,DwdObservationDataset.SOLAR, DwdObservationDataset.DEW_POINT, DwdObservationDataset.MOISTURE],
        resolution=DwdObservationResolution.MINUTE_10,
        period=DwdObservationPeriod.HISTORICAL,
        start_date=datetime(1996, 6, 14),
        end_date=datetime(2023, 10, 16),
    )

    result = stations.filter_by_station_id(963)

    return result.df, result.values.all().df


def main():
    """Run example."""
    logging.basicConfig(level=logging.INFO)
    station_info, values = station_example()
    parameters = values["parameter"].unique()
    print()
    df = pl.DataFrame({
        "data_time" : values["date"].unique(),
        "precipitation": values.filter(pl.col("parameter") == "precipitation_height")["value"],
        "precipitation_duration": values.filter(pl.col("parameter") == "precipitation_duration")["value"],
        "tair_2m_mean": values.filter(pl.col("parameter") == "temperature_air_mean_200")["value"],
        "tair_2m_max": values.filter(pl.col("parameter") == "temperature_air_max_200")["value"],
        "tair_2m_min": values.filter(pl.col("parameter") == "temperature_air_min_200")["value"],
        "tair_2m_dp_mean": values.filter(pl.col("parameter") == "temperature_dew_point_mean_200")["value"],
        "tair_5cm_mean": values.filter(pl.col("parameter") == "temperature_air_mean_200")["value"],
        "tair_5cm_max": values.filter(pl.col("parameter") == "temperature_air_max_005")["value"],
        "tair_5cm_min": values.filter(pl.col("parameter") == "temperature_air_min_005")["value"],
        "rH": values.filter(pl.col("parameter") == "humidity")["value"],
        "SWIN": values.filter(pl.col("parameter") == "radiation_global")["value"],
        "LWIN": values.filter(pl.col("parameter") == "radiation_sky_long_wave")["value"],
        "wind_speed": values.filter(pl.col("parameter") == "wind_speed")["value"],
        "wind_speed_min": values.filter(pl.col("parameter") == "wind_speed_min")["value"],
        "wind_gust_max": values.filter(pl.col("parameter") == "wind_gust_max")["value"],
        "wind_direction": values.filter(pl.col("parameter") == "wind_direction")["value"],
        "wind_direction_gust_max": values.filter(pl.col("parameter") == "wind_direction_gust_max")["value"],
        "pressure_air": values.filter(pl.col("parameter") == "pressure_air_site")["value"],
    })
    df.write_parquet("./diepholz_data_1996_2023.parquet")

if __name__ == "__main__":
    main()