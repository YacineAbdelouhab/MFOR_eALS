from __future__ import annotations

import argparse

from pyspark import SparkConf, SparkContext

from pyspark_eals_rdd.config import EALSConfig


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--input-format", choices=("json", "csv", "paper-rating"), default="json"
    )
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--header", action="store_true")


def create_context(app_name: str, config: EALSConfig | None = None) -> SparkContext:
    conf = (
        SparkConf()
        .setAppName(app_name)
        .setIfMissing("spark.master", "local[*]")
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        # La barre de progression de Spark reecrit la ligne courante avec des
        # retours chariot, ce qui efface les lignes que le job affiche lui-meme.
        .set("spark.ui.showConsoleProgress", "false")
    )
    if config and config.partitions:
        conf.set("spark.default.parallelism", str(config.partitions))
    sc = SparkContext.getOrCreate(conf)
    sc.setLogLevel("WARN")
    return sc
