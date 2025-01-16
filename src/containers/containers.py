"""Module with DPI conteiners"""
from dependency_injector import containers, providers

from src.services.plate_process import ProcessPlate, Storage
from src.services.preprocess_utils import OCRPredictor, PlatePredictor


class Container(containers.DeclarativeContainer):
    """Container for DPI plates"""
    config = providers.Configuration()

    store = providers.Singleton(
        Storage,
        config=config.content_process,
    )

    plate_predictor = providers.Singleton(
        PlatePredictor,
        config=config.plate_model_parameters,
    )

    ocr_predictor = providers.Singleton(
        OCRPredictor,
        config=config.ocr_model_parameters,
    )

    content_process = providers.Singleton(
        ProcessPlate,
        storage=store.provider(),
        plate_predictor=plate_predictor.provider(),
        ocr_predictor=ocr_predictor.provider(),
    )
