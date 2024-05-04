"""Module with DPI conteiners"""
from dependency_injector import containers, providers

from src.services.plate_process import ProcessPlate, Storage


class Container(containers.DeclarativeContainer):
    """Container for DPI plates"""
    config = providers.Configuration()

    store = providers.Factory(
        Storage,
        config=config.content_process,
    )

    content_process = providers.Singleton(
        ProcessPlate,
        storage=store.provider(),
    )
