"""Main module for FastAPI car plates service."""
import argparse

import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf

from src.containers.containers import Container
from src.routes import plates as plates_routes
from src.routes.routers import router as app_router


def create_app() -> FastAPI:
    """
    Create FastAPI application with DPI Containers
    :return: FastAPI application
    """
    container = Container()
    cfg = OmegaConf.load("configs/config.yaml")
    container.config.from_dict(cfg)
    container.wire([plates_routes])

    app = FastAPI()
    app.include_router(app_router, prefix="/plates", tags=["plates"])
    return app


if __name__ == "__main__":

    def arg_parse():
        """
        Parse command line
        :return: dictionary with command line arguments
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("port", type=int, help="port number")
        return parser.parse_args()

    app = create_app()
    args = arg_parse()
    uvicorn.run(app, port=args.port, host="127.0.0.1")
