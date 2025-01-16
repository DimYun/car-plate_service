## Car plates project. FastAPI service (part 3/3)

This is the project for car plate OCR recognition, which include:
1. [Neural network segmentation model for car plate area with number selection (part 1/3)](https://github.com/DimYun/car-plate-segm_model)
2. [Neural network OCR model for plate character recognition (part 2/3)](https://github.com/DimYun/car-plate-ocr_model)
3. API service for these two models (part 3/3)
4. [Additional example how to use API service in Telegram bot](https://github.com/DimYun/car-plate_tg-bot)

Fast API service is develop according two neural network model for car plate segmentation and OCR.

`src/container_task.py` contain 2 classes:
* `Storage` - can save calculated data into `*.json` (simple "caching" procedure).
* `ProcessPlates` - can load ONNX models and process car plate numbers. After that, it save results in `Storage`.

`app.py` contain FastAPI application, which have 2 handles:
1. `get_content` - return content according to `content_id`.
2. `process_content` - generate content with NN models for uploaded image.

`src/container_task.py` contain example of calculating procedure.


Used technologies:

* FastAPI
* Dependencies injector containers
* CI/CD (test, deploy, destroy)
* DVC
* Docker
* Unit & Integration tests with coverage report
* Linters (flake8 + wemake)

**Disclaimers**:

* the project was originally crated and maintained in GitLab local instance, some repo functionality may be unavailable
* the project was created by me and me only


Location for manual test:
* https://car_plates_api.lydata.duckdns.org
* docs https://car_plates_api.lydata.duckdns.org/docs#/default/process_content_process_content_post


## Setup of environment

First, create and activate `venv`:
    ```bash
    python3 -m venv venv
    . venv/bin/activate
    ```

Next, install dependencies:
    ```bash
    make install
    ```

### Commands

#### Preparation
* `make install` - install python dependencies

#### Run service
* `make run_app` - run servie. You can define argument `APP_PORT`

#### Build docker
* `make build` - you can define arguments `DOCKER_TAG`, `DOCKER_IMAGE`

#### Static analyse
* `make lint` - run linters

#### Tests
* `make run_unit_tests` - run unit tests
* `make run_integration_tests` - run integration tests
* `make run_all_tests` - run all tests
