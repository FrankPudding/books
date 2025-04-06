# books

### How to install dependencies
1. First, install poetry if not already installed (See [here](https://python-poetry.org/docs/#installing-with-the-official-installer))
1. Create a virtual environment with Python 3.13 or above
    ```bash
    poetry env use 3.13
    ```
1. Install dependences
    ```bash
    poetry install
    ```

### How to train a model
```bash
poetry run books train
```
This can take upwards of 25 minutes using the defualt config which trains on the entire 10k dataset. If you wish to use a different dataset, see the [Config](books/application/config.py) class for configuration options.
Training will print out a model uri once complete and that can be used for inference.


### How to run inference
```bash
poetry run python books predict --model-uri <your-model-uri> --text <text-to-be-evaluated>
```
Using the model uri from running the training pipeline, you can run inference on a chosen piece of text which will output a prediction of the sentiment.


## TODO
----
1. Implement Fast API to create an inference web server
1. Create a Dockerfile
1. Allow for providing config to the books CLI
1. Allow for batch inference in the books CLI
1. Improve logging