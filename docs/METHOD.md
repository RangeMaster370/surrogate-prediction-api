# Method

## Design
`app/main.py` holds the routes, `app/schemas.py` the validation rules, `app/model.py` the surrogate and `app/gp.py` the NumPy Gaussian process. The model is created once in the application lifespan and kept in `app.state`, so requests only do a prediction.

## Model
Same synthetic simulator and GP as the gp-surrogate-active-learning project: inputs ambient temperature, EVs charging and shifted share, output transformer hot-spot temperature with 1.0 degC noise. 40 Latin-hypercube runs, fixed seed 0, so the model version (a hash of the training data) is reproducible. The interval is plus or minus 1.96 predictive standard deviations including observation noise.

## Validation
Pydantic bounds: ambient -10 to 35, EVs 0 to 250, shifted share 0 to 1; extra fields forbidden; batches of 1 to 100.

## Testing
Unit-level tests through `fastapi.testclient` (httpx): health, model info, accuracy against the simulator truth, monotonic behaviour, seven invalid-input cases, batch equals single calls, batch limits, reproducible model version.

## Limits
Latency numbers are in process. The surrogate is retrained at every start-up (about a second); a real service would load a versioned artefact. Docker and CI are untested, and the Docker image uses Python 3.12 while the tests were run on 3.13.

## Next steps
Build and run the image, run CI on GitHub, load a stored model artefact, add request logging and metrics, and deploy to a cloud container service.
