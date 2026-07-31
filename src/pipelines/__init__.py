"""Runnable pipelines: the project's code as scripts rather than notebooks (Week 9).

Week 9 adds two entry points, both importable *and* runnable from a shell:

* :mod:`src.pipelines.training_pipeline` — load the raw data, build the full
  ``Pipeline`` (Week 3 preprocessing + the Week 8 final model), fit it, evaluate
  it on the held-out test set and save it to ``models/crop_model.joblib``.
* :mod:`src.pipelines.predict_pipeline` — load that artifact (training one first
  if it is missing) and turn a dictionary of measurements into a crop label.

Neither module imports anything from ``notebooks/``. The dependency runs the
other way: a notebook may call these, but nothing here needs a kernel, a cell
order or a display to work.

Unlike the other ``src`` packages this one re-exports **nothing**. Both modules
are executable scripts, and re-exporting them here would make Python import
each one twice when it is run with ``-m`` — once as
``src.pipelines.training_pipeline`` and once as ``__main__`` — which it warns
about at run time. Import from the modules directly::

    from src.pipelines.training_pipeline import train_pipeline
    from src.pipelines.predict_pipeline import predict
"""
