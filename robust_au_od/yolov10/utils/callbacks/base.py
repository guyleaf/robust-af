def replace_integration_callbacks(instance):
    """
    Replace integration callbacks with our implementation.

    Args:
        instance (Trainer, Predictor, Validator, Exporter): An object with a 'callbacks' attribute that is a dictionary
            of callback lists.
    """

    callbacks_list = []

    # Load training callbacks
    if "Trainer" in instance.__class__.__name__:
        from ultralytics.utils.callbacks.wb import callbacks as wb_cb

        from .wb import callbacks as new_wb_cb

        callbacks_list.extend([(wb_cb, new_wb_cb)])

    # replace the integration callbacks with our implementation
    for old_callbacks, new_callbacks in callbacks_list:
        for k, v in old_callbacks.items():
            try:
                instance.callbacks[k].remove(v)
            except ValueError:
                pass

        for k, v in new_callbacks.items():
            if v not in instance.callbacks[k]:
                instance.callbacks[k].append(v)
