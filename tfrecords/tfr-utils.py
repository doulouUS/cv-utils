import tensorflow as tf

def list_record_features(tfrecords_path: str):
    """[summary]

    Args:
        tfrecords_path (str): Path to tfrecord file

    Returns:
        tuple: first element is a dict describing the features present in the 
        tfrecords, second element is the number of samples in the tfrecord file.
    """
    # Dict of extracted feature information
    features = {}
    nb_samples = 0
    # Iterate records
    for rec in tf.data.TFRecordDataset([str(tfrecords_path)]):
        nb_samples += 1
        # Get record bytes
        example_bytes = rec.numpy()
        # Parse example protobuf message
        example = tf.train.Example()
        example.ParseFromString(example_bytes)
        # Iterate example features
        for key, value in example.features.feature.items():
            # Kind of data in the feature
            kind = value.WhichOneof('kind')
            # Size of data in the feature
            size = len(getattr(value, kind).value)
            # Check if feature was seen before
            if key in features:
                # Check if values match, use None otherwise
                kind2, size2 = features[key]
                if kind != kind2:
                    kind = None
                if size != size2:
                    size = None
            # Save feature data
            features[key] = (kind, size)
    return features, nb_samples