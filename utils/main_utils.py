import logging


def asserting_batch_size(length_data, batch_size):
    """
    Assert that each batch has the same batch size
    Args:
        length_data: The number of observations in the set for which we want a certain batch size
        batch_size: The number of observations in one batch

    Returns:
    An error in case the batch hav not all the same batch size
    """
    try:
        assert length_data % batch_size == 0

    except AssertionError as error:
        logging.logger.exception("batch_size must be chosen among the following: {batch_size_values}" \
                                 .format(batch_size_values=[value for value in range(1, length_data) if length_data % value == 0]))
        raise error