import rtCommon.bidsCommon as bidsCommon


def testEntitiesDictGeneration():
    """
    Ensure entitity dictionary is loaded and parsed properly
    Expected dictionary format:
      key: Full entity name
      value: Dictionary with keys "entitity, format, description"
    """
    entities = bidsCommon.loadBidsEntities()

    # Ensure entity count correct
    NUM_ENTITIES = 20  # Manually counted from the file entities.yaml
    assert len(entities) == NUM_ENTITIES

    # Check a sample of important keys are present
    importantKeySample = ["Subject", "Task", "Session"]
    for key in importantKeySample:
        assert key in entities.keys()

    # Ensure expected values are present for each entity
    expectedValueKeys = ["entity", "format", "description"]
    for valueDict in entities.values():
        for key in expectedValueKeys:
            assert key in valueDict.keys()
