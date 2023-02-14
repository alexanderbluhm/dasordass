from gddc.masking import get_masked_inputs, get_positions


def test_get_masked_inpputs():
    results, _ = get_masked_inputs('abc test', ['abc', 'abcd'], '[MASK]')
    assert results == ['[MASK] test']

    results, _ = get_masked_inputs('abcd test', ['abc', 'abcd'], '[MASK]')
    assert results == ['[MASK] test']

    results, _ = get_masked_inputs('abcd abc test', ['abc', 'abcd'], '[MASK]')
    assert results == ['[MASK] abc test', 'abcd [MASK] test']

    results, _ = get_masked_inputs('abcd abc test aabc abcd abc', [
        'abc', 'abcd'], '[MASK]')
    assert len(results) == 5


def test_get_positions():
    positions = get_positions('Abc test', ['abc', 'abcd'])
    # we ignore lower case by default, so the list should be empty
    assert positions == []

    positions = get_positions('Abc test', ['abc', 'abcd'], lower=True)
    assert positions == [(0, 2)]

    positions = get_positions('abcd', ['abc', 'abcd'])
    # only the longer search string should be found
    assert positions == [(0, 3)]

    positions = get_positions('Abcd abc test', [
        'abc', 'abcd'], lower=True)
    assert positions == [(0, 3), (5, 7)]
