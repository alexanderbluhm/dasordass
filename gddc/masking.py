'''
Preparing inputs with a mask string to be further processed
'''

from typing import List, Tuple, Optional


def get_masked_inputs(input: str, searches: List[str], mask_str: str, lower: Optional[bool] = False) -> List[str]:
    '''
    Returns a list of masked inputs. In each input, one search strings is replaced with the mask string.
    See `get_positions` for more details on the search.
    '''
    positions = get_positions(input, searches, lower)
    masked_inputs = []
    for pos_start, pos_end in positions:
        # replace the search string with the mask string
        masked = input[:pos_start] + mask_str + input[pos_end+1:]
        masked_inputs.append(masked)
    return masked_inputs, positions


def get_positions(input: str, searches: List[str], lower: Optional[bool] = False) -> List[Tuple[int, int]]:
    '''
    Returns a list of start and end positions of the search strings in the input.
    If a first match is found, the next search string is not checked in the same window.
    '''

    # sort searches for length
    searches = sorted(searches, key=lambda s: len(s), reverse=True)

    positions = []
    i = 0
    while i < len(input):
        for search in searches:
            window = input[i:i+len(search)]
            if lower:
                search = search.lower()
                window = window.lower()

            if window == search:
                positions.append((i, i+len(search)-1))
                # jump to the end of the search string
                i += len(search)
                # break the loop to not check the same window again
                break

        # if no match was found (no break before), look at the next window
        i += 1

    return positions
