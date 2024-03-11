import csv
import sys
from typing import Iterable, List, NamedTuple, Optional
from worldlayers.common import BlockSelector, parse_blockstate


class InvalidScannerOperation(Exception):
    pass


class DisplayBlockSelector(NamedTuple):
    # Blockstate selector
    # '+' separated list of CompactState
    selectors: str
    # Matplotlib colors / None for auto color
    color: Optional[str]
    # Legend name override / None - equal to `names`
    display_name: Optional[str]


def log(*args: object) -> None:
    print(*args, file=sys.stderr)


def split_to_blockstates(combined: str) -> List[BlockSelector]:
    return [parse_blockstate(state.strip()) for state in combined.split('+')]


def parse_block_selection(selectors: List[str]) -> List[DisplayBlockSelector]:
    '''Parse a list of display selectors

    Display selector is a CSV tuple of:
    1. Blockstate name (see `parse_blockstate()`)
    2. Matplotlib color to plot as (optional)
    3. Graph legend name (optional)
    '''
    blocks_shown = []
    for parts in csv.reader(selectors):
        while len(parts) < 3:
            parts.append('')
        
        # Test for selector correctness (calls parse, throws on error)
        split_to_blockstates(parts[0])

        blocks_shown.append(DisplayBlockSelector(
            selectors=parts[0],
            color=parts[1] if parts[1] else None,
            display_name=parts[2] if parts[2] else None,
        ))
    return blocks_shown


def load_block_selection(selection: List[str]) -> List[DisplayBlockSelector]:
    '''Parse block selection argument from command line

    Each selection may be:
    - Display selector (see `parse_block_selection()`)
    - Name of a file, prepended with "@", containing lines of display selectors.
      File can contain comments statring with "#".
    '''
    selectors: List[str] = []
    for part in selection:
        if part.startswith('@'):
            with open(part[1:]) as f:
                lines: Iterable[str] = f.readlines()
            lines = map(str.strip, lines)
            lines = filter(lambda s: s and not s.startswith('#'), lines)
            selectors.extend(lines)
        else:
            selectors.append(part)
    return parse_block_selection(selectors)
