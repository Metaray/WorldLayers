import argparse
import threading
import csv
import re
from typing import Any, List, NamedTuple, Optional, Iterable, Tuple
import numpy as np
from numpy.typing import NDArray
from common import *
from scanners import create_scan


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


def split_to_blockstates(combined: str) -> List[BlockSelector]:
    return [parse_blockstate(state.strip()) for state in combined.split('+')]


def operation_save(args: argparse.Namespace, scan_data: DimScanData) -> None:
    extract_file = args.output
    save_scan_data(extract_file, scan_data)
    log(f'Saved block counts to {extract_file}')


def visualize_print(args: argparse.Namespace, scan_data: DimScanData) -> None:
    chunks_scanned = scan_data.chunks_scanned
    totals = scan_data.histogram.sum(axis=0)
    volume = chunks_scanned * scan_data.height * 16**2

    if args.sumstates:
        visdata = [
            (
                None,
                sum(totals[scan_data.blockstate_to_idx[state]] for state in states),
                name,
            )
            for name, states in scan_data.name_to_blockstates.items()
        ]
    else:
        idx_to_blockstate = scan_data.idx_to_blockstate
        visdata = [(i, totals[i], idx_to_blockstate[i]) for i in range(scan_data.state_count)]

    if args.sort == 'count':
        visdata.sort(key=lambda ci: -ci[1])  # sort by block count descending
    elif args.sort == 'name':
        visdata.sort(key=lambda ci: ci[2])  # sort by blockstate lexicographically
    else:
        visdata.sort(key=lambda ci: ci[0])  # sort by state id ascending (legacy)
    
    def show_count(name: str, count: int) -> None:
        print(
            f'{name} = {count}',
            f'({count / volume:%})',
            f'({count / chunks_scanned:.4f} b/ch)',
            sep='\t'
        )

    log(f'{len(scan_data.name_to_blockstates)} unique blocks')
    log(f'{scan_data.state_count} block states')

    airsum = sum_blocks_selection(scan_data, AIR_BLOCKS).sum()
    show_count('Nonair blocks', totals.sum() - airsum)

    for _, total, name in visdata:
        show_count(name, total)


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


def load_block_selection(selection: str) -> List[DisplayBlockSelector]:
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


def visualize_plot(args: argparse.Namespace, scan_data: DimScanData) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    
    if args.layers:
        crop_histogram(scan_data, args.layers)

    chunks_scanned = scan_data.chunks_scanned
    blocks_shown = load_block_selection(args.select)

    if args.norm == 'base':
        y_label = f'Fraction relative to\n{args.normbase}'
        base_mx = sum_blocks_selection(scan_data, split_to_blockstates(args.normbase))
        def norm_func(x: NDArray[np.int64]) -> NDArray[np.float64]:
            return np.where(base_mx > 0, x / np.maximum(base_mx, 1), 0)
    
    elif args.norm == 'total':
        y_label = 'Fraction per block'
        def norm_func(x: NDArray[np.int64]) -> NDArray[np.float64]:
            return x / (16**2 * chunks_scanned)
    
    elif args.norm == 'chunk':
        y_label = 'Blocks per chunk layer'
        def norm_func(x: NDArray[np.int64]) -> NDArray[np.float64]:
            return x / chunks_scanned
    
    else:
        y_label = 'Total count per layer'
        def norm_func(x: NDArray[np.int64]) -> NDArray[np.float64]:
            return x.astype(np.float64)

    class GraphData(NamedTuple):
        counts: NDArray[np.float64]
        color: Optional[str]
        label: str
    
    fig, ax = plt.subplots(figsize=(8, 5))

    base_y = scan_data.base_y
    height = scan_data.height
    y_range = list(range(base_y, base_y + height))
    graphs: List[GraphData] = []
    
    for show_info in blocks_shown:
        display_name = show_info.display_name or show_info.selectors

        try:
            hist = sum_blocks_selection(scan_data, split_to_blockstates(show_info.selectors))
            print(f'{display_name} = {hist.sum()} blocks')
        except ValueError:
            log(f'Found no blocks matching selectors: {show_info.selectors}')
            hist = scan_data.zero_histogram
        
        graphs.append(GraphData(
            counts=norm_func(hist),
            color=show_info.color,
            label=display_name,
        ))

    if not args.cumulative:
        for graph in graphs:
            ax.plot(y_range, graph.counts, color=graph.color, label=graph.label)
    else:
        ax.stackplot(
            y_range,
            [g.counts for g in graphs],
            colors=[g.color for g in graphs],
            labels=[g.label for g in graphs],
        )

    if args.solids:
        nonair_hist = (chunks_scanned * 16**2) - sum_blocks_selection(scan_data, AIR_BLOCKS)
        ax.plot(y_range, norm_func(nonair_hist), color='lightgray', label='Non-air')
    
    ax.grid(True)
    ax.legend()
    ax.set_title('Block distribution by height')
    ax.set_xlabel('Y level')
    ax.set_ylabel(y_label)
    ax.set_xlim(y_range[0], y_range[-1])
    ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 4, 8], integer=True, min_n_ticks=7))
    
    if args.savefig:
        # plt.ylim(0.0, 0.01)
        plt.savefig(args.savefig, dpi=300)
    else:
        plt.show()


def vis_print_as_csv(args: argparse.Namespace, scan_data: DimScanData) -> None:
    if args.layers:
        crop_histogram(scan_data, args.layers)

    if '*' in args.select:
        blocks_shown = None
    else:
        blocks_shown = load_block_selection(args.select)

    height = scan_data.height
    base_y = scan_data.base_y
    
    table: List[List[Any]] = []
    if args.showy:
        table.append(['Y'] + list(range(base_y, base_y + height)))

    if blocks_shown is not None:
        for show_info in blocks_shown:
            display_name = show_info.display_name or show_info.selectors
            row: List[Any] = [display_name]
            try:
                hist = sum_blocks_selection(scan_data, split_to_blockstates(show_info.selectors))
                print(f'{display_name} = {hist.sum()} blocks')
                row.extend(hist)
            except ValueError:
                log(f'Found no blocks matching selectors: {show_info.selectors}')
                row.extend([0] * height)
            table.append(row)
    else:
        for name, idx in scan_data.blockstate_to_idx.items():
            row = [name]
            row.extend(scan_data.histogram[:, idx])
            table.append(row)

    if args.bylayer:
        for graph in zip(*table):
            print(*graph, sep=',')
    else:
        for graph in table:
            print(*graph, sep=',')


def operation_load_and_process(args: argparse.Namespace) -> None:
    if args.datasrc == 'load':
        scan_data = load_scan(args.statfile)
    elif args.datasrc == 'extract':
        scan_limit = args.limit or 2**32
        bounds = (0, 128)
        if args.layers:
            bounds = args.layers
            if bounds[0] >= bounds[1]:
                raise InvalidScannerOperation('Bounds must be a nonempty range')
        scan_data = create_scan(save_path=args.world, dim_id=args.dim, scan_limit=scan_limit, bounds=bounds)
    else:
        raise InvalidScannerOperation('Unknown scan data source')
    
    # Save new histogram even if it isn't a goal
    if args.datasrc != 'load' and args.vismode != 'save':
        t = threading.Thread(target=save_scan_data, args=('.last_scan.dat', scan_data.copy()))
        t.start()

    if args.vismode == 'print':
        visualize_print(args, scan_data)
    elif args.vismode == 'plot':
        visualize_plot(args, scan_data)
    elif args.vismode == 'csv':
        vis_print_as_csv(args, scan_data)
    elif args.vismode == 'save':
        operation_save(args, scan_data)
    else:
        raise InvalidScannerOperation('Unknown scan data processing action')


def main() -> None:
    def parse_dashed_range(s: str) -> Tuple[int, int]:
        """Parse numeric ranges like 123-456"""
        m = re.match(r'^(-?\d+)[-~/:](-?\d+)$', s, re.ASCII)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        raise ValueError('Not a valid range')


    def add_operation_parsers(parser: argparse.ArgumentParser) -> None:
        subparsers = parser.add_subparsers(dest='vismode', required=True, help='Visualization mode')
        
        parse_vis_print = subparsers.add_parser('print', help='Print total block statistics')
        parse_vis_print.add_argument('--sort', choices=['id', 'count', 'name'], default='count', help='Display ordering')
        parse_vis_print.add_argument('--sumstates', action='store_true', help='Sum counts of different states of a single block into one')
        
        parse_vis_plot = subparsers.add_parser('plot', help='Plot histogram of block distribution')
        parse_vis_plot.add_argument('select', nargs='+', help='Any number of selector arguments')
        parse_vis_plot.add_argument('--norm', choices=['none', 'base', 'total', 'chunk'], default='total', help='Histogram value normalization (default is total)')
        parse_vis_plot.add_argument('--normbase', default='minecraft:stone+minecraft:deepslate', help='Blockstate name (or names separated with "+") for relative normalization (used with --norm base)')
        parse_vis_plot.add_argument('--solids', action='store_true', help='Display graph for total non-air blocks')
        parse_vis_plot.add_argument('--cumulative', action='store_true', help='Display as cumulative graph')
        parse_vis_plot.add_argument('--layers', type=parse_dashed_range, default=None, help='Vertical range to display (default is full range)')
        parse_vis_plot.add_argument('--savefig', default=None, help='Save plot to specified file instead of displaying in a window')

        parse_vis_csv = subparsers.add_parser('csv', help='Print selected block histograms as CSV')
        parse_vis_csv.add_argument('select', nargs='+', help='Any number of selector arguments or one argument "*" to output all blocks')
        parse_vis_csv.add_argument('--layers', type=parse_dashed_range, default=None, help='Vertical range to use')
        parse_vis_csv.add_argument('--bylayer', action='store_true', help='Print counts for each Y layer on a separate line (default is line per selector)')
        parse_vis_csv.add_argument('--showy', action='store_true', help='Add Y level column to CSV')

        parse_save_scan = subparsers.add_parser('save', help='Save extracted block histogram')
        parse_save_scan.add_argument('output', help='Name of the new histogram file')


    parser = argparse.ArgumentParser(description='Create a histogram of block distribution by height of a Minecraft world')
    action_parsers = parser.add_subparsers(dest='datasrc', required=True, help='Data source')

    parse_load = action_parsers.add_parser('load', help='Load and display block histogram from file')
    parse_load.add_argument('statfile', help='Path of histogram file to display')
    add_operation_parsers(parse_load)

    parse_direct = action_parsers.add_parser('extract', help='Calculate block histogram for chosen world')
    parse_direct.add_argument('world', help='Path to world directory')
    parse_direct.add_argument('--dim', default=0, type=int, help='Numerical ID of target dimension (default is Overworld)')
    parse_direct.add_argument('--limit', type=int, default=None, help='Number of chunks to scan (default is all)')
    parse_direct.add_argument('--layers', type=parse_dashed_range, default=None, help='Vertial scan limits (default is 0-128)')
    add_operation_parsers(parse_direct)

    args = parser.parse_args()
    try:
        operation_load_and_process(args)
    except InvalidScannerOperation as err:
        log(err)


if __name__ == '__main__':
    main()
