import argparse
import threading
import re
from common import InvalidScannerOperation, log
from typing import Tuple, NamedTuple
from visualizations.matplotlib_plot import VisPlotArguments, VisStackPlotArguments, visualize_mass_stack_plot, visualize_plot
from visualizations.print_readable import VisPrintArguments, visualize_print
from visualizations.print_as_csv import VisPrintAsCsvArguments, vis_print_as_csv
from worldlayers.common import DimScanData, save_scan_data, load_scan
from worldlayers.scanners import create_scan


class OpSaveArguments(NamedTuple):
    output: str


def operation_save(args: OpSaveArguments, scan_data: DimScanData) -> None:
    extract_file = args.output
    save_scan_data(extract_file, scan_data)
    log(f'Saved block counts to {extract_file}')


def operation_load_and_process(args: argparse.Namespace) -> None:
    data_src: str = args.datasrc
    vis_mode: str = args.vismode

    if data_src == 'load':
        scan_data = load_scan(args.statfile)
    elif data_src == 'extract':
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
    if data_src != 'load' and vis_mode != 'save':
        t = threading.Thread(target=save_scan_data, args=('.last_scan.dat', scan_data.copy()))
        t.start()

    if vis_mode == 'print':
        visualize_print(
            VisPrintArguments(
                layers=args.layers,
                sumstates=args.sumstates,
                sort=args.sort,
            ),
            scan_data
        )
    elif vis_mode == 'plot':
        visualize_plot(
            VisPlotArguments(
                layers=args.layers,
                select=args.select,
                norm=args.norm,
                normbase=args.normbase,
                cumulative=args.cumulative,
                solids=args.solids,
                savefig=args.savefig,
            ),
            scan_data
        )
    elif vis_mode == 'stackplot':
        visualize_mass_stack_plot(
            VisStackPlotArguments(
                log_scale=args.logscale,
                sum_states=not args.nosum,
            ),
            scan_data
        )
    elif vis_mode == 'csv':
        vis_print_as_csv(
            VisPrintAsCsvArguments(
                layers=args.layers,
                select=args.select,
                showy=args.showy,
                bylayer=args.bylayer,
            ),
            scan_data
        )
    elif vis_mode == 'save':
        operation_save(OpSaveArguments(output=args.output), scan_data)
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
        parse_vis_print.add_argument('--layers', type=parse_dashed_range, default=None, help='Vertical range to use (default is full range)')
        
        parse_vis_plot = subparsers.add_parser('plot', help='Plot histogram of block distribution')
        parse_vis_plot.add_argument('select', nargs='+', help='Any number of selector arguments')
        parse_vis_plot.add_argument('--norm', choices=['none', 'base', 'total', 'chunk'], default='total', help='Histogram value normalization (default is total)')
        parse_vis_plot.add_argument('--normbase', default='minecraft:stone+minecraft:deepslate', help='Blockstate name (or names separated with "+") for relative normalization (used with --norm base)')
        parse_vis_plot.add_argument('--solids', action='store_true', help='Display graph for total non-air blocks')
        parse_vis_plot.add_argument('--cumulative', action='store_true', help='Display as cumulative graph')
        parse_vis_plot.add_argument('--layers', type=parse_dashed_range, default=None, help='Vertical range to display (default is full range)')
        parse_vis_plot.add_argument('--savefig', default=None, help='Save plot to specified file instead of displaying in a window')

        vis_stack_plot = subparsers.add_parser('stackplot', help='Plot everything as stack plot')
        vis_stack_plot.add_argument('--logscale', action='store_true', help='Use logarithmic scale for counts')
        vis_stack_plot.add_argument('--nosum', action='store_true', help='Don\'t sum up distinct states of a block')

        parse_vis_csv = subparsers.add_parser('csv', help='Print selected block histograms as CSV')
        parse_vis_csv.add_argument('select', nargs='*', help='Any number of selector arguments or none to output all blocks')
        parse_vis_csv.add_argument('--layers', type=parse_dashed_range, default=None, help='Vertical range to output (default is full range)')
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
