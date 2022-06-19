import argparse
import traceback
import threading
from typing import List, NamedTuple, Optional
from scanners import *


class InvalidScannerOperation(Exception):
    pass


def operation_save(args: argparse.Namespace, scan_data: DimScanData) -> None:
    extract_file = args.output
    save_scan_data(extract_file, scan_data)
    print(f'Saved block counts to {extract_file}')


def visualize_print(args: argparse.Namespace, scan_data: DimScanData) -> None:
    block_counts = scan_data.histogram
    chunks_scanned = scan_data.chunks_scanned
    idx_to_blockstate = scan_data.idx_to_blockstate
    HEI_LIM = scan_data.height
    STATE_LIM = scan_data.state_count
    
    totals = block_counts.sum(axis=0)
    volume = chunks_scanned * (HEI_LIM * 16**2)

    def show_count(name: str, count: int) -> None:
        parts = [
            f'{name} = {count}',
            f'({count / volume:%})',
            f'({count / chunks_scanned:.4f} b/ch)',
        ]
        print(*parts, sep='\t')

    nonzero = [(totals[idx], idx) for idx in range(16, STATE_LIM) if totals[idx] > 0]
    if args.sort == 'count':
        # 'count' descending
        nonzero.sort(key=lambda ci: -ci[0])
    else:
        # 'id' ascending
        nonzero.sort(key=lambda ci: ci[1])
    
    show_count('Nonair blocks', totals[16:].sum())
    for count, idx in nonzero:
        bid, meta = idx // 16, idx % 16
        if bid in idx_to_blockstate:
            name = idx_to_blockstate[bid]
        else:
            name = str(bid)
        show_count(format_blockstate((name, meta)), count)


class DisplayBlockSelector(NamedTuple):
    # Numeric block id or string name [ + '.' + numeric metadata / None for sum of all values ]
    # Examples:
    #   minecraft:stone
    #   minecraft:log.2
    #   123.6
    names: str
    # Matplotlib colors / None for auto color
    color: Optional[str]
    # Legend name override / None - equal to `names`
    display_name: Optional[str]

def parse_block_selection(selectors: List[str]) -> List[DisplayBlockSelector]:
    blocks_shown = []
    for line in map(str.strip, selectors):
        if not line or line.startswith('#'):
            continue
        
        parts = line.split(',')
        while len(parts) < 3:
            parts.append('')
        
        blocks_shown.append(DisplayBlockSelector(
            names=parts[0],
            color=parts[1] if parts[1] else None,
            display_name=parts[2] if parts[2] else None,
        ))
    return blocks_shown

def load_block_selection(files: str) -> List[DisplayBlockSelector]:
    lines = []
    for part in files.split(';'):
        if os.path.exists(part):
            with open(part) as f:
                lines.extend(f.readlines())
        else:
            lines.append(part) # treat non-files as seletors for convinience
    return parse_block_selection(lines)


def visualize_plot(args: argparse.Namespace, scan_data: DimScanData) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    
    if args.layers:
        scan_data.crop_histogram(args.layers)
    block_counts = scan_data.histogram
    chunks_scanned = scan_data.chunks_scanned
    HEI_LIM = scan_data.height

    show_total_nonair = args.solids
    cumulative = args.cumulative
    dump_csv = args.dumpcsv # TODO: do textual data output properly

    blocks_shown = load_block_selection(args.select)

    if args.norm == 'base':
        # Normalize by base block - relative fraction
        normalize_base = ('minecraft:stone', 0) # TODO: make this configurable
        base_mx = np.maximum(scan_data.get_block_hist(normalize_base), 1)
        norm_func = lambda x: x / base_mx
        y_label = f'Fraction relative to {format_blockstate(normalize_base)}'
    elif args.norm == 'total':
        # Normalize by total volume - chance per block
        norm_func = lambda x: x / (16**2 * chunks_scanned)
        y_label = 'Fraction per block'
    elif args.norm == 'chunk':
        # Normalize by number of chunks - blocks per chunk layer
        norm_func = lambda x: x / chunks_scanned
        y_label = 'Blocks per chunk layer'
    elif args.norm == 'none':
        norm_func = lambda x: x
        y_label = 'Total count per layer'
    else:
        raise InvalidScannerOperation('Invalid normalization mode')

    if cumulative:
        # TODO: Use plt.stackplot()
        blocks_shown.reverse()
        acc = np.zeros(HEI_LIM, np.float64)
    
    fig, ax = plt.subplots(figsize=(8, 5))

    graphs = []
    base_y = scan_data.base_y
    y_range = list(range(base_y, base_y + HEI_LIM))
    
    for show_info in blocks_shown:
        color = show_info.color
        disp_name = show_info.display_name or show_info.names

        hist = np.zeros(HEI_LIM, dtype=CTR_DTYPE)
        for bs_name in show_info.names.split('+'):
            try:
                hist += scan_data.get_block_hist(parse_blockstate_name(bs_name))
            except ValueError as err:
                # Ignore blocks we don't have / have invalid selector
                print(err)
        print(f'{disp_name} = {hist.sum()} blocks')
        
        hist = norm_func(hist)
        if not cumulative:
            graph = hist
        else:
            acc += hist
            graph = acc
        
        if dump_csv:
            graphs.append((graph, disp_name))
        
        if color:
            ax.plot(y_range, graph, color, label=disp_name)
        else:
            ax.plot(y_range, graph, label=disp_name)

    if show_total_nonair and not cumulative:
        graph = norm_func(block_counts[:,1:].sum(axis=1))
        ax.plot(y_range, graph, 'lightgray', label='Non-air')
        if dump_csv:
            graphs.append((graph, 'Non-air'))

    if dump_csv:
        print(','.join(x[1] for x in graphs))
        for graph in graphs:
            print(','.join(map(str, graph[0])))

    ax.grid(True)
    ax.legend()
    ax.set_title('Block distribution by height')
    ax.set_xlabel('Y level')
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 5], integer=True, min_n_ticks=7))
    
    if args.savefig:
        # plt.ylim(0.0, 0.01)
        plt.savefig(args.savefig, dpi=300)
    else:
        plt.show()


def operation_load_and_process(args: argparse.Namespace) -> None:
    if args.action == 'load':
        scan_data = load_scan(args.statfile)
    elif args.action == 'extract':
        scan_limit = args.limit or 2**32
        bounds = (0, 128)
        if args.layers:
            bounds = args.layers
            if bounds[0] >= bounds[1]:
                raise InvalidScannerOperation('Bounds must be a nonempty range')
        scan_data = create_scan(save_path=args.world, dim_id=args.dim, scan_limit=scan_limit, bounds=bounds)
    else:
        raise InvalidScannerOperation('Unknown scan load action')
    
    # Save new histogram even if it isn't a goal
    if args.action == 'extract' and args.vismode != 'save':
        t = threading.Thread(target=save_scan_data, args=('.last_scan.dat', scan_data))
        t.start()

    if args.vismode == 'print':
        visualize_print(args, scan_data)
    elif args.vismode == 'plot':
        visualize_plot(args, scan_data)
    elif args.vismode == 'save':
        operation_save(args, scan_data)
    else:
        raise InvalidScannerOperation('Unknown scan data processing action')



def add_operation_parsers(parser: argparse.ArgumentParser) -> None:
    subparsets = parser.add_subparsers(dest='vismode', required=True, help='Visualization choice')
    
    parse_vis_print = subparsets.add_parser('print', help='Write block counts to console')
    parse_vis_print.add_argument('--sort', choices=['id', 'count'], default='count', help='Block counts ordering')
    
    parse_vis_plot = subparsets.add_parser('plot', help='Plot histogram of block distribution')
    parse_vis_plot.add_argument('select', help='Names of files containing selectors separated with "+"')
    parse_vis_plot.add_argument('--norm', choices=['none', 'base', 'total', 'chunk'], default='total', help='Block counts ordering')
    parse_vis_plot.add_argument('--solids', action='store_true', help='Display plot for non-air blocks')
    parse_vis_plot.add_argument('--cumulative', action='store_true', help='Display as cumulative graph')
    parse_vis_plot.add_argument('--dumpcsv', action='store_true', help='Output graph data points as CSV')
    parse_vis_plot.add_argument('--layers', type=parse_dashed_range, default=None, help='Vertical range to display')
    parse_vis_plot.add_argument('--savefig', default=None, help='Save plot to file instead of displaying in a window')

    parse_save_scan = subparsets.add_parser('save', help='Save extracted block histogram')
    parse_save_scan.add_argument('output', help='Name of the new histogram file')

parser = argparse.ArgumentParser(description='Create a histogram of block distribution by height in a Minecraft world')
action_parsers = parser.add_subparsers(dest='action', required=True, help='Action to perform')

parse_load = action_parsers.add_parser('load', help='Load and display block histogram from file')
parse_load.set_defaults(action_func=operation_load_and_process)
parse_load.add_argument('statfile', help='Name of histogram file to display')
add_operation_parsers(parse_load)

parse_direct = action_parsers.add_parser('extract', help='Calculate block histogram for chosen world')
parse_direct.set_defaults(action_func=operation_load_and_process)
parse_direct.add_argument('world', help='Path to world directory')
parse_direct.add_argument('--dim', default=0, type=int, help='ID of target dimension (default is overworld)')
parse_direct.add_argument('--limit', type=int, default=None, help='Number of chunks to scan (default is all)')
parse_direct.add_argument('--layers', type=parse_dashed_range, default=None, help='Vertial scan limits (default is 0-128)')
add_operation_parsers(parse_direct)


args = parser.parse_args()
if hasattr(args, 'action_func'):
    try:
        args.action_func(args)
    except InvalidScannerOperation as err:
        print(err)
    except Exception as err:
        traceback.print_exc()
else:
    raise NotImplementedError(f'Action {args.action} doesn\'t have a handler')
