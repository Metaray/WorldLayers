import argparse
from common import log, split_to_blockstates, load_block_selection
from typing import Any, List
from worldlayers.common import DimScanData, crop_histogram, sum_blocks_selection


def vis_print_as_csv(args: argparse.Namespace, scan_data: DimScanData) -> None:
    if args.layers:
        crop_histogram(scan_data, args.layers)

    blocks_shown = load_block_selection(args.select)

    height = scan_data.height
    base_y = scan_data.base_y
    
    table: List[List[Any]] = []
    if args.showy:
        table.append(['Y'] + list(range(base_y, base_y + height)))

    if blocks_shown:
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
