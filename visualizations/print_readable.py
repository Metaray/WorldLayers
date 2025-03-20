import numpy as np
from common import log
from numpy.typing import NDArray
from typing import Tuple, NamedTuple
from worldlayers.common import CompactState, DimScanData, AIR_BLOCKS, crop_histogram, sum_blocks_selection


class VisPrintArguments(NamedTuple):
    layers: Tuple[int, int] | None
    sumstates: bool
    sort: str


def visualize_print(args: VisPrintArguments, scan_data: DimScanData) -> None:
    if args.layers:
        crop_histogram(scan_data, args.layers)
    
    chunks_scanned = scan_data.chunks_scanned
    histogram = scan_data.histogram
    volume = chunks_scanned * scan_data.height * 16**2

    visdata: list[tuple[int | None, NDArray[np.int64] | int, CompactState]]
    if args.sumstates:
        visdata = [
            (
                None,
                sum(histogram[:, scan_data.blockstate_to_idx[state]] for state in states),
                name,
            )
            for name, states in scan_data.name_to_blockstates.items()
        ]
    else:
        idx_to_blockstate = scan_data.idx_to_blockstate
        visdata = [(i, histogram[:, i], idx_to_blockstate[i]) for i in range(scan_data.state_count)]

    if args.sort == 'count':
        visdata.sort(key=lambda ci: -np.sum(ci[1]))  # sort by block count descending
    elif args.sort == 'name':
        visdata.sort(key=lambda ci: ci[2])  # sort by blockstate lexicographically
    else:
        visdata.sort(key=lambda ci: ci[0])  # sort by state id ascending (legacy)
    
    def important_range(hist: NDArray[np.int64]) -> Tuple[int, int]:
        # Exact range block was present
        # nonzero = np.argwhere(hist)
        # ymin = scan_data.base_y + nonzero[0][0]
        # ymax = scan_data.base_y + nonzero[-1][0]
        # return ymin, ymax

        # Range where most of the blocks were found (remove trails)
        # Removes 1% of blocks from lowest and highest Y levels
        hacc = np.cumsum(hist)
        mincount = hacc[-1] * 0.99
        l, r = 0, len(hacc) - 1
        while l < r:
            left_sum = hacc[r] - hacc[l]
            right_sum = hacc[r - 1] - hacc[l] + hist[l]
            if left_sum > right_sum:
                if left_sum < mincount:
                    break
                l += 1
            else:
                if right_sum < mincount:
                    break
                r -= 1
        return scan_data.base_y + l, scan_data.base_y + r
    
    def show_count(name: str, hist: NDArray[np.int64]) -> None:
        count = np.sum(hist)
        ymin, ymax = important_range(hist)
        print(
            f'{name} = {count}',
            f'({count / volume:%})',
            f'({count / chunks_scanned:.4f} b/ch)',
            f'(Y {ymin} ~ {ymax})',
            sep='\t',
        )

    log(f'{len(scan_data.name_to_blockstates)} unique blocks')
    log(f'{scan_data.state_count} block states')

    air_hist = sum_blocks_selection(scan_data, AIR_BLOCKS)
    show_count('Nonair blocks', histogram.sum(axis=1) - air_hist)

    for _, hist, name in visdata:
        if np.sum(hist) > 0:
            show_count(name, hist)
