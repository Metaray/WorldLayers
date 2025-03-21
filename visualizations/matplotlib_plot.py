import numpy as np
from common import log, split_to_blockstates, load_block_selection
from numpy.typing import NDArray
from typing import List, NamedTuple, Optional, Tuple
from worldlayers.common import DimScanData, AIR_BLOCKS, crop_histogram, sum_block_states, sum_blocks_selection


class VisPlotArguments(NamedTuple):
    layers: Tuple[int, int] | None
    select: list[str]
    norm: str
    normbase: str
    cumulative: bool
    solids: bool
    savefig: bool


def visualize_plot(args: VisPlotArguments, scan_data: DimScanData) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    
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


class VisStackPlotArguments(NamedTuple):
    log_scale: bool
    sum_states: bool


def visualize_mass_stack_plot(args: VisStackPlotArguments, scan_data: DimScanData) -> None:
    import matplotlib.pyplot as plt

    if args.sum_states:
        scan_data = sum_block_states(scan_data)

    base_y = scan_data.base_y
    height = scan_data.height
    y_range = list(range(base_y, base_y + height))

    layers = np.swapaxes(scan_data.histogram, 0, 1)
    idxs = np.arange(layers.shape[0])
    
    select = np.sum(layers, axis=1) > 0
    layers = layers[select,:]
    idxs = idxs[select]
    
    order = np.argsort(np.sum(layers, axis=1))
    layers = layers[order,:]
    idxs = idxs[order]

    idx_map = scan_data.idx_to_blockstate
    labels = [idx_map[i] for i in idxs]

    print(len(labels))

    plt.stackplot(y_range, layers, labels=labels)

    if args.log_scale:
        plt.yscale('log')
    
    plt.grid(True)
    # plt.legend()
    plt.show()
