import os
import uNBT as nbt
import numpy as np
import json
import re
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Union


BlockState = Tuple[str, Optional[Dict[str, str]]]
CompactState = str
BlockMapping = Dict[CompactState, int]
BlockMappingInverse = Dict[int, CompactState]
BlockSelector = Union[str, BlockState]


AIR_BLOCKS = ('minecraft:air', 'minecraft:cave_air', 'minecraft:void_air')


def resource_location(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), name)


def load_old_blockid_mapping() -> BlockMapping:
    """Load static mapping for versions <1.13"""
    # Pretty sure ids didn't change since alpha (maybe even infdev)
    with open(resource_location('ids.json')) as id_file:
        ids = json.load(id_file)
    mapping = {}
    for block_info in ids['blocks']:
        if block_info['data'] == 0:
            # Use name of base block (meta=0)
            name = block_info['text_id']
            id = block_info['id']
            mapping[name] = id
    return mapping


def get_blockid_mapping(save_path: str) -> BlockMapping:
    """Create mapping to and from symbolic block names"""
    try:
        root = nbt.read_nbt_file(os.path.join(save_path, 'level.dat'))
    except FileNotFoundError:
        root = None
    
    if root and 'FML' in root:
        # Forge stores all ids in level.dat
        mapping = {}
        for block_info in root['FML']['Registries']['minecraft:blocks']['ids']:
            name = block_info['K'].value
            id = block_info['V'].value
            mapping[name] = id
        print(f'Got Forge block id mapping. {len(mapping)} entries.')
    
    else:
        # For normal minecraft save load from prepared file
        mapping = load_old_blockid_mapping()
        print(f'No mapping, using fallback. {len(mapping)} entries.')

    return mapping


_OLD_META_PROPERTY = '_'

def _parse_cs_dict(s: str) -> Dict[str, str]:
    """Parse comma separated key=value dict"""
    s = s.strip()
    if not s:
        return {}
    state = {}
    for kv in s.split(','):
        k, v = kv.split('=')
        state[k.strip()] = v.strip()
    return state


def parse_blockstate(bs_name: CompactState) -> BlockState:
    """Parse compact blockstate representation

    Examples:
    - namespace:something
    - minecraft:thing[facing=south,waterlogged=true]
    - old_style_name_with_metadata[123]
    - 240[1]
    """
    m = re.match(r'^(.+?)(\[(.*)\])?$', bs_name, re.ASCII)
    if not m:
        raise ValueError('Invalid blockstate')
    
    name = m.group(1)
    if ':' in name:
        if not re.match(r'^[0-9a-zA-Z._-]+?:[0-9a-zA-Z._/\:-]+$', name) or '::' in name:
            raise ValueError('Invalid blockstate name')
    else:
        if not re.match(r'^[0-9a-zA-Z._/\-]+$', name):
            raise ValueError('Invalid blockstate name')
    
    if m.group(2):
        if m.group(3).isnumeric():
            state = {_OLD_META_PROPERTY: m.group(3)}
        else:
            state = _parse_cs_dict(m.group(3))
    else:
        state = None
    return name, state


def serialize_blockstate(blockstate: BlockState) -> CompactState:
    name, data = blockstate
    if data is None:
        return name
    if _OLD_META_PROPERTY in data:
        return f'{name}[{data[_OLD_META_PROPERTY]}]'
    return f'{name}[{",".join(f"{k}={data[k]}" for k in sorted(data.keys()))}]'


def parse_dashed_range(s: str) -> Tuple[int, int]:
    """Parse numeric ranges like 123-456"""
    m = re.match(r'^(-?\d+)[-~/](-?\d+)$', s, re.ASCII)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError('Not a valid range')


class DimScanData:
    """Class holding collected data and (useful) parameters of a scan"""

    def __init__(
        self,
        histogram: np.ndarray,
        chunks_scanned: int,
        blockstate_to_idx: BlockMapping,
        base_y: int,
        old_to_new: Optional[Dict[int, CompactState]] = None,
    ):
        self.histogram = histogram # [height][state]
        self.chunks_scanned = chunks_scanned
        self.blockstate_to_idx = blockstate_to_idx
        self.base_y = base_y
        self.old_to_new = old_to_new

        self.name_to_blockstates: Dict[str, List[CompactState]] = defaultdict(list)
        for blockstate in blockstate_to_idx.keys():
            i = blockstate.find('[')
            name = blockstate[:i] if i != -1 else blockstate
            self.name_to_blockstates[name].append(blockstate)
    
    def copy(self) -> 'DimScanData':
        return DimScanData(
            self.histogram,
            self.chunks_scanned,
            self.blockstate_to_idx,
            self.base_y,
            self.old_to_new,
        )

    @property
    def height(self) -> int:
        return self.histogram.shape[0]

    @property
    def state_count(self) -> int:
        return self.histogram.shape[1]
    
    @property
    def idx_to_blockstate(self) -> BlockMappingInverse:
        return {idx: state for state, idx in self.blockstate_to_idx.items()}
    
    @property
    def zero_histogram(self) -> np.ndarray:
        return np.zeros(self.height, self.histogram.dtype)


def get_block_hist(scan: DimScanData, blockstate: BlockSelector) -> np.ndarray:
    if isinstance(blockstate, tuple):
        name, props = blockstate
    else:
        name, props = blockstate, None
    
    # Support old system of directly choosing block ids
    if name.isnumeric():
        if scan.old_to_new is None:
            raise ValueError("This scan doesn't support old ID selectors")
        
        bid = int(name)
        if props is not None:
            if _OLD_META_PROPERTY not in props:
                raise ValueError('Not an old metadata selector')
            meta = int(props[_OLD_META_PROPERTY])
            sids = [bid * 16 + meta]
        else:
            sids = [bid * 16 + meta for meta in range(16)]
        
        hist = scan.zero_histogram
        found_any = False
        for sid in sids:
            if sid in scan.old_to_new:
                hist += scan.histogram[:, scan.blockstate_to_idx[scan.old_to_new[sid]]]
                found_any = True
        
        if found_any:
            return hist
        else:
            raise ValueError(f'Mapping {serialize_blockstate(blockstate)} not found')
    
    # Add default namespace if there is none
    if ':' not in name:
        name = 'minecraft:' + name
    
    if name in scan.name_to_blockstates:
        if props is not None:
            sname = serialize_blockstate(blockstate)
            if sname in scan.blockstate_to_idx:
                idx = scan.blockstate_to_idx[sname]
                return scan.histogram[:, idx]
        
        else:
            idxs = [
                scan.blockstate_to_idx[sname]
                for sname in scan.name_to_blockstates[name]
            ]
            return scan.histogram[:, idxs].sum(axis=1)
    
    raise ValueError(f'Mapping {serialize_blockstate(blockstate)} not found')


def try_get_block_hist(scan: DimScanData, blockstate: BlockSelector) -> np.ndarray:
    try:
        return get_block_hist(scan, blockstate)
    except ValueError:
        return scan.zero_histogram


def sum_blocks_selection(scan: DimScanData, selectors: List[BlockSelector]) -> np.ndarray:
    found_any = False
    hist = scan.zero_histogram
    for selector in selectors:
        try:
            hist += get_block_hist(scan, selector)
            found_any = True
        except ValueError:
            pass
    if not found_any:
        raise ValueError
    return hist


def crop_histogram(scan: DimScanData, bounds: Tuple[int, int]) -> None:
    """In-place cropping of histogram height bounds"""
    y_low, y_hi = bounds
    new_hei = y_hi - y_low
    if y_low >= y_hi:
        raise ValueError('Invalid cropping range')
    if y_low < scan.base_y or new_hei > scan.height:
        raise ValueError('Cropping range doesn\'t match available data')
    
    if y_low == scan.base_y and new_hei == scan.height:
        return
    scan.histogram = scan.histogram[y_low-scan.base_y:y_hi-scan.base_y]
    scan.base_y = y_low


def load_scan(path: str) -> DimScanData:
    data = nbt.read_nbt_file(path)

    # Here are a bunch of checks and fallbacks to support old scan file versions
    if 'Version' in data:
        version = data['Version'].value
    else:
        version = 3
    
    chunks_scanned = data['ChunkCount'].value
    
    # Read IdLimit for backwards compatibility
    if 'IdLimit' in data:
        ID_LIM = data['IdLimit'].value
    else:
        ID_LIM = 2**12
    
    if 'StateCount' in data:
        STATE_LIM = data['StateCount'].value
    else:
        STATE_LIM = ID_LIM * 16
    
    if 'ScanHeight' in data:
        HEI_LIM = data['ScanHeight'].value
    else:
        HEI_LIM = 16 * 8
    
    if 'BaseY' in data:
        base_y = data['BaseY'].value
    else:
        base_y = 0

    dtype = np.dtype(data['DataType'].value) if 'DataType' in data else np.int64
    block_counts = np.frombuffer(data['Data'].value, dtype).reshape((HEI_LIM, STATE_LIM))
    
    blockstate_to_idx = {}
    for name, code in data['BlockMapping'].items():
        blockstate_to_idx[name] = code.value
    
    if version < 4:
        block_counts, blockstate_to_idx, old_to_new = convert_to_new_bs_format(block_counts, blockstate_to_idx)
    else:
        old_to_new = None
        if 'OldIdMapping' in data:
            old_to_new = {
                mapping['old'].value: mapping['new'].value
                for mapping in data['OldIdMapping']
            }

    print(f'Loaded data for {chunks_scanned} chunks, Y levels {base_y} ~ {base_y + HEI_LIM}')
    return DimScanData(
        histogram=block_counts,
        chunks_scanned=chunks_scanned,
        blockstate_to_idx=blockstate_to_idx,
        base_y=base_y,
        old_to_new=old_to_new,
    )


def save_scan_data(extract_file: str, scan_data: DimScanData) -> None:
    tag = nbt.TagCompound({
        'Version': nbt.TagInt(5),
        'ChunkCount': nbt.TagLong(scan_data.chunks_scanned),
        'StateCount': nbt.TagInt(scan_data.state_count),
        'ScanHeight': nbt.TagInt(scan_data.height),
        'BaseY': nbt.TagInt(scan_data.base_y),
        'DataType': nbt.TagString(scan_data.histogram.dtype.name),
        'BlockMapping': nbt.TagCompound({
            name: nbt.TagInt(code)
            for name, code in scan_data.blockstate_to_idx.items()
        }),
        'Data': nbt.TagByteArray(scan_data.histogram.tobytes()),
    })
    if scan_data.old_to_new is not None:
        tag['OldIdMapping'] = nbt.TagList(
            nbt.TagCompound, [
                nbt.TagCompound({
                    'old': nbt.TagInt(old),
                    'new': nbt.TagString(new),
                })
                for old, new in scan_data.old_to_new.items()
            ]
        )
    nbt.write_nbt_file(extract_file, tag)


def convert_to_new_bs_format(
    block_counts: np.ndarray,
    blockstate_to_idx: BlockMapping
) -> Tuple[np.ndarray, BlockMapping, Dict[int, CompactState]]:
    state_count = block_counts.shape[1]

    # Keep air at 0
    # Discard mappings for blocks with nonzero count
    nonzero_sids = []
    nonzero_states = []
    for sid in range(state_count):
        state_hist = block_counts[:,sid:sid+1]
        if state_hist.sum() or sid == 0:
            nonzero_sids.append(sid)
            nonzero_states.append(state_hist)
    block_counts = np.concatenate(nonzero_states, axis=1)
    
    idx_to_bs = {v: k for k, v in blockstate_to_idx.items()}

    new_mapping = {}
    old_to_new = {}
    for new_sid, old_sid in enumerate(nonzero_sids):
        id = old_sid // 16
        meta = old_sid % 16
        name = idx_to_bs[id]
        blockstate = serialize_blockstate((name, {_OLD_META_PROPERTY: str(meta)}))
        new_mapping[blockstate] = new_sid
        old_to_new[old_sid] = blockstate

    return block_counts, new_mapping, old_to_new
