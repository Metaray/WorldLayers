import os
import uNBT as nbt
import numpy as np
import json
import re
from collections import defaultdict
from typing import List, Tuple, Optional, Dict


BlockState = Tuple[str, Optional[Dict[str, str]]]
CompactState = str
BlockMapping = Dict[CompactState, int]
BlockMappingInverse = Dict[int, CompactState]


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


BS_META = '_'
def _parse_cs_dict(s: str) -> Dict[str, str]:
    """Parse comma separated key=value dict"""
    s = s.strip()
    if not s:
        return {}
    if s.isnumeric():
        return {BS_META: s}
    state = {}
    for kv in s.split(','):
        k, v = kv.split('=')
        state[k.strip()] = v.strip()
    return state

def parse_blockstate_name(bs_name: CompactState) -> BlockState:
    """Parse things like "minecraft:thing[facing=south,burning=true]" """
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
            # Prettier interface for old metadata-based blockstates
            # Example: minecraft:wool[3]
            state = {BS_META: m.group(3)}
        else:
            state = _parse_cs_dict(m.group(3))
    else:
        state = None
    return name, state

def format_blockstate(blockstate: BlockState) -> CompactState:
    name, data = blockstate
    if data is None:
        return name
    if BS_META in data:
        # Prettier formatting for old metadata-based blockstates
        return f'{name}[{data[BS_META]}]'
    return f'{name}[{",".join(f"{k}={data[k]}" for k in sorted(data.keys()))}]'


def parse_dashed_range(s: str) -> Tuple[int, int]:
    """Parse numeric ranges like 123-456"""
    m = re.match(r'^(-?\d+)[-~/](-?\d+)$', s, re.ASCII)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError('Not a valid range')


class DimScanData:
    """Class holding collected data and (useful) parameters of a scan"""

    def __init__(self, 
                 histogram: np.ndarray, 
                 chunks_scanned: int, 
                 blockstate_to_idx: BlockMapping, 
                 base_y: int):
        self.histogram = histogram # [height][state]
        self.chunks_scanned = chunks_scanned
        self.blockstate_to_idx = blockstate_to_idx
        self.base_y = base_y

        self.name_to_blockstates: Dict[str, List[CompactState]] = defaultdict(list)
        for blockstate in blockstate_to_idx.keys():
            i = blockstate.find('[')
            name = blockstate[:i] if i != -1 else blockstate
            self.name_to_blockstates[name].append(blockstate)

    @property
    def height(self) -> int:
        return self.histogram.shape[0]

    @property
    def state_count(self) -> int:
        return self.histogram.shape[1]
    
    @property
    def idx_to_blockstate(self) -> BlockMappingInverse:
        return {idx: state for state, idx in self.blockstate_to_idx.items()}

    def get_block_hist(self, blockstate: BlockState) -> np.ndarray:
        # When `props` is None - act as wildcard
        name, props = blockstate
        
        # Support old system of directly choosing block ids
        if name.isnumeric():
            return self.histogram[:, int(name)]
        
        # Add default namespace if there is none
        if ':' not in name:
            name = 'minecraft:' + name
        
        if name in self.name_to_blockstates:
            if props is not None:
                sname = format_blockstate(blockstate)
                if sname in self.blockstate_to_idx:
                    idx = self.blockstate_to_idx[sname]
                    return self.histogram[:, idx]
            
            else:
                idxs = [
                    self.blockstate_to_idx[sname]
                    for sname in self.name_to_blockstates[name]
                ]
                return self.histogram[:, idxs].sum(axis=1)
        
        raise ValueError(f'Mapping {format_blockstate(blockstate)} not found')
    
    def try_get_block_hist(self, blockstate: BlockState) -> np.ndarray:
        try:
            return self.get_block_hist(blockstate)
        except ValueError:
            return np.zeros(self.height, dtype=self.histogram.dtype)
    
    def crop_histogram(self, bounds: Tuple[int, int]) -> None:
        """In-place cropping of histogram height bounds"""
        y_low, y_hi = bounds
        new_hei = y_hi - y_low
        if y_low >= y_hi:
            raise ValueError('Invalid cropping range')
        if y_low < self.base_y or new_hei > self.height:
            raise ValueError('Cropping range doesn\'t match available data')
        
        if y_low == self.base_y and new_hei == self.height:
            return
        self.histogram = self.histogram[y_low-self.base_y:y_hi-self.base_y]
        self.base_y = y_low


def load_scan(path: str) -> DimScanData:
    data = nbt.read_nbt_file(path)

    version = data.get('Version', 3)
    
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
        block_counts, blockstate_to_idx = convert_to_new_bs_format(block_counts, blockstate_to_idx)

    print(f'Loaded data for {chunks_scanned} chunks, Y levels {base_y} ~ {base_y + HEI_LIM}')
    return DimScanData(
        histogram=block_counts,
        chunks_scanned=chunks_scanned,
        blockstate_to_idx=blockstate_to_idx,
        base_y=base_y
    )


def save_scan_data(extract_file: str, scan_data: DimScanData) -> None:
    nbt.write_nbt_file(
        extract_file,
        nbt.TagCompound({
            'ChunkCount' : nbt.TagLong(scan_data.chunks_scanned),
            'StateCount' : nbt.TagInt(scan_data.state_count),
            'ScanHeight' : nbt.TagInt(scan_data.height),
            'BaseY' : nbt.TagInt(scan_data.base_y),
            'DataType' : nbt.TagString(scan_data.histogram.dtype.name),
            'BlockMapping' : nbt.TagCompound({
                name : nbt.TagInt(code)
                for name, code in scan_data.blockstate_to_idx.items()
            }),
            'Data' : nbt.TagByteArray(scan_data.histogram.tobytes()),
            'Version' : nbt.TagInt(4),
        })
    )


def convert_to_new_bs_format(block_counts: np.ndarray, blockstate_to_idx: BlockMapping) -> Tuple[np.ndarray, BlockMapping]:
    state_count = block_counts.shape[1]

    # Keep air at 0
    # Discard mappings for blocks which weren't found
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
    for new_sid, old_sid in enumerate(nonzero_sids):
        id = old_sid // 16
        meta = old_sid % 16
        name = idx_to_bs[id]
        blockstate = format_blockstate((name, {BS_META: str(meta)}))
        new_mapping[blockstate] = new_sid

    return block_counts, new_mapping
