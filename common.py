# Refactoring / new feature list:
# - Fully update to blockstate-based system
# - Compress results of old scanners (must keep select-by-id!)

import os
import uNBT as nbt
import numpy as np
import json
import re
from typing import Tuple, Optional, Dict


BlockState = Tuple[str, Optional[int]]
BlockStateNew = Tuple[str, Optional[Dict[str, str]]]
CompactState = str
BlockMappingType = Dict[CompactState, int]
BlockMappingInverseType = Dict[int, CompactState]


def resource_location(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), name)


def load_old_blockid_mapping() -> BlockMappingType:
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

def get_blockid_mapping(save_path: str) -> BlockMappingType:
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

def parse_blockstate_name_new(bs_name: CompactState) -> BlockStateNew:
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
    
    if m.group(3).isnumeric():
        # Prettier interface for old metadata-based blockstates
        # Example: minecraft:wool[3]
        state = {BS_META: m.group(3)}
    else:
        state = _parse_cs_dict(m.group(3)) if m.group(2) else None
    return name, state

def format_blockstate_new(blockstate: BlockStateNew) -> CompactState:
    name, data = blockstate
    if data is None:
        return name
    if BS_META in data:
        # Prettier formatting for old metadata-based blockstates
        return f'{name}[{data[BS_META]}]'
    return f'{name}[{",".join(f"{k}={v}" for k, v in data.items())}]'


def parse_blockstate_name(bs_name: CompactState) -> BlockState:
    m = re.match(r'^([\w:]+)(\.(\d+))?$', bs_name, re.ASCII)
    if not m:
        raise ValueError('Invalid blockstate name')
    meta = None
    if m.group(3):
        meta = int(m.group(3))
    name = m.group(1)
    return name, meta

def format_blockstate(blockstate: BlockState) -> CompactState:
    name, meta = blockstate
    return f'{name}{f".{meta}" if meta is not None else ""}'


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
                 blockstate_to_idx: BlockMappingType, 
                 base_y: int):
        self.histogram = histogram # [height][state]
        self.chunks_scanned = chunks_scanned
        self.blockstate_to_idx = blockstate_to_idx
        self.base_y = base_y

        if self.state_count % 16 != 0:
            raise ValueError('All IDs need 16 states')

    @property
    def height(self) -> int:
        return self.histogram.shape[0]

    @property
    def state_count(self) -> int:
        return self.histogram.shape[1]
    
    @property
    def idx_to_blockstate(self) -> BlockMappingInverseType:
        backmap = {}
        for state, idx in self.blockstate_to_idx.items():
            backmap[idx] = state
        return backmap

    def get_block_hist(self, blockstate: BlockState) -> np.ndarray:
        # When `meta` is None - act as wildcard
        name, meta = blockstate
        
        id = None
        if name.isnumeric():
            # support old system of directly choosing block ids
            id = int(name)
        else:
            if ':' not in name:
                found = None
                default = 'minecraft:' + name
                if default in self.blockstate_to_idx:
                    found = default
                else:
                    for fullname in self.blockstate_to_idx.keys():
                        if fullname.endswith(':' + name):
                            found = fullname
                            break
                if found:
                    print(f'No namespace specified for {name}, using {found}')
                    name = found
            if name in self.blockstate_to_idx:
                id = self.blockstate_to_idx[name]
        
        if id is None:
            raise ValueError(f'Mapping {format_blockstate(blockstate)} not found')
        bid = (id << 4) | ((meta or 0) & 15)

        if meta is not None:
            return self.histogram[:,bid]
        else:
            return self.histogram[:,bid:bid+16].sum(axis=1)
    
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
        })
    )


def convert_to_new_bs_format(block_counts: np.ndarray, blockstate_to_idx: BlockMappingType) -> Tuple[np.ndarray, BlockMappingType]:
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
    
    new_mapping = {}
    for new_sid, old_sid in enumerate(nonzero_sids):
        id = old_sid // 16
        meta = old_sid % 16
        name = blockstate_to_idx[id]
        blockstate = format_blockstate((name, {BS_META: str(meta)}))
        new_mapping[blockstate] = new_sid

    return block_counts, new_mapping
