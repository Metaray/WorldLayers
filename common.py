# Refactoring / new feature list:
# - Fully update to blockstate-based system
# - Compress results of old scanners (must keep select-by-id!)

import os
import uNBT as nbt
import numpy as np
import json
import re
from typing import Tuple, Optional, Union, Dict


BlockMappingType = Dict[str, int]
BlockMappingInverseType = Dict[int, str]
BlockState = Tuple[Union[str, int], Optional[int]]


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
def parse_cs_dict(s: str) -> Dict[str, str]:
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

def parse_blockstate_name_new(bs_name: str) -> Tuple[str, Dict[str, str]]:
    """Parse things like "minecraft:thing[facing=south,burning=true]" """
    # TODO: check name is actually legal
    # - contains only following in namespace: 0123456789abcdefghijklmnopqrstuvwxyz_-.
    # - can contain additionally in name: /\
    # - namespace is optional, default is minecraft:
    m = re.match(r'(.+?)(\[(.+)\])?$', bs_name, re.ASCII)
    if not m:
        raise ValueError('Invalid blockstate name')
    name = m.group(1)
    state = parse_cs_dict(m.group(3) or '')
    return name, state

def parse_blockstate_name(bs_name: str) -> BlockState:
    m = re.match(r'^([\w:]+)(\.(\d+))?$', bs_name, re.ASCII)
    if not m:
        raise ValueError('Invalid blockstate name')
    meta = None
    if m.group(3):
        meta = int(m.group(3))
    name = m.group(1)
    if name.isnumeric():
        return int(name), meta
    else:
        return name, meta

def format_blockstate(blockstate: BlockState) -> str:
    name, meta = blockstate
    return f'{name}.{meta if meta is not None else "*"}'

def parse_dashed_range(s: str) -> Tuple[int, int]:
    """Parse numeric ranges like 123-456"""
    m = re.match(r'^(-?\d+)[-~/](-?\d+)$', s, re.ASCII)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError('Not a valid range')



CTR_DTYPE = np.int64 # Histogram counter type

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
        if isinstance(name, int):
            # support old system of directly choosing block ids
            id = name
        elif isinstance(name, str):
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

    dtype = np.dtype(data['DataType'].value) if 'DataType' in data else CTR_DTYPE
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
            'DataType' : nbt.TagString(CTR_DTYPE.__name__),
            'BlockMapping' : nbt.TagCompound({
                name : nbt.TagInt(code)
                for name, code in scan_data.blockstate_to_idx.items()
            }),
            'Data' : nbt.TagByteArray(scan_data.histogram.tobytes()),
        })
    )


def convert_to_new_format(block_counts: np.ndarray, 
                          blockstate_to_idx: BlockMappingType,
                          chunks_scanned: int,
                          base_y: int
                          ) -> DimScanData:
    raise NotImplementedError
    state_count = block_counts.shape[1]

    nonzero_sids = [0]
    nonzero_states = [block_counts[:,0:1]]
    for sid in range(1, state_count):
        state_hist = block_counts[:,sid:sid+1]
        if state_hist.sum():
            nonzero_sids.append(sid)
            nonzero_states.append(state_hist)
    block_counts = np.concatenate(nonzero_states, axis=1)
    
    # Discards mappings for blocks which weren't found
    new_mapping = {}
    for new_sid, old_sid in enumerate(nonzero_sids):
        id = old_sid // 16
        meta = old_sid % 16
        name = blockstate_to_idx[id]
        blockstate = (name, frozenset({ (BS_META, meta) }))
        new_mapping[blockstate] = new_sid

    return DimScanData(
        histogram=block_counts,
        chunks_scanned=chunks_scanned,
        blockstate_to_idx=new_mapping,
        base_y=base_y
    )
