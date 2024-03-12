from pathlib import Path
import numpy as np
import uNBT as nbt
from typing import Iterable
from ..common import log


CTR_DTYPE = np.int64 # Histogram counter type


def iter_world_chunks(save_path: str, dim_id: int, world_type: str) -> Iterable[nbt.TagCompound]:
    if world_type == 'alpha':
        # Alpha format has same data as Region, but chunks are stored in individual files
        # Numbers are base36 encoded
        # 2 folder layers: first is X % 64, second is Z % 64
        # Chunk file name: c.X.Y.dat
        for file in Path(save_path).glob('*/*/c.*.*.dat'):
            yield nbt.read_nbt_file(str(file))
    
    else:
        for rfile_info in nbt.enumerate_world(save_path, world_type)[dim_id]:
            log('Scanning', rfile_info.path)
            region = nbt.Region.from_file(rfile_info.path)
            for chunk in region.iter_nonempty():
                yield chunk.nbt
