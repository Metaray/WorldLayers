#include <stdint.h>
#include <stddef.h>

#if defined(_WIN32)
  #define EXPORTED_API __declspec(dllexport)
#else
  #define EXPORTED_API
#endif

#define NIBBLE_LO(x) ((uint32_t)((x) & 0xf))
#define NIBBLE_HI(x) ((uint32_t)(((x) >> 4) & 0xf))


EXPORTED_API void dsci_v0(
    uint64_t *blockCount,
    const uint8_t *blocks,
    const uint8_t *metadata
) {
    const size_t idLim = 256 * 16;

    for (size_t xz = 0; xz < 16 * 16; ++xz) {
        for (size_t y = 0; y < 128; y += 2) {
            size_t idx = y + xz * 128;
            
            uint32_t bid1 = ((uint32_t)blocks[idx + 0] << 4)
                          | NIBBLE_LO(metadata[idx / 2]);
            
            uint32_t bid2 = ((uint32_t)blocks[idx + 1] << 4)
                          | NIBBLE_HI(metadata[idx / 2]);
            
            blockCount[(y + 0) * idLim + bid1]++;
            blockCount[(y + 1) * idLim + bid2]++;
        }
    }
}


EXPORTED_API void dsci_v2(
    uint64_t *blockCount,
    const uint32_t idLim,
    const uint32_t ySection,
    const uint8_t *blocks,
    const uint8_t *add,
    const uint8_t *add2,
    const uint8_t *metadata
) {
    for (size_t y = 0; y < 16; ++y) {
        for (size_t zx = 0; zx < (16 * 16 / 2); ++zx) {
            size_t idx = zx + y * (16 * 16 / 2);
            
            uint32_t bid1 = (NIBBLE_LO(add2[idx]) << 16)
                          | (NIBBLE_LO(add[idx]) << 12)
                          | ((uint32_t)blocks[idx * 2 + 0] << 4)
                          | NIBBLE_LO(metadata[idx]);
            
            uint32_t bid2 = (NIBBLE_HI(add2[idx]) << 16)
                          | (NIBBLE_HI(add[idx]) << 12)
                          | ((uint32_t)blocks[idx * 2 + 1] << 4)
                          | NIBBLE_HI(metadata[idx]);
            
            // if (bid1 < idLim)
            blockCount[(ySection * 16 + y) * idLim + bid1]++;
            // if (bid2 < idLim)
            blockCount[(ySection * 16 + y) * idLim + bid2]++;
        }
    }
}


static void unpack_block_idxs_wc(
    const uint64_t *packed,
    const uint32_t idxBits,
    uint16_t *unpacked
) {
    if (idxBits > 16) return; // exclude impossible values, allows for better optimization
    const uint16_t mask = (1 << idxBits) - 1;
    size_t iout = 0, iin = 0;
    uint64_t packBuf = 0;
    uint32_t bufSize = 0;
    for (;;) {
        packBuf |= (packed[iin] & 0xffffffffu) << bufSize;
        bufSize += 32;
        while (bufSize >= idxBits) {
            unpacked[iout++] = packBuf & mask;
            if (iout >= 16 * 16 * 16) {
                return;
            }
            packBuf >>= idxBits;
            bufSize -= idxBits;
        }
        packBuf |= (packed[iin++] >> 32) << bufSize;
        bufSize += 32;
        while (bufSize >= idxBits) {
            unpacked[iout++] = packBuf & mask;
            if (iout >= 16 * 16 * 16) {
                return;
            }
            packBuf >>= idxBits;
            bufSize -= idxBits;
        }
    }
}

static void unpack_block_idxs_nc(
    const uint64_t *packed,
    const uint32_t idxBits,
    uint16_t *unpacked
) {
    if (idxBits > 16) return; // exclude impossible values, allows for better optimization
    const uint16_t mask = (1 << idxBits) - 1;
    size_t iout = 0, iin = 0;
    for (;;) {
        uint64_t buf = packed[iin++];
        for (uint32_t i = 64; i >= idxBits; i -= idxBits) {
            unpacked[iout++] = buf & mask;
            if (iout >= 16 * 16 * 16) {
                return;
            }
            buf >>= idxBits;
        }
    }
}

EXPORTED_API void dsci_v13(
    uint64_t *blockCount,
    const uint32_t idLim,
    const uint32_t ySection,
    const uint64_t *blockStates,
    const uint32_t *paletteMap,
    const uint32_t maxPaletteIdx,
    const _Bool carry
) {
    uint32_t idxBits = 4;
    while ((1u << idxBits) < maxPaletteIdx + 1) {
        idxBits++;
    }
    
    uint16_t paletteIdxs[16 * 16 * 16];
    if (carry)
        unpack_block_idxs_wc(blockStates, idxBits, paletteIdxs);
    else
        unpack_block_idxs_nc(blockStates, idxBits, paletteIdxs);
    
    for (size_t y = 0; y < 16; ++y) {
        for (size_t zx = 0; zx < (16 * 16); ++zx) {
            size_t idx = zx + y * (16 * 16);
            uint32_t bid = paletteMap[paletteIdxs[idx]];
            blockCount[(ySection * 16 + y) * idLim + bid]++;
        }
    }
}
