# WorldLayers

CLI program to scan your Minecraft worlds and plot block distribution by height or print overall block statistics.

### Features:
- Plot distribution of selected blocks by Y layer.
- Print total statistics of the number of blocks scanned.
- Supports scanning Anvil and Region worlds (beta 1.7 - now).
- Save and load scan results from a file. No need to re-scan to visualize different set of blocks.

### Example output

- Ore distribution in version 1.12.2

![Ore in 1.12.2](images/example1.png)

- Overall block counts in version 1.19

```
Loaded data for 2458 chunks, Y levels -64 ~ 96
Nonair blocks = 78163359	(77.635685%)	(31799.5765 b/ch)
minecraft:deepslate.0 = 29401483	(29.202996%)	(11961.5472 b/ch)
minecraft:stone.0 = 29031674	(28.835684%)	(11811.0960 b/ch)
minecraft:tuff.0 = 2658187	(2.640242%)	(1081.4430 b/ch)
minecraft:water.0 = 2333244	(2.317492%)	(949.2449 b/ch)
minecraft:dirt.0 = 2215879	(2.200920%)	(901.4967 b/ch)
minecraft:andesite.0 = 2164731	(2.150117%)	(880.6880 b/ch)
minecraft:diorite.0 = 2077764	(2.063737%)	(845.3068 b/ch)
minecraft:granite.0 = 2054355	(2.040486%)	(835.7832 b/ch)
minecraft:bedrock.0 = 1887501	(1.874759%)	(767.9011 b/ch)
minecraft:cave_air.0 = 1873444	(1.860797%)	(762.1823 b/ch)
minecraft:gravel.0 = 602045	(0.597981%)	(244.9329 b/ch)
minecraft:grass_block.0 = 407801	(0.405048%)	(165.9076 b/ch)
minecraft:coal_ore.0 = 220141	(0.218655%)	(89.5610 b/ch)
minecraft:oak_leaves.0 = 164205	(0.163096%)	(66.8043 b/ch)
minecraft:sand.0 = 117400	(0.116607%)	(47.7624 b/ch)
minecraft:birch_leaves.0 = 90281	(0.089672%)	(36.7295 b/ch)
minecraft:copper_ore.0 = 82047	(0.081493%)	(33.3796 b/ch)
...
```

## Command line syntax
```python cli.py [parameters...]```
- TODO

## Requirements
- Python 3.9 or higher
- [uNBT](https://github.com/Metaray/uNBT)
- numpy
- matplotlib
- gcc (to compile accelerators library)
