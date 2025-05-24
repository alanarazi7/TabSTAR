from tabular.datasets.manual_curation_obj import CuratedFeature, CuratedTarget
from tabular.preprocessing.objects import SupervisedTask

'''
Dataset Name: fps_benchmark
====
Examples: 24624
====
URL: https://www.openml.org/search?type=data&id=44992
====
Description: **Data Description**

This dataset contains FPS measurement of video games executed on computers. Each row of the dataset describes the outcome of FPS measurement (outcome is attribute FPS) for a video game executed on a computer. A computer is characterized by the CPU and the GPU. For both the name is resolved to technical specifications (features starting with Cpu and Gpu).

The technical specification of CPU and GPU are technical specification that describe the factory state of the respective component.

The game is characterized by the name, the displayed resolution, and the quality setting that was adjusted during the measurement (features starting with Game).

From the original data, only those observations are considered which *Dataset* feature's value is *fps-benchmark*, then the column is removed.

**Attribute Description**

CPU

1. *CpuNumberOfCores* - number of physical cores
2. *CpuNumberOfThreads* - number of threads
3. *CpuBaseClock* - base clock in Mhz
4. *CpuCacheL1* - total size of level 1 cache in kB
5. *CpuCacheL2* - total size of level 2 cache in kB
6. *CpuCacheL3* - total size of level 3 cache in MB
7. *CpuDieSize* - physical size of the die in square meter
8. *CpuFrequency* - frequency in Mhz
9. *CpuMultiplier* - multiplier of Cpu
10. *CpuMultiplierUnlocked* - 0=multiplier locked, 1=multiplier unlocked
11. *CpuProcessSize* - used process size in nanometer
12. *CpuTDP* - thermal design power in watt
13. *CpuNumberOfTransistors* - count of transistors in million
14. *CpuTurbo Clock* - turbo clock in Mhz
15. *CpuName*

GPU

16. *GpuBandwidth* bandwidth in MB/s
17. *GpuBaseClock* base clock in MHz
18. *GpuBoostClock* boost clock in MHz
19. *GpuNumberOfComputeUnits* number of computing units
20. *GpuDieSize* physical size of die in square meter
21. *GpuNumberOfExecutionUnits* number of execution units
22. *GpuFP32Performance* theoretical Float 32 performance in MFLOP/s
23. *GpuMemoryBus* width of memory bus in bits
24. *GpuMemorySize* size of memory in MB
25. *GpuPixelRate* theoretical pixel rate in MPixel/s
26. *GpuProcessSize* used process size in nanometer
27. *GpuNumberOfROPs* number of render output units
28. *GpuNumberOfShadingUnits* number of shading units
29. *GpuNumberOfTMUs* number of texture mapping units
30. *GpuTextureRate* theoretical texture rate in KTexel/s
31. *GpuNumberOfTransistors* number of transistors in million
32. *GpuArchitecture* architecture code
33. *GpuMemoryType* memory type
34. *GpuOpenCL* version of OpenCL
35. *GpuShaderModel* version of shader model
36. *GpuVulkan* version of Vulkan
37. *GpuOpenGL* version of OpenGL
38. *GpuName*
39. *GpuBus.interface* - bus interface
40. *GpuDirectX*

GAME

41. *GameName*
42. *GameResolution*
43. *GameSetting*
44. *FPS* - target feature
====
Target Variable: FPS (numeric, 2675 distinct): ['97.1', '100.7', '90.9', '108.4', '71.1', '74.4', '88.7', '85.0', '89.5', '115.5']
====
Features:

CpuName (nominal, 19 distinct): ['Intel Core i7-7700K', 'Intel Core i5-7500', 'AMD Ryzen 7 3800X', 'Intel Core i5-9400F', 'Intel Core i7-9700K', 'AMD Ryzen 5 2600', 'AMD Ryzen 5 3600', 'AMD Ryzen 5 1600X', 'Intel Core i9-9900K', 'Intel Core i5-7600K']
CpuNumberOfCores (numeric, 5 distinct): ['6', '8', '4', '2', '12']
CpuNumberOfThreads (numeric, 6 distinct): ['12', '16', '6', '4', '8', '24']
CpuBaseClock (numeric, 1 distinct): ['100']
CpuCacheL1 (numeric, 7 distinct): ['576.0', '384.0', '256.0', '768.0', '512.0', '128.0', '1152.0']
CpuCacheL2 (numeric, 7 distinct): ['3072.0', '1536.0', '1024.0', '4096.0', '2048.0', '512.0', '6144.0']
CpuCacheL3 (numeric, 8 distinct): ['16', '32', '9', '12', '6', '8', '3', '64']
CpuDieSize (numeric, 3 distinct): ['0.0001', '0.0002']
CpuFrequency (numeric, 8 distinct): ['3600.0', '3800.0', '3400.0', '3900.0', '3700.0', '4200.0', '2900.0', '2800.0']
CpuMultiplier (numeric, 8 distinct): ['36', '38', '34', '39', '37', '42', '29', '28']
CpuMultiplierUnlocked (nominal, 2 distinct): ['1', '0']
CpuProcessSize (numeric, 3 distinct): ['14', '7', '12']
CpuTDP (numeric, 5 distinct): ['95', '65', '91', '105', '51']
CpuNumberOfTransistors (numeric, 3 distinct): ['3800.0', '4800.0']
CpuTurboClock (numeric, 13 distinct): ['4500.0', '4000.0', '4200.0', '3800.0', '3900.0', '4400.0', '5000.0', '4900.0', '4700.0', '4100.0']
GpuName (nominal, 27 distinct): ['NVIDIA GeForce GTX 1660 SUPER', 'NVIDIA GeForce RTX 2080 Ti', 'NVIDIA GeForce GTX 1660 Ti', 'NVIDIA GeForce RTX 2060', 'NVIDIA GeForce GTX 1060 6 GB GDDR5X', 'AMD Radeon Pro Vega 64', 'NVIDIA GeForce GTX 980 Ti', 'AMD Radeon RX 570', 'AMD Radeon RX 5700', 'NVIDIA GeForce RTX 2080']
GpuArchitecture (nominal, 6 distinct): ['Turing', 'Pascal', 'Maxwell 2.0', 'GCN 4.0', 'RDNA 1.0', 'GCN 5.0', 'Fermi', 'Volta', 'Generation 9.0', 'Generation 7.5']
GpuBandwidth (numeric, 17 distinct): ['448000.0', '256000.0', '336600.0', '336000.0', '256300.0', '224400.0', '616000.0', '160200.0', '402400.0', '84100.0']
GpuBaseClock (numeric, 21 distinct): ['1506.0', '1000.0', '1605.0', '1607.0', '1530.0', '1350.0', '1050.0', '1257.0', '1469.0', '1250.0']
GpuBoostClock (numeric, 22 distinct): ['1545.0', '1770.0', '1683.0', '1785.0', '1709.0', '1216.0', '1340.0', '1350.0', '1725.0', '1905.0']
GpuBus.interface (nominal, 2 distinct): ['PCIe 3.0 x16', 'PCIe 4.0 x16', 'PCI', 'PCIe 3.0 x1', 'AGP 8x', 'MXM', 'MXM-A (3.0)', 'MXM-I', 'MXM-B (3.0)', 'PCIe 2.0 x1']
GpuNumberOfComputeUnits (numeric, 5 distinct): ['36.0', '40.0', '64.0', '32.0']
GpuDieSize (numeric, 13 distinct): ['0.0003', '0.0004', '0.0003', '0.0002', '0.0005', '0.0006', '0.0004', '0.0001', '0.0003', '0.0008']
GpuDirectX (nominal, 2 distinct): ['12', '12 Ultimate', '11.2', '10.1', '11.1', '9.0c', '9.0b', '10.0']
GpuNumberOfExecutionUnits (numeric, 1 distinct): []
GpuFP32Performance (numeric, 25 distinct): ['5027000.0', '4375000.0', '6060000.0', '6175000.0', '7119000.0', '11060000.0', '7949000.0', '9754000.0', '2332000.0', '2138000.0']
GpuMemoryBus (numeric, 8 distinct): ['256.0', '192.0', '352.0', '384.0', '160.0', '128.0', '96.0', '2048.0']
GpuMemorySize (numeric, 8 distinct): ['8000.0', '6000.0', '4000.0', '11000.0', '12000.0', '5000.0', '3000.0', '16000.0']
GpuMemoryType (nominal, 4 distinct): ['GDDR5', 'GDDR6', 'GDDR5X', 'HBM2', 'GDDR3', 'HBM', 'DDR3', 'DDR2']
GpuOpenCL (nominal, 2 distinct): ['1.2', '2', '1.1', '1', '2.1']
GpuOpenGL (nominal, 1 distinct): ['4.6', '4.4', '3.3', '4', '2.1', '2', '4.3']
GpuPixelRate (numeric, 25 distinct): ['107700.0', '85680.0', '136000.0', '77820.0', '42880.0', '49440.0', '86400.0', '110400.0', '121900.0', '36430.0']
GpuProcessSize (numeric, 5 distinct): ['12', '16', '14', '28', '7']
GpuNumberOfROPs (numeric, 8 distinct): ['64', '48', '32', '88', '96', '40', '56', '24']
GpuShaderModel (nominal, 2 distinct): ['6.4', '6.5', '6.3', '5.1', '5', '4.1', '4', '6', '3']
GpuNumberOfShadingUnits (numeric, 17 distinct): ['2304.0', '2560.0', '1408.0', '768.0', '2048.0', '1920.0', '1280.0', '4352.0', '1664.0', '2816.0']
GpuNumberOfTMUs (numeric, 17 distinct): ['144.0', '160.0', '88.0', '48.0', '128.0', '120.0', '80.0', '272.0', '104.0', '176.0']
GpuTextureRate (numeric, 25 distinct): ['157100.0', '136700.0', '189400.0', '193000.0', '222500.0', '345600.0', '248400.0', '304800.0', '72860.0', '66820.0']
GpuNumberOfTransistors (numeric, 13 distinct): ['7200.0', '10800.0', '6600.0', '5700.0', '13600.0', '8000.0', '5200.0', '3300.0', '10300.0', '18600.0']
GpuVulkan (nominal, 3 distinct): ['1.2.131', '1.1.126', '1.1.125', '1.1.80', '1.1.97', '1.1.103']
GameName (nominal, 24 distinct): ['counterStrikeGlobalOffensive', 'grandTheftAuto5', 'rainbowSixSiege', 'starcraft2', 'farCry5', 'battletech', 'airMechStrike', 'warframe', 'totalWar3Kingdoms', 'frostpunk']
GameResolution (numeric, 1 distinct): ['1080.0']
GameSetting (nominal, 2 distinct): ['med', 'max', 'low', 'high']
'''

CONTEXT = "Gaming Frames Per Second Benchmark"
TARGET = CuratedTarget(raw_name="FPS", new_name="Frames Per Second", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ['CpuBaseClock', 'GpuNumberOfExecutionUnits', "GameResolution"]
FEATURES = [CuratedFeature(raw_name="CpuName", new_name="CPU Name"),
            CuratedFeature(raw_name="GpuName", new_name="GPU Name"),
            CuratedFeature(raw_name="GameName", new_name="Game Name"),
            CuratedFeature(raw_name="GameSetting", new_name="Game Setting"),
            CuratedFeature(raw_name="CpuNumberOfCores", new_name="Number of CPU Cores"),
            CuratedFeature(raw_name="CpuNumberOfThreads", new_name="Number of CPU Threads"),
            CuratedFeature(raw_name="CpuCacheL1", new_name="CPU Cache L1"),
            CuratedFeature(raw_name="CpuCacheL2", new_name="CPU Cache L2"),
            CuratedFeature(raw_name="CpuCacheL3", new_name="CPU Cache L3"),
            CuratedFeature(raw_name="CpuDieSize", new_name="CPU Die Size"),
            CuratedFeature(raw_name="CpuFrequency", new_name="CPU Frequency"),
            CuratedFeature(raw_name="CpuMultiplier", new_name="CPU Multiplier"),
            CuratedFeature(raw_name="CpuMultiplierUnlocked", new_name="CPU Multiplier Unlocked"),
            CuratedFeature(raw_name="CpuProcessSize", new_name="CPU Process Size"),
            CuratedFeature(raw_name="CpuTDP", new_name="CPU Thermal Design Power"),
            CuratedFeature(raw_name="CpuNumberOfTransistors", new_name="Number of CPU Transistors"),
            CuratedFeature(raw_name="CpuTurboClock", new_name="CPU Turbo Clock"),
            CuratedFeature(raw_name="GpuBandwidth", new_name="GPU Bandwidth"),
            CuratedFeature(raw_name="GpuBaseClock", new_name="GPU Base Clock"),
            CuratedFeature(raw_name="GpuBoostClock", new_name="GPU Boost Clock"),
            CuratedFeature(raw_name="GpuNumberOfComputeUnits", new_name="Number of GPU Compute Units"),
            CuratedFeature(raw_name="GpuDieSize", new_name="GPU Die Size"),
            CuratedFeature(raw_name="GpuFP32Performance", new_name="GPU Float 32 Performance"),
            CuratedFeature(raw_name="GpuMemoryBus", new_name="GPU Memory Bus"),
            CuratedFeature(raw_name="GpuMemorySize", new_name="GPU Memory Size"),
            CuratedFeature(raw_name="GpuMemoryType", new_name="GPU Memory Type"),
            CuratedFeature(raw_name="GpuPixelRate", new_name="GPU Pixel Rate"),
            CuratedFeature(raw_name="GpuProcessSize", new_name="GPU Process Size"),
            CuratedFeature(raw_name="GpuNumberOfROPs", new_name="Number of GPU Render Output Units"),
            CuratedFeature(raw_name="GpuShaderModel", new_name="GPU Shader Model"),
            CuratedFeature(raw_name="GpuNumberOfShadingUnits", new_name="Number of GPU Shading Units"),
            CuratedFeature(raw_name="GpuNumberOfTMUs", new_name="Number of GPU Texture Mapping Units"),
            CuratedFeature(raw_name="GpuTextureRate", new_name="GPU Texture Rate"),
            CuratedFeature(raw_name="GpuNumberOfTransistors", new_name="Number of GPU Transistors"),
            CuratedFeature(raw_name="GpuArchitecture", new_name="GPU Architecture"),
            CuratedFeature(raw_name="GpuVulkan", new_name="GPU Vulkan"),
            CuratedFeature(raw_name="GpuDirectX", new_name="GPU DirectX"),
            CuratedFeature(raw_name="GpuBus.interface", new_name="GPU Bus Interface"),
            ]