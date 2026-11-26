# Example sets

Each set contains input images, the masks expected from VitaFormer seed 42, and the input size to use.

| Set | Images | Input size to choose in VitaFormer | Expected masks were computed on |
|---|---|---|---|
| `crc/` | 2 colorectal cancer organoid images (2048 × 2044 px, RGB) | **Native resolution** | GPU (masks of the paper) |
| `nvir/` | 6 liver spheroid images (1216 × 1216 px, greyscale) | **Square side S, box**, S = **256** | CPU |

The CRC set covers the image with the most annotated Dead pixels in the test set and an image of median accuracy. At native
resolution each CRC image takes about 3.5 minutes on a laptop CPU; the NViR set takes about 30 seconds in total.

## Run and check
1. Start VitaFormer and import all images of one set from its `input` folder.
2. Choose the input size from the table and press **Start**.
3. (NViR, optional) In **Controls**, load `nvir/plate_map.csv`, press **Check groups**, then **Create normalised table**.
   Expected normalised values are in `nvir/expected_output/normalised/`.
4. Drag the run folder (for example `C:\Users\<you>\VitaFormer_results\20261124-141553_run`) onto `check_examples.bat`.

The check confirms that the frozen weights and the right input size were used, that the input files are unchanged, and that
every mask agrees with the expected mask on at least 99.9 % of its pixels. CPU and GPU results can differ in a few pixels at
class boundaries; identical results are reported as *PASS (identical)*.

`check_examples.bat` uses the Python runtime installed with VitaFormer
(`%LOCALAPPDATA%\Programs\VitaFormer\runtime\python.exe`). With another Python (NumPy and Pillow required):
`python check_examples.py "<run folder>"`.
