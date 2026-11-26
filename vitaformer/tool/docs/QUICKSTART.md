# VitaFormer – quick start / 快速上手 (v1.1.0)

## 1. Install once / 首次安装
- **Windows 10/11 (64-bit):** double-click `install_windows.bat`.
  - The script downloads Python and the fixed package versions (about 250 MB in total) into the `runtime` folder inside this
    directory. Nothing else on the computer is changed.
  - For an NVIDIA GPU, run `install_windows.bat gpu` from a command window instead.
  - On a slow network the first installation can take more than 30 minutes (24 min measured on a Linux server). If it stops,
    run the installer again; finished steps are skipped.
- **Linux / macOS:** run `./install.sh`.
- The model file `weights/vitaformer_m2_seed42_final.pth` (361 MB) is included in the package. Its checksum is verified on every
  start, and the tool refuses to run with any other file.

**中文：** Windows 用户双击 `install_windows.bat` 即可完成安装。安装程序会把 Python 和固定版本的依赖（共约 250 MB）下载到本文件夹下的 `runtime` 目录，不改动电脑上的其他内容。有 NVIDIA 显卡时，在命令行窗口运行 `install_windows.bat gpu`。

## 2. Start / 启动
- **Windows:** double-click `start_VitaFormer.bat`. **Linux/macOS:** `./start.sh`.
- The browser opens at `http://127.0.0.1:8765`. Images are processed on this computer only. Keep the black window open while you
  work; close it to stop.
- Starting again while the tool is already running opens the existing window instead of a second copy.

**中文：** 双击 `start_VitaFormer.bat`，浏览器会自动打开。图片只在本机处理，不会上传。使用期间不要关闭黑色窗口，关闭它即退出。

## 3. Use / 使用（4 步）
1. **Import & segment / 导入并分割**
   - Choose files or a folder, or drag them in. Each file is checked; files that cannot be used are listed with the reason and
     are not processed.
   - Choose the **input size**:
     - *Native* (default; validated for CRC and DeepOrganoid images);
     - *Scale factor, bilinear* (validated: 0.25 for GemcitabineScreen);
     - *Square side S, box* (validated: 256 for NViR liver spheroids; square images only).
   - Pick the size by looking at the masks of a few images, never from the result you want to study.
   - Press **Start**. Progress is shown; **Cancel** keeps the images already finished. Measured on a server CPU (8 threads):
     about 40 s per 2048 × 2044 image at native size, and 16 s for the 6 example images at S = 256 (including model loading).
     A GPU is much faster.
2. **Inspect masks / 检查掩膜**
   - Switch between overlay, original, mask and side-by-side views.
   - Change the mask opacity, scroll to zoom and drag to move. The arrow keys change image.
   - Look for missed organoids, false positives at the well edge and merged neighbours.
3. **Controls (optional) / 对照（可选）**
   - Give each image a group (for example the same plate and day) and mark its control images. You can also load a CSV with the
     columns `image,group,role`; a template can be downloaded.
   - **Check groups** shows each group, its controls and any problem before anything is calculated.
   - **Create normalised table** then writes a new `normalised_N` folder.
4. **Export / 导出**
   - Everything is already saved in the run folder. Use **Open folder** or **Download as ZIP**.

## 4. Try the example / 试运行示例
1. Import the 6 images in `examples_nvir/input` and choose **Square side S = 256**. Run, then inspect the masks.
2. In **Controls**, load `examples_nvir/plate_map.csv`, then press **Check groups** and **Create normalised table**.
3. Check the result: `runtime\python.exe examples_nvir\check_example.py "<run folder>"` (Windows) or
   `runtime/env/bin/python examples_nvir/check_example.py "<run folder>"` (Linux). The check should report PASS.

## 5. What the numbers mean / 结果含义
- **Image measurements** are in pixels and come from the mask alone. "Region count" = connected regions; touching organoids
  merge, so it is **not an organoid count**.
- **Relative to controls** compares images within one group. It is not a viability percentage.
- **Alive/Dead are model classes.** Dead predictions have **not** been externally validated and must not be reported as cell death.
- The tool does **not** predict CellTiter-Glo, ATP or any other assay value.

**中文：** 图像测量值的单位是像素。“区域数量”是连通区域的数量，不等于类器官数量。相对对照的读出只在同一分组内比较，不是活率。Alive/Dead 是模型类别，Dead 尚未经外部验证。本工具不预测 CTG、ATP 等实验数值。
