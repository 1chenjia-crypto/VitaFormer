# User trial – task sheet for a tester who did not develop the tool / 非开发者试用任务单

**Organiser:** give the tester the package folder and `QUICKSTART.md` only. Do not help unless the tester is stuck for more than
5 minutes, and write down every point where help was given. **Tester:** speak aloud while working; it is the tool that is being
tested, not you.

| # | Task | Done without help? (Y/N) | Time (min) | Problems, confusing wording |
|---|---|---|---|---|
| 1 | Install the tool (Windows: `install_windows.bat`) | | | |
| 2 | Start the tool; the page opens in the browser | | | |
| 3 | Import the 6 example images from `examples_nvir/input` | | | |
| 4 | Choose input size "Square side S" = 256 and start | | | |
| 5 | While it runs, find how many images are done; then cancel a second run and confirm finished images are kept | | | |
| 6 | Open image 3 (E06); switch overlay / original / mask / side by side; zoom into the spheroid edge | | | |
| 7 | Say in your own words what "Alive", "Dead" and "Region count" mean here | | | |
| 8 | In Controls, mark H02, H03 and H04 as controls and give all 6 images the group `plate2_day4` (by hand, not with the CSV) | | | |
| 9 | Press "Check groups"; then remove the control role from all three and check again. What does the tool say? | | | |
| 10 | Create the normalised table; find it on disk | | | |
| 11 | Open the results folder; find the mask, the measurements table and the record of the model version | | | |
| 12 | Import a file that is not an image (any .txt or a renamed file). What happens? | | | |
| 13 | Close the tool and start it twice. What happens? | | | |
| 14 | Run `check_example.py` on your run folder (organiser may help with the command) | | | |

**After the trial (1–5 scale):** How easy was it overall? Was anything misleading? Would you trust the exported numbers, and why?

**Record:** tester role (student / technician / biologist), computer (Windows version, CPU/GPU), date, total time, organiser.
