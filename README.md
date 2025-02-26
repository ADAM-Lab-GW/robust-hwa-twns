# Robust Hardware-Aware Neural Networks for FeFET-based Accelerators

This repository contains code for the following paper:

```
@article{yousuf2024hwatwns,
  title={Robust Hardware-Aware Neural Networks for FeFET-based Accelerators},
  author={Yousuf, Osama and Glasmann, Andreu and Mazzoni, Alexander L. and Najmaei, Sina and Adam, Gina C.},
  journal={IEEE Transactions on Nanotechnology},
  year={2025}
}
```

## Installation

- Clone this repository and navigate inside:

```
git clone https://github.com/ADAM-Lab-GW/robust-hwa-twns &&
cd robust-hwa-twns
```

- Set up a virtual environment and install dependencies:
```
python3 -m venv env &&
source env/bin/activate &&
pip install -r requirements.txt
```

**NOTE:** The scripts have been tested with Python 3.8.10 and Ubuntu 20.04.6 LTS. Minor changes to packages may be required for other Python versions or operating systems.

## Repository Structure

| Directory/Files    | Description |
| -------- | ------- |
| `processed/`  | Directory where pre-processed inference results are stored for each dataset.     |
| `eda.ipynb` | Exploratory notebook where pre-trained network weights can be loaded and tested.     |
| `models.py`    | Code for PyTorch model definitions.   |
| `plots.py`    | Code for various plotting functions utilized in the `eda.ipynb` notebook.    |

## Citations

To cite *Robust Hardware-Aware Neural Networks for FeFET-based Accelerators*, use the following BibTeX entry:

```
@article{yousuf2024hwatwns,
  title={Robust Hardware-Aware Neural Networks for FeFET-based Accelerators},
  author={Yousuf, Osama and Glasmann, Andreu and Mazzoni, Alexander L. and Najmaei, Sina and Adam, Gina C.},
  journal={IEEE Transactions on Nanotechnology},
  year={2025}
}
```

## License

Distributed under the BSD-3 License. See LICENSE.md for more information.

## Contact

- Osama Yousuf (osamayousuf@gwu.edu)
- Gina C. Adam (ginaadam@gwu.edu)
