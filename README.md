# MCPS

Code for simulating plasmas inside a magnetic confinement device.

## Setup

```sh
# install pipenv
pip install --user pipenv

# install dependencies
pipenv install --dev

# use the virtual env
pipenv shell
```

Set the number of particles to run the simulation with.

```sh
# number of particles (default 50)
N = 50
```

Run the program:

```sh
python3 mcps.py
```
