# Bird project playground

## Installation instructions

Install [`pyenv`](https://github.com/pyenv/pyenv) and setup it.
The `.python-version` file in this repo should pick up the version of Python that we are using.
Check with `which python` and `python --version`.
If it is not correct, check your `pyenv` installation.

```shell
python -m venv env
. env/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

If you want to use ipykernel/notebook, install it now.
