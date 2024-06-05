# Mishax

## Introduction

Mishax is a utility library for mechanistic interpretability research, with its motivations explained in this [blog post](https://www.alignmentforum.org/posts/C5KAZQib3bzzpeyrg/progress-update-1-from-the-gdm-mech-interp-team-full-update#Instrumenting_LLM_model_internals_in_JAX). It enables users to do 2 things:

`mishax.ast_patcher` enables running code from some other library (e.g. a deep learning codebase) with some source-level code modifications applied. For mechanistic interpretability this can be used to stick probes in the model and intervene at arbitrary locations. This otherwise requires forking the code that’s being modified, but that comes with more maintenance requirements.

`mishax.safe_greenlet`, given a complicated function `f` that allows running arbitrary callbacks somewhere deep inside (e.g. using Flax’s `intercept_methods`), enables transforming it into an ordinary-looking Python `for` loop that iterates over internal values and allows them to be replaced with other values. Behind the scenes, this will run `f` in a kind of separate “thread” –- but the user can mostly ignore that, and use the loop to read and write representations into the model during a forward pass, in a way that interoperates well with the rest of JAX.

## Setup

```shell
python3 -m venv $HOME/mishax-venv
source $HOME/mishax-venv/bin/activate
python3 -m pip install git+git://github.com/google-deepmind/mishax.git
```

To deactivate the virtual environment, run `deactivate`.

## Run tests

```shell
source $HOME/mishax-venv/bin/activate
python3 -m pip install git+git://github.com/google-deepmind/mishax.git[dev]
python3 -m mishax.ast_patcher_test
python3 -m mishax.safe_greenlet_test
```
