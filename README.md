# FEI VSB-TUO: BIA

![Python Version](https://img.shields.io/badge/python-3.13-blue)

## Table of Contents

- [Introduction](#introduction)
- [List of problems](#list-of-problems)
- [Dependencies](#dependencies)

## Introduction

A set of Jupyter notebooks implementing and visualizing classical biologically inspired optimization algorithms. Each notebook covers one algorithm with a worked example and animated visualization of the search process.

## List of algorithms

- Blind Search algorithm
- Hill Climbing algorithm
- Simulated annealing
- Genetic algorithm applied to Travelling Salesman Problem
- Differential evolution and its improved versions
- Particle Swarm Optimization with inertia weight
- Self-organizing Migration Algorithm - AllToOne
- Ant Colony Optimization applied to Travelling Salesman Problem
- Firefly algorithm
- Teaching-learning Based Optimization
- Performance comparison across the above

## Running it

Requires Python 3.13+ and `ffmpeg` on the system PATH (used by `imageio` to render animations).

```bash
pip install -r requirements.txt
jupyter notebook
```
