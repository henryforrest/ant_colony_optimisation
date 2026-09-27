# Ant colony optimisation for bin packing

Coursework from a nature-inspired computation module at the University of Exeter: an
ant colony optimisation (ACO) solver, written from scratch in Python, for a balancing
variant of the bin-packing problem. Given n weighted items and m bins, the aim is to
spread the items so that the heaviest and lightest bins differ by as little as possible.

## Problem

Two instances are defined in `main.py`, both with 500 items:

- BPP1: 10 bins, and item i has weight i (for i = 1 to 500).
- BPP2: 50 bins, and item i has weight i^2 / 2.

Fitness is the total weight of the heaviest bin minus that of the lightest, and
lower is better. BPP2 is the harder of the two: its largest items are a sizeable
fraction of a bin's ideal share, so exact balance is much harder to reach.

## Algorithm

- Pheromone matrix: a construction graph with one row per item and one column per
  bin, initialised with uniform random values in [0, 1). The generator is seeded
  (`random.seed(1)`) so a run is reproducible; I changed the seed by hand between trials.
- Path construction: each ant takes the items in order and picks a bin for each with
  `random.choices`, weighted by that item's pheromone row. There is no heuristic term
  and no alpha/beta exponent; pheromone alone drives the choice.
- Evaporation: after every ant in an iteration has built a solution, each entry is
  multiplied by `EVAPORATION_RATE` (0.4 by default). As coded this is the fraction
  retained, so a higher value means slower evaporation.
- Deposit: each ant adds `100 / fitness` to every (item, bin) entry on its path, so
  better solutions reinforce their choices more strongly.
- Budget: `NUM_ANTS` (10) ants per iteration for `10000 / NUM_ANTS` iterations, which
  is 10,000 fitness evaluations per run. The best allocation seen is kept and the
  best-so-far fitness is recorded after every iteration.

## Experiments

The default run calls `bpp1()` then `bpp2()` with p = 10 ants and e = 0.4, printing
the best fitness each iteration and then the final allocation. Three plotting
helpers used for the write-up are included but not called:

- `heatmap()`: a sensitivity sweep over p in {5, 10, 15, 20} and e in
  {0.5, 0.6, 0.7, 0.8, 0.9} on BPP2, plotting the final best fitness for each pair.
- `graph1()`: best-fitness-per-iteration curves for BPP2 at two evaporation rates
  (labelled e = 0.6 and e = 0.9).
- `bin_distribution()`: a bar chart of the final weight in each of BPP2's 50 bins.

One caveat: these helpers assign to `NUM_ANTS` and `EVAPORATION_RATE` as local
variables rather than through `global`, so to reproduce a setting you need to edit
the two constants at the top of the file.

## Running

    pip install numpy matplotlib seaborn
    python main.py

Both problems together take about 20 seconds on a recent laptop (roughly 5 s for
BPP1 and 13 s for BPP2). The heatmap sweep runs BPP2 twenty times, so allow several
minutes if you call it.
