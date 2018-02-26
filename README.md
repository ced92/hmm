# HMM model

This repository contains logic and information relating to a project in advanced machine learning. The main goal was to
derive and implement polynomial time algorithms for computing probabilities. The problem/event model
is a casino model in which a player visits 'k' number of tables and at each table either observes a dice outcome or does not.
After visiting all the tables, we are also given the sum of all outcomes whether or not they were observed. Given this information,
the goal is then to calculate probabilities of vairous events.

## Example Sequence
Observation Sequence: 5,1,-1,-1,4 (-1 implies not observed)

Observed Sum: 14
