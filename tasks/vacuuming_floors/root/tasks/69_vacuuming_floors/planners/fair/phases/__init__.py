"""One module per phase group of the fair planner of task 69. Each module lists its phases in PHASES = {name: step function}; a step function
is called as step(core, action, q, world) every control while the phase is active, and ends the phase by core.change(next, reason)."""
