"""Discovery bootstrap contracts and validators (discovery spec §8).

``contracts`` is data-only and never imports ``contracts/validation``,
``contracts/state``, ``contracts/planning``, or ``discovery_bootstrap/validation``;
validators depend on the models plus the cycle-free
``contracts/validation_support`` primitives only.
"""
