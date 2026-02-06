"""
Network Module for Power Market Simulation

Enthält alle Funktionen für PyPSA-Netzwerk-Erstellung und -Management.
"""

from .network_builder import NetworkBuilder, build_copper_plate_network
from .generators import (
    validate_generator_config,
    get_generation_mix_summary,
    calculate_total_capacity_by_carrier,
    get_merit_order_curve,
    estimate_available_capacity,
    add_generator_from_config,
    scale_generators,
    get_renewable_share,
    get_generator_statistics,
    create_generator_color_map,
    export_generators_to_csv
)
from .storage import (
    validate_storage_config,
    add_storage_from_config,
    calculate_storage_statistics,
    get_storage_timeseries,
    calculate_storage_cycles,
    calculate_storage_revenue,
    distribute_storage_capacity,
    get_storage_color_map,
    export_storage_to_csv
)

__all__ = [
    # Network Builder
    'NetworkBuilder',
    'build_copper_plate_network',

    # Generators
    'validate_generator_config',
    'get_generation_mix_summary',
    'calculate_total_capacity_by_carrier',
    'get_merit_order_curve',
    'estimate_available_capacity',
    'add_generator_from_config',
    'scale_generators',
    'get_renewable_share',
    'get_generator_statistics',
    'create_generator_color_map',
    'export_generators_to_csv',

    # Storage
    'validate_storage_config',
    'add_storage_from_config',
    'calculate_storage_statistics',
    'get_storage_timeseries',
    'calculate_storage_cycles',
    'calculate_storage_revenue',
    'distribute_storage_capacity',
    'get_storage_color_map',
    'export_storage_to_csv',
]
