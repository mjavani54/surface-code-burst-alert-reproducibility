"""Stim memory circuits with an optional burst of elevated circuit noise."""

import stim


def memory_circuit(
    distance: int,
    rounds: int,
    basis: str,
    background_rate: float,
    burst_rate: float,
    burst_round: int = 0,
    burst_length: int = 1,
) -> stim.Circuit:
    """Build a rotated memory; burst_round is 1-based, with 0 for no burst."""
    if basis not in ("x", "z"):
        raise ValueError("basis must be x or z")
    if not 0 <= burst_round <= rounds:
        raise ValueError("burst_round must be between 0 and rounds")
    if burst_length < 1 or (burst_round and burst_round + burst_length - 1 > rounds):
        raise ValueError("the burst must fit inside the syndrome rounds")
    if not 0 <= background_rate <= burst_rate < 0.5:
        raise ValueError("require 0 <= background_rate <= burst_rate < 0.5")

    ideal = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}", distance=distance, rounds=rounds
    ).flattened()
    readout = "MX" if basis == "x" else "M"
    data_qubits = {
        target.value
        for operation in ideal
        if operation.name == readout
        for target in operation.targets_copy()
    }
    if len(data_qubits) != distance**2:
        raise ValueError("unexpected Stim data-qubit readout schedule")

    circuit = stim.Circuit()
    round_number = 1
    # Stim's generated schedule has two H layers, four CX layers, and one
    # measurement/reset layer per round. Verify the schedule before using it.
    expected_layers = ["H", "CX", "CX", "CX", "CX", "H", "MR"]
    layers: list[str] = []
    for operation in ideal:
        name = operation.name
        targets = operation.targets_copy()
        inside_burst = burst_round and burst_round <= round_number < burst_round + burst_length
        rate = burst_rate if inside_burst else background_rate
        if name in ("H", "CX"):
            layers.append(name)
            circuit.append(operation)
            active_qubits = {target.value for target in targets}
            circuit.append("DEPOLARIZE1" if name == "H" else "DEPOLARIZE2", targets, rate)
            idle_data = sorted(data_qubits - active_qubits)
            if idle_data:
                circuit.append("DEPOLARIZE1", idle_data, rate)
        elif name == "MR":
            layers.append(name)
            if layers != expected_layers:
                raise ValueError(f"unexpected Stim gate layers in round {round_number}: {layers}")
            layers.clear()
            # MR(p) flips the recorded check bit and resets the ancilla.
            circuit.append(name, targets, rate)
            circuit.append("DEPOLARIZE1", sorted(data_qubits), rate)
            round_number += 1
        elif name in ("R", "RX"):
            circuit.append(operation)
            initialized_data = sorted(data_qubits.intersection(target.value for target in targets))
            if initialized_data:
                circuit.append("Z_ERROR" if name == "RX" else "X_ERROR", initialized_data, background_rate)
        elif name == readout:
            # Destructive data readout follows the final syndrome round.
            circuit.append("TICK")
            circuit.append(name, targets, background_rate)
        elif name in ("TICK", "QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"):
            circuit.append(operation)
        else:
            raise ValueError(f"unexpected Stim instruction: {name}")

    if round_number != rounds + 1 or circuit.num_observables != 1:
        raise ValueError("unexpected number of rounds or logical observables")
    return circuit
