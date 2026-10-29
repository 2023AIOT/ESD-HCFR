import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Node:
    x: float
    y: float
    energy: float
    head_rounds: int = 0


class SimpleWSN:
    E_ELEC = 50e-9
    E_FS = 10e-12
    E_MP = 1.3e-15
    E_AGG = 5e-9
    BITS = 4000

    def __init__(self, nodes=80, area=150.0, energy=0.5, ratio=0.1, seed=44):
        if nodes < 1 or not all(math.isfinite(v) for v in (area, energy, ratio)):
            raise ValueError('Require positive node count and finite parameters')
        if area <= 0 or energy <= 0 or not 0 < ratio <= 1:
            raise ValueError('Require area > 0, energy > 0, and 0 < ratio <= 1')
        rng = random.Random(seed)
        self.nodes = [Node(rng.uniform(0, area), rng.uniform(0, area), energy)
                      for _ in range(nodes)]
        self.bs = (area / 2, area / 2)
        self.initial_energy = energy
        self.ratio = ratio
        self.distances = [[math.hypot(a.x - b.x, a.y - b.y)
                           for b in self.nodes] for a in self.nodes]
        self.bs_distances = [math.hypot(n.x - self.bs[0], n.y - self.bs[1])
                             for n in self.nodes]

    def tx_cost(self, distance):
        threshold = math.sqrt(self.E_FS / self.E_MP)
        amplifier = (self.E_FS * distance ** 2 if distance <= threshold
                     else self.E_MP * distance ** 4)
        return self.BITS * (self.E_ELEC + amplifier)

    def spend(self, index, cost):
        """An underpowered operation fails and exhausts the node's energy."""
        node = self.nodes[index]
        success = node.energy >= cost
        node.energy = max(0.0, node.energy - cost)
        return success

    def select_heads(self, living):
        """Favor remaining energy, fewer prior head rounds and short BS links.

        This is a new heuristic, not the original hybrid optimization solver.
        """
        count = max(1, math.ceil(len(living) * self.ratio))
        def score(i):
            n = self.nodes[i]
            return n.energy / ((1 + n.head_rounds) * self.tx_cost(self.bs_distances[i]))
        return sorted(living, key=lambda i: (-score(i), i))[:count]

    def run(self, rounds):
        if rounds < 1:
            raise ValueError('rounds must be positive')
        history = []
        for r in range(1, rounds + 1):
            living = [i for i, n in enumerate(self.nodes) if n.energy > 0]
            if not living:
                break
            before = sum(n.energy for n in self.nodes)
            heads = self.select_heads(living)
            clusters = {h: [] for h in heads}
            for i in living:
                if i not in clusters:
                    h = min(heads, key=lambda h: (self.distances[i][h], h))
                    clusters[h].append(i)
            delivered = 0
            for h, members in clusters.items():
                self.nodes[h].head_rounds += 1
                buffered = 1  # The head's own reading.
                for i in members:
                    # All assigned members attempt transmission, even if their
                    # head has exhausted its energy earlier in this round.
                    sent = self.spend(i, self.tx_cost(self.distances[i][h]))
                    if sent and self.nodes[h].energy > 0:
                        if self.spend(h, self.BITS * self.E_ELEC):
                            buffered += 1
                if self.nodes[h].energy <= 0:
                    continue
                if not self.spend(h, buffered * self.BITS * self.E_AGG):
                    continue
                if self.spend(h, self.tx_cost(self.bs_distances[h])):
                    delivered += buffered
            remaining = sum(n.energy for n in self.nodes)
            history.append(dict(round=r, alive=sum(n.energy > 0 for n in self.nodes),
                                heads=len(heads), generated=len(living),
                                delivered=delivered, delivery_ratio=delivered / len(living),
                                remaining_energy_j=remaining,
                                consumed_energy_j=before - remaining))
        return history


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--nodes', type=int, default=80)
    parser.add_argument('--rounds', type=int, default=1000)
    parser.add_argument('--area', type=float, default=150.0, help='Square side length in metres')
    parser.add_argument('--energy', type=float, default=0.5, help='Initial joules per node')
    parser.add_argument('--ch-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--output', default='simple_wsn_results')
    args = parser.parse_args()
    try:
        net = SimpleWSN(args.nodes, args.area, args.energy, args.ch_ratio, args.seed)
        history = net.run(args.rounds)
    except ValueError as exc:
        parser.error(str(exc))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'rounds.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    generated = sum(row['generated'] for row in history)
    delivered = sum(row['delivered'] for row in history)
    summary = dict(model='Simple WSN demonstration; not paper reproduction',
                   parameters=vars(args), completed_rounds=len(history),
                   final_alive=history[-1]['alive'],
                   first_node_death_round=next((row['round'] for row in history
                                                if row['alive'] < args.nodes), None),
                   all_nodes_dead_round=next((row['round'] for row in history
                                              if row['alive'] == 0), None),
                   generated_readings=generated, delivered_readings=delivered,
                   delivery_ratio=delivered / generated,
                   remaining_energy_j=history[-1]['remaining_energy_j'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'nodes.json').write_text(json.dumps(
        [vars(n) for n in net.nodes], indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('Results saved to:', output.resolve())


if __name__ == '__main__':
    main()
