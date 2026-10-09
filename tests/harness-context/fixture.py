#!/usr/bin/env python3
"""Standalone generated-data exercise; writes only beside this script."""
import itertools
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'state.json'
AUDIT = ROOT / 'audit.jsonl'
ROUNDS = 100
ROWS = 128
DIGITS = 64


def best(batch):
    plans = []
    for items in itertools.combinations(batch['items'], 3):
        ids = [x['id'] for x in items]
        names = set(ids)
        cost = sum(x['cost'] for x in items)
        value = sum(x['value'] for x in items)
        if cost > batch['budget'] or {'A', 'C'} <= names:
            continue
        if 'E' in names and 'F' not in names:
            continue
        if not names.intersection({'B', 'G'}):
            continue
        plans.append({'round': batch['round'], 'ids': ids, 'cost': cost, 'value': value})
    return min(plans, key=lambda x: (-x['value'], x['ids'])) if plans else None


def fresh(n):
    rng = random.SystemRandom()
    while True:
        batch = {'round': n, 'budget': rng.randint(60, 75),
                 'items': [{'id': name, 'cost': rng.randint(9, 31),
                            'value': rng.randint(17, 77)} for name in 'ABCDEFGH']}
        if best(batch) is not None:
            return batch


def emit(batch):
    print('BEGIN_PAYLOAD round=' + str(batch['round']))
    print('Processed archive; these records require no action:')
    rng = random.Random(202610090000 + batch['round'])
    for row in range(ROWS):
        data = ''.join(str(rng.randrange(10)) for _ in range(DIGITS))
        print(f"R{batch['round']:03d}.{row:03d} {data}")
    print('CURRENT_INVENTORY ' + json.dumps(batch, separators=(',', ':')))
    print(f"END_PAYLOAD round={batch['round']} archive_rows={ROWS}")


def main():
    mode = sys.argv[1]
    if mode == 'start':
        if STATE.exists() or AUDIT.exists():
            raise SystemExit('Use a fresh folder; an existing run is never overwritten.')
        batch = fresh(0)
        STATE.write_text(json.dumps({'completed': 0, 'current': batch}))
        with AUDIT.open('a') as stream:
            stream.write(json.dumps({'event': 'inventory', 'batch': batch}) + '\n')
        emit(batch)
        return
    if mode != 'submit':
        raise SystemExit('Usage: fixture.py start | submit JSON_PROPOSAL')
    state = json.loads(STATE.read_text())
    if state['current'] is None:
        raise SystemExit('Exercise already completed.')
    proposal = json.loads(sys.argv[2])
    expected = best(state['current'])
    accepted = proposal == expected
    with AUDIT.open('a') as stream:
        stream.write(json.dumps({'event': 'submission', 'actual': proposal,
                                 'expected': expected, 'accepted': accepted}) + '\n')
    if not accepted:
        print(json.dumps({'status': 'rejected', 'current_inventory': state['current']}))
        return
    completed = state['completed'] + 1
    print(json.dumps({'status': 'accepted', 'completed': completed}))
    if completed == ROUNDS:
        STATE.write_text(json.dumps({'completed': completed, 'current': None}))
        print('EXERCISE_COMPLETE accepted_rounds=' + str(completed))
        return
    batch = fresh(completed)
    STATE.write_text(json.dumps({'completed': completed, 'current': batch}))
    with AUDIT.open('a') as stream:
        stream.write(json.dumps({'event': 'inventory', 'batch': batch}) + '\n')
    emit(batch)


if __name__ == '__main__':
    main()
