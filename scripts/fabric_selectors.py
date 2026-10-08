#!/usr/bin/env python3
"""Read-only IPv4 interface selection shared by startup and preflight."""
import argparse
import ipaddress
import json
import subprocess
import sys


class MissingFabricAddress(ValueError):
    """A configured link has no address yet; startup may wait for it."""


def parse_selectors(selectors):
    if not selectors:
        raise ValueError('FABRIC_SUBNETS must contain at least one IPv4 selector')
    parsed = []
    for selector in selectors:
        try:
            if selector.endswith('.'):
                octets = selector[:-1].split('.')
                if not 1 <= len(octets) <= 3:
                    raise ValueError('prefix must contain one to three octets')
                address = '.'.join(octets + ['0'] * (4 - len(octets)))
                network = ipaddress.IPv4Network(address + '/' + str(8 * len(octets)))
            elif '/' in selector:
                network = ipaddress.IPv4Network(selector)
            else:
                network = ipaddress.IPv4Network(str(ipaddress.IPv4Address(selector)) + '/32')
        except ValueError as error:
            raise ValueError(f'Invalid FABRIC_SUBNETS selector {selector!r}: {error}') from error
        for previous, other in parsed:
            if network.overlaps(other):
                raise ValueError(f'Overlapping FABRIC_SUBNETS selectors: {previous!r} and {selector!r}')
        parsed.append((selector, network))
    return parsed


def select_fabric_addresses(addresses, selectors):
    candidates = {(link['ifname'], item['local']) for link in addresses
                  for item in link.get('addr_info', []) if item['family'] == 'inet'}
    selected = []
    for selector, network in parse_selectors(selectors):
        matches = sorted((iface, address) for iface, address in candidates
                         if ipaddress.IPv4Address(address) in network)
        if not matches:
            raise MissingFabricAddress(f'No local IPv4 address matches fabric selector {selector!r}')
        if len(matches) != 1:
            detail = ', '.join(f'{iface}={address}' for iface, address in matches)
            raise ValueError(f'Ambiguous fabric selector {selector!r}: {detail}; use one selector per port')
        iface, address = matches[0]
        if any(previous_iface == iface for previous_iface, _ in selected):
            raise ValueError(f'Fabric selectors select interface {iface!r} more than once; use distinct ports')
        selected.append((iface, address))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate', action='store_true', help='Validate selectors without reading interfaces')
    parser.add_argument('selectors', nargs='*', help='IPv4 CIDR networks, dotted prefixes or exact IPv4 addresses')
    args = parser.parse_args()
    try:
        parse_selectors(args.selectors)
        if args.validate:
            return 0
        addresses = json.loads(subprocess.check_output(['ip', '-j', '-4', 'addr'], text=True))
        for iface, address in select_fabric_addresses(addresses, args.selectors):
            print(iface + '\t' + address)
        return 0
    except MissingFabricAddress as error:
        print(error, file=sys.stderr)
        return 1
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
