"""Exercise network selection with address fixtures; never change host links."""
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from fabric_selectors import MissingFabricAddress, parse_selectors, select_fabric_addresses
from preflight import fabric_addresses


def links(first='192.168.2.1', second='192.168.2.5'):
    return [{'ifname': name, 'addr_info': [{'family': 'inet', 'local': address}]}
            for name, address in [('roce0', first), ('roce1', second)]]


class FabricSelectors(unittest.TestCase):
    def test_adjacent_slash30_networks_select_distinct_ports_on_both_nodes(self):
        for first, second in [('192.168.2.1', '192.168.2.5'), ('192.168.2.2', '192.168.2.6')]:
            with self.subTest(first=first):
                selected = select_fabric_addresses(links(first, second), ['192.168.2.0/30', '192.168.2.4/30'])
                self.assertEqual(selected, [('roce0', first), ('roce1', second)])

    def test_existing_prefixes_and_equivalent_cidrs_select_the_same_ports(self):
        addresses = links('192.168.200.10', '192.168.201.10')
        old = select_fabric_addresses(addresses, ['192.168.200.', '192.168.201.'])
        new = select_fabric_addresses(addresses, ['192.168.200.0/24', '192.168.201.0/24'])
        self.assertEqual(old, new)
        self.assertEqual(old, [('roce0', '192.168.200.10'), ('roce1', '192.168.201.10')])

    def test_mixed_formats_preserve_selector_order(self):
        self.assertEqual(select_fabric_addresses(links(), ['192.168.2.4/30', '192.168.2.1']),
                         [('roce1', '192.168.2.5'), ('roce0', '192.168.2.1')])

    def test_exact_address_does_not_match_a_longer_address(self):
        self.assertEqual(select_fabric_addresses(links('192.168.2.1', '192.168.2.10'), ['192.168.2.1']),
                         [('roce0', '192.168.2.1')])

    def test_short_prefixes_still_work(self):
        self.assertEqual(str(parse_selectors(['10.'])[0][1]), '10.0.0.0/8')
        self.assertEqual(str(parse_selectors(['192.168.'])[0][1]), '192.168.0.0/16')

    def test_slash31_and_slash32_membership(self):
        self.assertEqual(select_fabric_addresses(links('192.168.2.0', '192.168.2.2'), ['192.168.2.0/31', '192.168.2.2/32']),
                         [('roce0', '192.168.2.0'), ('roce1', '192.168.2.2')])

    def test_invalid_selectors_are_rejected(self):
        for value in ['192.168.2.0/33', '192.168.2.1/30', '256.1.2.', '192.*.', '192.168.2', '::/64', 'not-an-ip']:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Invalid FABRIC_SUBNETS'):
                parse_selectors([value])
        with self.assertRaisesRegex(ValueError, 'at least one'):
            parse_selectors([])

    def test_overlapping_and_duplicate_selectors_are_rejected(self):
        for selectors in [['192.168.2.', '192.168.2.4/30'], ['192.168.2.', '192.168.2.'],
                          ['192.168.2.0/30', '192.168.2.1']]:
            with self.subTest(selectors=selectors), self.assertRaisesRegex(ValueError, 'Overlapping'):
                parse_selectors(selectors)

    def test_broad_prefix_matching_two_ports_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Ambiguous.*roce0.*roce1'):
            select_fabric_addresses(links(), ['192.168.2.'])

    def test_multiple_addresses_on_one_port_are_rejected(self):
        addresses = links()
        addresses[0]['addr_info'].append({'family': 'inet', 'local': '192.168.2.2'})
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            select_fabric_addresses(addresses, ['192.168.2.0/30', '192.168.2.4/30'])

    def test_two_selectors_cannot_select_the_same_port(self):
        addresses = links()
        addresses[1]['ifname'] = 'roce0'
        with self.assertRaisesRegex(ValueError, 'more than once'):
            select_fabric_addresses(addresses, ['192.168.2.0/30', '192.168.2.4/30'])

    def test_missing_second_link_is_not_silently_ignored(self):
        with self.assertRaisesRegex(MissingFabricAddress, '192.168.2.4/30'):
            select_fabric_addresses(links()[:1], ['192.168.2.0/30', '192.168.2.4/30'])

    def test_ipv6_and_empty_interfaces_are_ignored(self):
        addresses = links() + [{'ifname': 'empty'}, {'ifname': 'other', 'addr_info': [{'family': 'inet6', 'local': '::1'}]}]
        self.assertEqual(select_fabric_addresses(addresses, ['192.168.2.0/30', '192.168.2.4/30']),
                         [('roce0', '192.168.2.1'), ('roce1', '192.168.2.5')])

    def test_preflight_uses_the_same_selection(self):
        selectors = ['192.168.2.0/30', '192.168.2.4/30']
        self.assertEqual(fabric_addresses(links(), selectors), set(select_fabric_addresses(links(), selectors)))
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            fabric_addresses(links(), ['192.168.2.'])

    def cli(self, addresses, *selectors, startup=False, missing_gid=False):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / 'addresses.json').write_text(json.dumps(addresses))
            ip = directory / 'ip'
            ip.write_text('#!/bin/sh\ncat "' + str(directory / 'addresses.json') + '"\n')
            ip.chmod(0o755)
            command = [sys.executable, str(ROOT / 'scripts/fabric_selectors.py'), *selectors]
            if startup:
                source = (ROOT / 'files/entrypoint.sh').read_text()
                block = source[source.index('if [[ -z "${NCCL_IB_HCA:-}"'):source.index('# The EXACT interface holding VLLM_HOST_IP')]
                script = '\n'.join([
                    'set -euo pipefail',
                    'FABRIC_SELECTOR_SCRIPT=' + shlex.quote(str(ROOT / 'scripts/fabric_selectors.py')),
                    'FABRIC_SUBNETS=' + shlex.quote(' '.join(selectors)),
                    'read -r -a FABRIC_SELECTORS <<< "$FABRIC_SUBNETS"',
                    'NCCL_IB_HCA=""; NCCL_IB_GID_INDEX=""; FABRIC_LAYOUT=mesh; ROCE_SETTLE_S=0',
                    'MISSING_GID=' + str(int(missing_gid)),
                    'fabric_port() { if [[ "$1" == roce1 && "$MISSING_GID" == 1 ]]; then return 1; fi; printf "rdma_%s 3\\n" "$1"; }',
                    block,
                ])
                # The loop calls python3 just like the container; point the
                # fixture to the same interpreter running this test suite.
                (directory / 'python3').symlink_to(sys.executable)
                command = ['bash', '-c', script]
            return subprocess.run(command,
                                  env=dict(os.environ, PATH=temp + ':' + os.environ['PATH']),
                                  capture_output=True, text=True)

    def test_startup_cli_outputs_both_interfaces(self):
        result = self.cli(links(), '192.168.2.0/30', '192.168.2.4/30')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'roce0\t192.168.2.1\nroce1\t192.168.2.5\n')

    def test_startup_cli_missing_link_returns_retryable_status(self):
        result = self.cli(links()[:1], '192.168.2.0/30', '192.168.2.4/30')
        self.assertEqual(result.returncode, 1)
        self.assertIn('No local IPv4', result.stderr)
        self.assertEqual(result.stdout, '')

    def test_startup_cli_ambiguity_is_fatal(self):
        result = self.cli(links(), '192.168.2.')
        self.assertEqual(result.returncode, 2)
        self.assertIn('Ambiguous', result.stderr)
        self.assertEqual(result.stdout, '')

    def test_validation_never_reads_host_interfaces(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/fabric_selectors.py'), '--validate', '192.168.2.0/30', '192.168.2.4/30'],
                                env=dict(os.environ, PATH='/missing'), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_actual_startup_loop_selects_both_devices(self):
        result = self.cli(links(), '192.168.2.0/30', '192.168.2.4/30', startup=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('NCCL_IB_HCA=rdma_roce0,rdma_roce1 gid=3', result.stdout)

    def test_actual_startup_loop_rejects_partial_address_discovery(self):
        result = self.cli(links()[:1], '192.168.2.0/30', '192.168.2.4/30', startup=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('FATAL: not all fabric selectors', result.stderr)
        self.assertNotIn('fabric: NCCL_IB_HCA=', result.stdout)

    def test_actual_startup_loop_rejects_partial_gid_discovery(self):
        result = self.cli(links(), '192.168.2.0/30', '192.168.2.4/30', startup=True, missing_gid=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('FATAL: not all fabric selectors', result.stderr)
        self.assertNotIn('fabric: NCCL_IB_HCA=', result.stdout)

    def test_actual_startup_loop_fails_on_ambiguous_selection(self):
        result = self.cli(links(), '192.168.2.', startup=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('Ambiguous', result.stderr)
        self.assertNotIn('fabric: NCCL_IB_HCA=', result.stdout)


if __name__ == '__main__':
    unittest.main()
