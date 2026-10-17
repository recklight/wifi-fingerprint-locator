"""Beamforming-report capture files: bit layout, round trips, permutations and damaged files."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from wifi_fingerprint_locator.beamform_report import (
    BEAMFORM_REPORT_CODE,
    HEADER_SIZE,
    ChannelSnapshot,
    RadioMeta,
    encode_beamform_payload,
    expected_csi_length,
    frame_bytes,
    pack_csi,
    parse_beamform_payload,
    read_beamform_file,
    summarize_capture,
    unpack_csi,
    unwrap_timestamps,
    write_beamform_file,
)
from wifi_fingerprint_locator.errors import BeamformFormatError


def reference_unpack(block, n_tx, n_rx):
    """Bit-by-bit decoder written straight from the documented layout."""
    data = [*block, 0]
    csi = np.zeros((n_tx, n_rx, 30), dtype=complex)
    index = 0
    for subcarrier in range(30):
        index += 3
        remainder = index % 8
        for rx in range(n_rx):
            for tx in range(n_tx):
                values = []
                for offset in (0, 1):
                    byte = index // 8 + offset
                    raw = ((data[byte] >> remainder) | (data[byte + 1] << (8 - remainder))) & 0xFF
                    values.append(raw - 256 if raw > 127 else raw)
                csi[tx, rx, subcarrier] = values[0] + 1j * values[1]
                index += 16
    return csi


def meta_for(n_tx=1, n_rx=3, permutation=(0, 1, 2), valid=True, rate=0x0101):
    rssi = (40, 41, 42)[:n_rx] + (0,) * (3 - n_rx)
    return RadioMeta(5, n_tx, n_rx, rssi, -90, 40, permutation[:n_rx], valid, rate)


def random_csi(rng, n_tx, n_rx):
    return rng.integers(-128, 128, (n_tx, n_rx, 30)) + 1j * rng.integers(
        -128, 128, (n_tx, n_rx, 30)
    )


@pytest.mark.parametrize(("n_tx", "n_rx"), [(1, 1), (1, 3), (2, 3), (3, 3), (2, 2)])
def test_vectorized_decoder_matches_the_bit_by_bit_reference(rng, n_tx, n_rx):
    block = rng.integers(0, 256, expected_csi_length(n_tx, n_rx), dtype=np.uint8).tobytes()
    assert np.array_equal(unpack_csi(block, n_tx, n_rx), reference_unpack(block, n_tx, n_rx))


@pytest.mark.parametrize(("n_tx", "n_rx"), [(1, 3), (2, 3), (3, 2)])
def test_pack_and_unpack_are_inverse(rng, n_tx, n_rx):
    csi = random_csi(rng, n_tx, n_rx)
    assert np.array_equal(unpack_csi(pack_csi(csi), n_tx, n_rx), csi)


def test_csi_length_follows_the_documented_formula():
    for n_tx in (1, 2, 3):
        for n_rx in (1, 2, 3):
            assert expected_csi_length(n_tx, n_rx) == (30 * (n_rx * n_tx * 16 + 3) + 7) // 8


def test_pack_rejects_values_outside_eight_bits():
    with pytest.raises(BeamformFormatError, match="integers in"):
        pack_csi(np.full((1, 3, 30), 200.0))
    with pytest.raises(BeamformFormatError, match="shaped"):
        pack_csi(np.zeros((1, 3, 29)))


def test_payload_round_trip_restores_physical_antenna_order(rng):
    csi = random_csi(rng, 2, 3)
    meta = meta_for(2, 3, permutation=(2, 0, 1))
    payload = encode_beamform_payload(ChannelSnapshot(csi, 123, meta))
    reported = unpack_csi(payload[HEADER_SIZE:], 2, 3)
    # Reported column i holds physical antenna permutation[i].
    assert np.array_equal(reported[:, 0], csi[:, 2])
    decoded = parse_beamform_payload(payload)
    assert np.array_equal(decoded.csi, csi)
    assert decoded.meta == meta and decoded.timestamp_us == 123
    unpermuted = parse_beamform_payload(payload, apply_permutation=False)
    assert np.array_equal(unpermuted.csi, reported)


def test_invalid_permutation_is_kept_in_reported_order_and_counted(tmp_path, rng):
    csi = random_csi(rng, 1, 3)
    meta = meta_for(permutation=(1, 1, 0), valid=False)
    path = tmp_path / "capture.dat"
    write_beamform_file(path, [ChannelSnapshot(csi, 0, meta)] * 2)
    reports, statistics = read_beamform_file(path)
    assert statistics.n_invalid_permutation == 2
    assert np.array_equal(reports[0].csi, csi)
    assert reports[0].meta.permutation_valid is False


def test_length_field_mismatch_is_skipped_or_raised(tmp_path, rng):
    good = encode_beamform_payload(ChannelSnapshot(random_csi(rng, 1, 3), 0, meta_for()))
    bad = bytearray(good)
    struct.pack_into("<H", bad, 16, 10)
    path = tmp_path / "capture.dat"
    path.write_bytes(frame_bytes(BEAMFORM_REPORT_CODE, bytes(bad)) + frame_bytes(0xBB, good))
    reports, statistics = read_beamform_file(path)
    assert len(reports) == 1 and statistics.n_malformed == 1
    with pytest.raises(BeamformFormatError, match="CSI length field"):
        read_beamform_file(path, strict=True)


def test_truncated_file_keeps_complete_reports(tmp_path, rng):
    snapshots = [ChannelSnapshot(random_csi(rng, 1, 3), k, meta_for()) for k in range(3)]
    path = tmp_path / "capture.dat"
    size = write_beamform_file(path, snapshots)
    path.write_bytes(path.read_bytes()[: size - 50])
    reports, statistics = read_beamform_file(path)
    assert len(reports) == 2 and statistics.n_truncated == 1
    with pytest.raises(BeamformFormatError, match="cut off"):
        read_beamform_file(path, strict=True)


def test_other_frame_codes_are_skipped(tmp_path, rng):
    path = tmp_path / "capture.dat"
    snapshot = ChannelSnapshot(random_csi(rng, 1, 3), 0, meta_for())
    write_beamform_file(path, [snapshot], extra_frames=[(0xC1, b"x" * 9), (0x01, b"")])
    reports, statistics = read_beamform_file(path)
    assert len(reports) == 1 and statistics.n_other_codes == 2


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="capture file not found"):
        read_beamform_file(tmp_path / "absent.dat")


def test_snapshot_validates_shape_and_timestamp():
    with pytest.raises(ValueError, match="disagrees"):
        ChannelSnapshot(np.zeros((2, 3, 30)), 0, meta_for(1, 3))
    with pytest.raises(ValueError, match="32-bit"):
        ChannelSnapshot(np.zeros((1, 3, 30)), 2**32, meta_for())


def test_bandwidth_comes_from_the_ht40_flag():
    assert meta_for(rate=0x0101).bandwidth_mhz == 20
    assert meta_for(rate=0x0901).bandwidth_mhz == 40


def test_timestamps_are_unwrapped_across_the_32_bit_boundary():
    raw = [2**32 - 1_000_000, 500_000, 1_500_000, 2**32 - 1, 10]
    unwrapped = unwrap_timestamps(raw)
    assert np.all(np.diff(unwrapped) > 0)
    assert int(unwrapped[1] - unwrapped[0]) == 1_500_000


def test_capture_summary_counts_streams_and_zero_entries(rng):
    snapshots = [
        ChannelSnapshot(np.zeros((1, 3, 30)), 0, meta_for(1)),
        ChannelSnapshot(random_csi(rng, 2, 3), 2_000_000, meta_for(2)),
    ]
    summary = summarize_capture(snapshots)
    assert summary["n_single_stream"] == 1 and summary["n_multi_stream"] == 1
    assert summary["n_zero_entries"] >= 90
    assert summary["duration_s"] == 2.0 and summary["bandwidth_mhz"] == 20
