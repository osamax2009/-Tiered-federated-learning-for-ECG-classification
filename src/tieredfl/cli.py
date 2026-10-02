"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import config as config_module
from .evaluation.ablation import iot_baselines
from .experiment import prepare, run_all
from .federated.tiers import DeviceTier
from .report import summarise, write_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tieredfl", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="log at debug level")
    subcommands = parser.add_subparsers(dest="command", required=True)

    run_cmd = subcommands.add_parser("run", help="run the federated comparison")
    _add_common(run_cmd)
    run_cmd.add_argument("--rounds", type=int, help="override the number of rounds")
    run_cmd.add_argument("--scale", type=float, help="fraction of each client's records to use")
    run_cmd.add_argument("--seed", type=int, help="override the run seed")
    run_cmd.add_argument(
        "--personal-weight",
        type=float,
        help="weight on the iot tier head at evaluation, in [0, 1]",
    )
    run_cmd.add_argument(
        "--noise-sigma",
        type=float,
        help="standard deviation of gaussian noise added to transmitted readouts",
    )
    run_cmd.add_argument("--no-cache", action="store_true", help="ignore the feature cache")

    cache_cmd = subcommands.add_parser("cache", help="extract features and exit")
    _add_common(cache_cmd)
    cache_cmd.add_argument("--scale", type=float, help="fraction of each client's records to use")
    cache_cmd.add_argument("--seed", type=int, help="override the run seed")

    inspect_cmd = subcommands.add_parser("inspect", help="report what data is available")
    _add_common(inspect_cmd)

    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--config", type=Path, required=True, help="path to a YAML config")
    parser.add_argument("--data-root", type=Path, help="override the dataset root")
    parser.add_argument("--output-dir", type=Path, help="override the results directory")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        cfg = _load_config(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.command == "inspect":
        return _inspect(cfg)
    if args.command == "cache":
        prepare(cfg)
        return 0
    return _run(cfg)


def _load_config(args: argparse.Namespace) -> config_module.RunConfig:
    overrides = {}
    if getattr(args, "data_root", None) is not None:
        overrides["data_root"] = str(args.data_root)
    if getattr(args, "output_dir", None) is not None:
        overrides["output_dir"] = str(args.output_dir)
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed
    if getattr(args, "scale", None) is not None:
        overrides["scale"] = args.scale
    if getattr(args, "no_cache", False):
        overrides["cache_dir"] = None

    cfg = config_module.load(args.config, overrides)

    federation_changes = {}
    if getattr(args, "rounds", None) is not None:
        federation_changes["rounds"] = args.rounds
    if getattr(args, "personal_weight", None) is not None:
        federation_changes["personal_weight"] = args.personal_weight
    if getattr(args, "noise_sigma", None) is not None:
        federation_changes["transmission_noise_sigma"] = args.noise_sigma
    if federation_changes:
        cfg = config_module.with_federation(cfg, **federation_changes)
        config_module.validate(cfg)

    return cfg


def _run(cfg: config_module.RunConfig) -> int:
    started = time.monotonic()
    outcome = run_all(cfg)
    prepared = outcome["prepared"]
    gated, baseline = outcome["gated"], outcome["baseline"]

    iot_states, iot_labels = prepared.evaluation.tier(DeviceTier.IOT)
    baselines = iot_baselines(
        global_wout=gated.final_wout,
        tier_head=outcome["tier_head"],
        features=prepared.features,
        tiers=prepared.tiers,
        states=iot_states,
        labels=iot_labels,
        n_classes=cfg.federation.n_classes,
        ridge=cfg.federation.ridge,
        present_classes=prepared.evaluation.iot_present_classes(),
    )

    duration = time.monotonic() - started
    path = write_results(
        cfg, prepared, [gated, baseline], baselines, outcome["tier_head"], duration
    )

    print()
    print(summarise(cfg, prepared, gated, baseline, baselines))
    print()
    print(f"wrote {path} in {duration / 60:.1f} min")
    return 0


def _inspect(cfg: config_module.RunConfig) -> int:
    from .data.loader import find_headers, resolve_root

    root = resolve_root(cfg.data_root)
    print(f"data root: {root}{'' if root.is_dir() else ' (missing)'}")
    print()

    wanted = {}
    for spec in cfg.clients:
        wanted[spec.dataset] = wanted.get(spec.dataset, 0) + spec.n_records

    ok = True
    for dataset, needed in sorted(wanted.items()):
        available = len(find_headers(cfg.data_root, dataset))
        status = "ok" if available >= needed else "short"
        ok &= available >= needed
        print(
            f"  {dataset:24s} {available:7d} records available, {needed:7d} requested  [{status}]"
        )

    print()
    print(f"feature cache key: {cfg.feature_cache_key()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
