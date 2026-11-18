"""Match every self-contained witness input to the formal MATLAB exports."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import scipy.io as sio


def decoded(values):
    if isinstance(values, list):
        return [decoded(item) for item in values]
    return float.fromhex(values)


def main():
    export_dir, witness_path, report_path = map(Path, sys.argv[1:4])
    document = json.loads(witness_path.read_text(encoding="utf8"))
    actual = {}
    for path in sorted(export_dir.glob("*.mat")):
        data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        n, m = int(data["n"]), int(data["m"])
        A = [np.asarray(block, dtype=float).reshape(-1, n) for block in np.atleast_1d(data["A"])]
        b = [np.asarray(block, dtype=float).reshape(-1) for block in np.atleast_1d(data["b"])]
        points = np.atleast_1d(data["pts"])
        for point in points:
            key = path.stem, str(point.alg), int(point.run)
            if key in actual:
                raise ValueError("duplicate exported point")
            actual[key] = dict(
                x_hex=np.asarray(point.x, float).reshape(n),
                gradients_hex=np.asarray(point.JG, float).reshape(m, n),
                B_hex=np.array([block[:n, :] for block in A]),
                delta_hex=np.array([block[0] for block in b]),
                lower_hex=np.asarray(data["l"], float).reshape(n),
                upper_hex=np.asarray(data["u"], float).reshape(n),
                H_export_hex=np.asarray(point.H, float).reshape(m),
                export_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            if any(not np.all(block == block[0]) for block in b):
                raise ValueError("nonconstant support thresholds")
    if len(actual) != 134:
        raise ValueError("export coverage must be 134")
    checked = 0
    for record in document["records"]:
        key = record["prob"], record["alg"], int(record["run"])
        exported = actual.pop(key)
        if exported.pop("export_sha256") != record["export_sha256"]:
            raise ValueError("export hash mismatch")
        for field, values in exported.items():
            recovered = np.asarray(decoded(record["model"][field]), dtype=float)
            if not np.array_equal(recovered, values):
                raise ValueError("witness input mismatch %r %s" % (key, field))
        checked += 1
    if actual:
        raise ValueError("unmatched exports")
    report = dict(points=checked, mismatches=0, export_files=len(list(export_dir.glob("*.mat"))))
    report_path.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

