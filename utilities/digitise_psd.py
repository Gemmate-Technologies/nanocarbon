# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
Convert a WebPlotDigitizer CSV export into the PSD .npy nanocarbon reads.

Usage: python utilities/digitise_psd.py --csv wpd_export.csv --out use_cases/data/my_psd.npy [--x-units nm|A] [--no-normalise]

The library expects a (K, 2) array: pore diameter in Angstrom, and the distribution value
(dV/dD for a volume PSD, the psd_kind='volume' default). WebPlotDigitizer
(https://automeris.io/wpd/) exports a two-column CSV; this script sorts and deduplicates it,
converts the diameters from nm (the default; mesoporous papers usually report nm), divides
the values by their integral unless --no-normalise is given (only the shape matters), and
saves the .npy with a PNG preview, <out>.preview.png, of the raw and converted curves to
catch digitization errors before they reach a batch.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

if os.environ.get('MPLBACKEND') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def _convert(csv_path: str, x_units: str = 'nm', normalise: bool = True) -> np.ndarray:
    """
    Read the CSV and return the PSD in the library's units.

    Parameters
    ----------
    csv_path : str
        WebPlotDigitizer export (x, y), no header.
    x_units : {'nm', 'A'}, optional
        Unit of the diameters in the CSV. Default 'nm'.
    normalise : bool, optional
        Divide the values by their integral over the diameters. Default True.

    Returns
    -------
    (K, 2) ndarray
        Diameter (Angstrom), value; sorted, deduplicated, negative values clipped to 0.

    Raises
    ------
    ValueError
        On an unknown x_units, or a non-positive integral when normalizing.
    """
    df = pd.read_csv(csv_path, header=None, names=['x', 'y'])
    df = df.dropna().sort_values('x').reset_index(drop=True)
    df = df.drop_duplicates(subset=['x'], keep='first').reset_index(drop=True)
    df['y'] = df['y'].clip(lower=0.0)

    # Unit conversion on the diameter axis.
    if x_units == 'nm':
        df['x'] *= 10.0
    elif x_units == 'A':
        pass
    else:
        raise ValueError(f'Unknown x_units {x_units!r}; use "nm" or "A"')

    # to_numpy(), not .values: a plain ndarray for np.column_stack
    arr = np.column_stack([df['x'].to_numpy(), df['y'].to_numpy()])

    if normalise:
        integral = float(np.trapezoid(arr[:, 1], arr[:, 0]))
        if integral <= 0:
            raise ValueError(
                f'PDF integral is non-positive ({integral!r}); '
                'check the CSV — y values may be all zero or negative.'
            )
        arr[:, 1] /= integral

    return arr


def _save_preview_png(
    csv_path: str,
    arr: np.ndarray,
    out_png: str,
    x_units: str,
) -> None:
    """
    Save a side-by-side preview of the raw CSV and the converted array, which catches an axis
    flip or a wrong factor at a glance.

    Parameters
    ----------
    csv_path : str
        The CSV.
    arr : (K, 2) ndarray
        The converted PSD.
    out_png : str
        Output image.
    x_units : str
        Unit of the raw diameters, for the axis label.
    """
    raw = pd.read_csv(csv_path, header=None, names=['x', 'y']).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(raw['x'], raw['y'], 'o-', ms=3)
    axes[0].set_xlabel(f'pore diameter [{x_units} — raw CSV]')
    axes[0].set_ylabel('density (raw)')
    axes[0].set_title('Source (WPD export)')
    axes[0].grid(alpha=0.3)

    axes[1].plot(arr[:, 0], arr[:, 1], 'o-', ms=3)
    axes[1].set_xlabel('pore diameter [Å]')
    axes[1].set_ylabel('PDF (normalised to integrate to 1)')
    axes[1].set_title('Converted (.npy contents)')
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """
    CLI entry point; see the module docstring.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; None reads sys.argv.
    """
    parser = argparse.ArgumentParser(
        description='Convert a WebPlotDigitizer PSD CSV into a .npy array.',
    )
    parser.add_argument('--csv', required=True, help='Input CSV (WPD export).')
    parser.add_argument(
        '--out', required=True,
        help='Output .npy path (Nx2: diameter Å, PDF).',
    )
    parser.add_argument(
        '--x-units', default='nm', choices=('nm', 'A'),
        help='Units of the x axis in the CSV.  Default: nm.',
    )
    parser.add_argument(
        '--no-normalise', action='store_true',
        help=(
            'Do NOT divide the y values by the integral.  Use if the '
            'CSV already contains a normalised PDF.'
        ),
    )
    args = parser.parse_args(argv)

    arr = _convert(
        args.csv,
        x_units=args.x_units,
        normalise=not args.no_normalise,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    np.save(args.out, arr)

    preview_png = args.out + '.preview.png'
    _save_preview_png(args.csv, arr, preview_png, args.x_units)

    print(f'[ok] wrote {args.out} ({arr.shape[0]} points)')
    print(f'[ok] preview: {preview_png}')
    print(f'[ok] integral after normalisation: '
          f'{np.trapezoid(arr[:, 1], arr[:, 0]):.4f} (should be ~1.0)')


if __name__ == '__main__':
    main()
