//! Python bindings for viter: `viter._viter`.
//!
//! The Python-facing surface is documented in `python/viter/__init__.pyi`. Every long
//! running Rust call releases the GIL with [`pyo3::Python::detach`], and every Rust error
//! becomes a [`ViterError`].

mod alignment;
mod convert;
mod model;
mod serve;
mod train;

use pyo3::prelude::*;

pub(crate) use convert::err;

pyo3::create_exception!(
    viter._viter,
    ViterError,
    pyo3::exceptions::PyException,
    "Raised for every error coming out of viter's Rust core."
);

/// The `viter._viter` extension module.
#[pymodule]
fn _viter(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("ViterError", m.py().get_type::<ViterError>())?;

    m.add_class::<model::Model>()?;
    m.add_class::<alignment::Alignment>()?;
    m.add_class::<alignment::Phone>()?;
    m.add_class::<alignment::Word>()?;
    m.add_class::<alignment::AlignSummary>()?;

    m.add_function(wrap_pyfunction!(train::train, m)?)?;
    m.add_function(wrap_pyfunction!(train::import_mfa, m)?)?;
    m.add_function(wrap_pyfunction!(serve::serve, m)?)?;
    m.add_function(wrap_pyfunction!(main, m)?)?;
    Ok(())
}

/// Run the `viter` command-line interface in-process.
///
/// `argv` defaults to `sys.argv`; the first element is the program name, as in C.
/// Returns the process exit code (0 on success, 1 on error) instead of exiting, so an
/// embedder stays in control.
#[pyfunction]
#[pyo3(signature = (argv = None))]
fn main(py: Python<'_>, argv: Option<Vec<String>>) -> PyResult<i32> {
    let argv = match argv {
        Some(a) => a,
        None => py
            .import("sys")?
            .getattr("argv")?
            .extract::<Vec<String>>()
            .unwrap_or_else(|_| vec!["viter".to_string()]),
    };
    // clap needs argv[0]; an empty list would make it read the first real argument as one.
    let argv = if argv.is_empty() {
        vec!["viter".to_string()]
    } else {
        argv
    };
    viter_cli::init_logging().map_err(err)?;
    // `run_exit_code` reports `--help`/`--version` (0) and usage errors (2) as codes rather
    // than calling `std::process::exit`, which would take the interpreter down with it.
    Ok(py.detach(|| viter_cli::run_exit_code(argv)))
}
