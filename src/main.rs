use clap::{App, Arg};
use cpython::{PyDict, PyErr, Python};
use ndarray::parallel::par_azip;
use ndarray::Array2;
use ordered_float::OrderedFloat;
use std::collections::HashMap;
use thiserror::Error;
use time::OffsetDateTime;

mod buf;
use buf::*;

#[derive(Debug, Error)]
enum ChartError {
    #[error("Error with python")]
    Python,
    #[error("Error with reqwest")]
    Reqwest(#[from] reqwest::Error),
}

fn get_date_from_buf(buf: &str) -> &str {
    buf.split('\n')
        .nth(4)
        .expect("Malformed buf file: not enough lines")
        .split(" = ")
        .nth(3)
        .expect("Malformed buf file: line too short")
}

fn build_url(model: &str, hour: &str, site: &str) -> String {
    // I'd prefer not to allocate this every time but can't build a constant
    // HashMap<String, &str> and HashMap<&'static str, &'static str> doesn't work
    let lowers: HashMap<String, &str> = [("gfs".to_string(), "gfs3"), ("nam".to_string(), "nam")]
        .iter()
        .cloned()
        .collect();
    format!(
        "http://www.meteo.psu.edu/bufkit/data/{}/{}/{}_{}.buf",
        model.to_ascii_uppercase(),
        hour,
        lowers[&model.to_ascii_lowercase()],
        site.to_ascii_lowercase()
    )
}

fn fetch_data(
    site: &str,
    hour: Option<&str>,
    model: &str,
) -> std::result::Result<String, reqwest::Error> {
    match hour {
        Some(h) => {
            let url = build_url(model, h, site);
            reqwest::blocking::get(&url)?.text()
            //Ok(std::fs::read_to_string("gfs3_kcon.buf").unwrap())
        }
        None => {
            let current_time = OffsetDateTime::now_utc();
            let (first_try, second_try) = match current_time.hour() {
                0..=5 => ("00", "18"),
                6..=11 => ("06", "00"),
                12..=17 => ("12", "06"),
                18..=23 => ("18", "12"),
                _ => panic!("Shouldn't be here"),
            };
            let first_url = build_url(model, first_try, site);
            let second_url = build_url(model, second_try, site);
            let first_text = reqwest::blocking::get(&first_url)?.text()?;
            let second_text = reqwest::blocking::get(&second_url)?.text()?;
            //let first_text = std::fs::read_to_string("gfs3_kcon.buf").unwrap();
            //let second_text = std::fs::read_to_string("gfs3_kcon.buf").unwrap();
            let first_date = get_date_from_buf(&first_text);
            let second_date = get_date_from_buf(&second_text);
            if first_date > second_date {
                Ok(first_text)
            } else {
                Ok(second_text)
            }
        }
    }
}

fn windvect(drct: &Array2<f64>, sknt: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let mut u = Array2::default(drct.raw_dim());
    let mut v = Array2::default(drct.raw_dim());
    par_azip!((u in &mut u, v in &mut v, &drct in drct, &sknt in sknt) { *u = sknt * (-drct+270.).to_radians().cos(); *v = sknt * (-drct+270.).to_radians().sin();});
    (u, v)
}

fn calc_rh(tc: &Array2<f64>, dptc: &Array2<f64>) -> Array2<f64> {
    // Use the Magnus formula from Wikipedia
    let mut rh = Array2::default(dptc.raw_dim());
    let b = 18.678;
    let c = 257.14;
    par_azip!((rh in &mut rh, t in tc, dp in dptc) {*rh = (dp*b/(c+dp) - t*b/(c+t)).exp() * 100.});
    rh
}

// Take pressure because final product will have P as y-axis
// Have to do logs somewhere, so it might as well be here
fn calc_fzlvl(tc: &Array2<f64>, pres: &Array2<f64>) -> Vec<f64> {
    let mut freezing = Vec::with_capacity(tc.shape()[1]);
    for (ts, ps) in tc.outer_iter().zip(pres.outer_iter()) {
        // Just straight up iterate looking for where ts goes from positive to negative
        // It could happen more than once but we only take the first
        let idx_of_first_neg = ts
            .into_iter()
            .enumerate()
            .find_map(|(i, val)| {
                if val.is_sign_negative() {
                    Some(i)
                } else {
                    None
                }
            })
            .expect("Whole column is above freezing");

        // If the surface is below freezing, return nan
        if idx_of_first_neg == 0 {
            freezing.push(std::f64::NAN);
        } else {
            // Do a log interpolation to the correct height
            let tdiff = ts[idx_of_first_neg] - ts[idx_of_first_neg - 1];
            let pdiff = (ps[idx_of_first_neg] / ps[idx_of_first_neg - 1]).ln();
            let pfrez =
                (ps[idx_of_first_neg - 1].ln() - ts[idx_of_first_neg - 1] * pdiff / tdiff).exp();
            freezing.push(pfrez);
        }
    }
    freezing
}

fn calc_unstable(thetae: &Array2<f64>) -> Array2<bool> {
    let mut unstable = Array2::default(thetae.raw_dim());
    for t in 0..(thetae.shape()[0]) {
        for p in 0..(thetae.shape()[1] - 2) {
            // Original has condition to skip low levels
            unstable[[t, p]] =
                thetae[[t, p]] > thetae[[t, p + 1]] && thetae[[t, p]] > thetae[[t, p + 2]];
        }
    }
    unstable
}

fn main() -> Result<(), ChartError> {
    let matches = App::new("Time Height Generator")
        .version("0.1")
        .author("Alex Weech <aweech340@gmail.com>")
        .about("Generates nice time-height charts")
        .arg(
            Arg::with_name("site")
                .help("The station id")
                .index(1)
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("model")
                .help("Model to use")
                .index(2)
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("run")
                .short("r")
                .long("run")
                .takes_value(true)
                .help("The model run to use"),
        )
        .get_matches();
    let site = matches
        .value_of("site")
        .expect("Clap should take care of this");
    let model = matches
        .value_of("model")
        .expect("Clap should take care of this");
    let hour = matches.value_of("run");

    let data = fetch_data(site, hour, model)?;
    let buf = BufFile::parse(&data).expect("Bad buf file");
    let timeheight_dict = buf.timevsheight();

    // Plot has u, v, thetae, rh, freezing level, and dthetae/dp
    // Given pressure, temp, wet bulb, dewpoint, thetae, wind direction,
    // wind speed, vertical velocity, and height
    // Need to calculate u, v, rh, freezing level, and whether dthetae/dp > 0
    let (u, v) = windvect(&timeheight_dict["DRCT"], &timeheight_dict["SKNT"]);
    let rh = calc_rh(&timeheight_dict["TMPC"], &timeheight_dict["DWPC"]);
    let frzlevel = calc_fzlvl(&timeheight_dict["TMPC"], &timeheight_dict["PRES"]);
    let unstable = calc_unstable(&timeheight_dict["THTE"]);
    let times = buf.timestamps();
    let pmax = timeheight_dict["PRES"]
        .iter()
        .map(|x| OrderedFloat(*x))
        .max()
        .unwrap()
        .into_inner();
    let pticks = {
        let mut ticks = Vec::new();
        let mut p = pmax.ceil();
        // Go up to the next multiple of 50
        while p % 50. != 0. {
            p += 1.;
        }
        // Make ticks every 50 hPa up to 250
        while p >= 250. {
            ticks.push(p);
            p -= 50.;
        }
        ticks
    };

    // Time to load in Python
    // Follow the strategy of sending all the variables over
    // and then just having a script to minimize copying
    let gil = Python::acquire_gil();
    let py = gil.python();
    let pyenv = move |py| {
        let locals = PyDict::new(py);
        locals.set_item(py, "u", to_vec_vec(&u))?;
        locals.set_item(py, "v", to_vec_vec(&v))?;
        locals.set_item(py, "rh", to_vec_vec(&rh))?;
        locals.set_item(py, "frzlevel", frzlevel)?;
        locals.set_item(py, "unstable", to_vec_vec(&unstable))?;
        locals.set_item(py, "thetae", to_vec_vec(&timeheight_dict["THTE"]))?;
        locals.set_item(py, "pres", to_vec_vec(&timeheight_dict["PRES"]))?;
        locals.set_item(py, "times", times)?;
        locals.set_item(py, "site", site.to_ascii_uppercase())?;
        locals.set_item(py, "model", model.to_ascii_uppercase())?;
        locals.set_item(py, "hour", buf.model_hour())?;
        locals.set_item(py, "pticks", pticks)?;

        py.run(
            r#"
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import datetime
import numpy as np
dttimes = []
for t in times:
    dttimes.append(mdates.date2num(datetime.datetime.fromtimestamp(t)))

rhcolors = np.array([
    [160, 140, 0],
    [200, 180, 0],
    [255, 255, 0],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [0, 255, 0],
    [0, 200, 0],
    [0, 150, 0],
    [0, 150, 0]])/255.

pres = np.array(pres)
rh = np.array(rh)
thetae = np.array(thetae)
u = np.array(u)
v = np.array(v)
unstable = np.array(unstable)

fig, ax = plt.subplots(figsize=(13, 9))
ax.set_yscale("log")
ax.set_ylim(pticks[0], pticks[-1])
ax.set_xlim(dttimes[-1], dttimes[0])
ax.grid(linestyle="--")

mesht, _ = np.meshgrid(dttimes, pres[0:, 0])
ax.contourf(mesht, pres, rh, levels=range(0, 101, 10), colors=rhcolors)
ax.contour(mesht, pres, thetae, levels=range(200, 500, 2), colors=["k"], linewidths=[1])
ax.plot(dttimes, frzlevel, color="blue", linewidth=3)
ax.barbs(mesht[::3, ::3], pres[::3, ::3], u[::3, ::3], v[::3, ::3], linewidth=0.5)
ax.contour(mesht, pres, unstable, levels=[-0.5, 0.5], colors=["r"], linewidths=[2])

ax.set_title(f"{site} {model} Time-Height")
ax.set_ylabel("Pressure (hPa)")
ax.set_yticks(pticks)
ax.get_yaxis().set_major_formatter(mtick.ScalarFormatter())
ax.get_xaxis().set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%a\n%HZ"))

fig.savefig(f'{site}_{model}_{hour:02}z_th.png')
plt.show()
        "#,
            None,
            Some(&locals),
        )?;

        Ok(())
    };
    pyenv(py).map_err(|e: PyErr| {
        e.print_and_set_sys_last_vars(py);
        ChartError::Python
    })
}

// Straight up copy the elements
fn to_vec_vec<T: Clone>(arr: &Array2<T>) -> Vec<Vec<T>> {
    let mut outer = Vec::with_capacity(arr.shape()[0]);
    for row in arr.t().outer_iter() {
        outer.push(row.to_vec());
    }
    outer
}
