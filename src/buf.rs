use nom::bytes::complete::{tag, take};
use nom::character::complete::{alpha1, alphanumeric0, digit1, line_ending, multispace1};
use nom::combinator::map;
use nom::error::{ErrorKind, ParseError};
use nom::multi::{count, many1, separated_list};
use nom::number::complete::double;
use nom::sequence::{separated_pair, terminated, tuple};
use nom::IResult;
use ndarray::{Array2, Array};
use std::collections::HashMap;
use thiserror::Error;
use time::OffsetDateTime;

#[derive(Debug, Error, PartialEq)]
pub enum BufError<I: std::fmt::Debug> {
    #[error("Unknown parsing error")]
    Unknown(String),
    #[error("A parsing error")]
    NomError(I, ErrorKind),
}

impl<I: std::fmt::Debug> ParseError<I> for BufError<I> {
    fn from_error_kind(input: I, kind: ErrorKind) -> Self {
        BufError::NomError(input, kind)
    }

    fn append(_: I, _: ErrorKind, other: Self) -> Self {
        other
    }
}

fn key_time_curr(time: &str) -> OffsetDateTime {
    OffsetDateTime::parse(format!("20{} +0000", time), "%Y%m%d/%H%M %z").unwrap()
}

#[derive(Debug)]
pub struct StationInfo {
    stid: String,
    stnm: u64,
    slat: f64,
    slon: f64,
    selv: f64,
}

#[derive(Debug)]
pub struct TimeInfo {
    time: OffsetDateTime,
    stim: u64,
}

#[derive(Debug)]
pub struct BufFile {
    stninfo: StationInfo,
    times: Vec<TimeInfo>,
    columns: HashMap<String, Vec<Vec<f64>>>,
    integrated: HashMap<String, Vec<f64>>,
    surface: HashMap<String, Vec<Option<f64>>>,
}



impl BufFile {

    /// Return an owned array of time (rows) vs height (columns)
    pub fn timevsheight(&self) -> HashMap<String, Array2<f64>> {
        // Flatten and collect
        let mut data = HashMap::new();
        for (key, value) in self.columns.iter() {
            let oned: Vec<_> = value.iter().flatten().map(|x| *x).collect();
            let shape = (value.len(), value[0].len());
            data.insert(key.to_owned(), Array::from_shape_vec(shape, oned).expect("Messages have different number of levels"));
        }
        data
    }

    pub fn timestamps(&self) -> Vec<i64> {
        self.times.iter().map(|x| x.time.timestamp()).collect()
    }

    pub fn model_hour(&self) -> u8 {
        self.times[0].time.hour()
    }

    pub fn parse(buf: &str) -> Result<Self, BufError<&str>> {
        // Get the header
        let (rest, (_, (_, snparm), _, (_, stnprm), _, _)) = match tuple((
            line_ending,
            key_head,
            line_ending,
            key_head,
            line_ending,
            line_ending,
        ))(buf)
        {
            Ok(s) => s,
            Err(e) => return Err(BufError::Unknown(e.to_string())),
        };

        // Declare main body of parser
        let stninfo_parse = map(
            tuple((
                separated_pair(alpha1, tag(" = "), alpha1),
                space,
                key_int,
                space,
                key_time,
                space,
                key_num,
                space,
                key_num,
                space,
                key_num,
                space,
                key_int,
                space,
            )),
            |(
                (_, stid),
                _,
                (_, stnm),
                _,
                time,
                _,
                (_, slat),
                _,
                (_, slon),
                _,
                (_, selv),
                _,
                (_, stim),
                _,
            )| {
                (
                    StationInfo {
                        stid: stid.to_owned(),
                        stnm,
                        slat,
                        slon,
                        selv,
                    },
                    TimeInfo { time, stim },
                )
            },
        );
        let kvspace = terminated(key_num, space);

        let integrated_parse = map(many1(kvspace), |x| x.into_iter().collect::<HashMap<_, _>>());

        let alphaspace = terminated(alpha1, space);

        let columnheader_parse = many1(alphaspace);

        let colkeylen = snparm.len();
        let parse_level = count(numspace, colkeylen);
        let column_parse = map(many1(parse_level), |rows| {
            let mut columns = HashMap::new();
            for key in snparm.iter() {
                columns.insert(key.to_string(), Vec::new());
            }
            for row in rows {
                for (idx, v) in row.iter().enumerate() {
                    columns.get_mut(snparm[idx]).unwrap().push(*v);
                }
            }
            columns
        });

        let atm_parse = tuple((
            stninfo_parse, // Takes care of line_ending
            integrated_parse,
            columnheader_parse,
            column_parse,
        ));
        let collect_atm = map(many1(atm_parse), |msgs| {
            let mut times = Vec::new();
            let mut integrated = HashMap::new();
            let mut columns = HashMap::new();
            for key in snparm.iter() {
                columns.insert(key.to_string(), Vec::new());
            }
            for key in stnprm.iter() {
                integrated.insert(key.to_string(), Vec::new());
            }
            let mut ostninfo = None;
            for msg in msgs {
                let ((stninfo, timeinfo), integr, _, col) = msg;
                times.push(timeinfo);
                for (key, value) in integr {
                    integrated.get_mut(key).unwrap().push(value);
                }
                for (key, value) in col {
                    columns.get_mut(&key).unwrap().push(value);
                }
                ostninfo = Some(stninfo);
            }
            (ostninfo.unwrap(), times, integrated, columns)
        });

        // Execute parser of atm messages
        let (rest, (stninfo, times, integrated, columns)) = match collect_atm(rest) {
            Ok(s) => s,
            Err(e) => return Err(BufError::Unknown(e.to_string())),
        };

        // Get header for surface messages
        let (rest, _) = match surface_start_tag(rest) {
            Ok(s) => s,
            Err(e) => return Err(BufError::Unknown(e.to_string())),
        };
        let varname = map(tuple((alpha1, alphanumeric0)), |(s, e)| {
            format!("{}{}", s, e)
        });
        let anspace = terminated(varname, space);
        let (rest, surf_header) = match many1(anspace)(rest) {
            Ok(s) => s,
            Err(e) => return Err(BufError::Unknown(e.to_string())),
        };

        // Declare parser for surface messages
        let surf_row = tuple((
            intparse,
            space,
            timeparse,
            space,
            count(numspace, surf_header.len()),
        ));
        let surf_parser = map(many1(surf_row), |rows| {
            let mut data = HashMap::new();
            for key in surf_header.iter() {
                data.insert(key.to_string(), Vec::new());
            }
            let mut times = Vec::new();
            for (_, _, time, _, vars) in rows {
                times.push(time);
                for (idx, var) in vars.iter().enumerate() {
                    data.get_mut(&surf_header[idx])
                        .unwrap()
                        .push(if *var == -9999. { None } else { Some(*var) });
                }
            }
            (data, times)
        });
        let (_, (surface, _)) = match surf_parser(rest) {
            Ok(s) => s,
            Err(e) => return Err(BufError::Unknown(e.to_string())),
        };

        Ok(BufFile {
            stninfo,
            times,
            integrated,
            columns,
            surface,
        })
    }
}

// All the small functions used in parsing

fn key_head(i: &str) -> IResult<&str, (&str, Vec<&str>)> {
    separated_pair(alpha1, tag(" = "), separated_list(tag(";"), alpha1))(i)
}

fn key_num(i: &str) -> IResult<&str, (&str, f64)> {
    separated_pair(alpha1, tag(" = "), double)(i)
}

fn intparse(i: &str) -> IResult<&str, u64> {
    map(digit1, |x: &str| x.parse::<u64>().unwrap())(i)
}

fn key_int(i: &str) -> IResult<&str, (&str, u64)> {
    separated_pair(alpha1, tag(" = "), intparse)(i)
}

fn surface_start_tag(i: &str) -> IResult<&str, &str> {
    tag("STN YYMMDD/HHMM ")(i)
}

fn space(i: &str) -> IResult<&str, &str> {
    multispace1(i)
}

fn timeparse(i: &str) -> IResult<&str, OffsetDateTime> {
    map(take(11usize), |time| key_time_curr(time))(i)
}

fn key_time(i: &str) -> IResult<&str, OffsetDateTime> {
    map(
        separated_pair(tag("TIME"), tag(" = "), timeparse),
        |(_, t)| t,
    )(i)
}

fn numspace(i: &str) -> IResult<&str, f64> {
    terminated(double, space)(i)
}
