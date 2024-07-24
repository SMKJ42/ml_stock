use std::{
    error::Error,
    fs::{self, File},
};

use super::{CompaniesPriceData, CompanyPriceData};
use chrono::NaiveDate;
use serde::{Deserialize, Deserializer};

#[derive(Debug, Clone)]
pub struct DataConfig {
    pub train_start: NaiveDate,
    pub train_end: NaiveDate,
    pub train_companies: CompaniesPriceData,
    pub validate_start: NaiveDate,
    pub validate_end: NaiveDate,
    pub validate_companies: CompaniesPriceData,
}

impl DataConfig {
    pub fn init(&mut self) -> Self {
        println!("Fetching price data...");

        self.train_companies
            .refresh_data(self.train_start, self.train_end);
        self.validate_companies
            .refresh_data(self.validate_start, self.validate_end);

        self.to_owned()
    }

    pub fn new() -> Result<DataConfig, Box<dyn Error>> {
        let file_path = "config.json";
        let config_file = File::open(file_path.clone())?;
        let data_config: DataConfig = serde_json::from_reader(config_file)?;
        return Ok(data_config);
    }
}

impl<'de> Deserialize<'de> for DataConfig {
    fn deserialize<D>(deserializer: D) -> Result<DataConfig, D::Error>
    where
        D: Deserializer<'de>,
    {
        let data = DataConfigJson::deserialize(deserializer)?;
        let dates = data.dates;
        let train_start = dates.train_start;
        let train_end = dates.train_end;
        let validate_start = dates.valid_start;
        let validate_end = dates.valid_end;

        let mut companies = CompaniesPriceData::new();

        let forbes2000 = gather_companies(data.forbes2000, "forbes2000".to_string());
        let nasdaq = gather_companies(data.nasdaq, "nasdaq".to_string());
        let nyse = gather_companies(data.nyse, "nyse".to_string());
        let sp500 = gather_companies(data.sp500, "sp500".to_string());

        companies.append(forbes2000);
        companies.append(nasdaq);
        companies.append(nyse);
        companies.append(sp500);

        return Ok(DataConfig {
            train_start,
            train_end,
            validate_start,
            validate_end,
            train_companies: companies.clone(),
            validate_companies: companies,
        });
    }
}

#[derive(Debug, Deserialize)]
pub struct DataConfigJson {
    pub dates: DataConfigDates,
    pub forbes2000: Vec<String>,
    pub nasdaq: Vec<String>,
    pub nyse: Vec<String>,
    pub sp500: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct DataConfigDates {
    pub train_start: NaiveDate,
    pub train_end: NaiveDate,
    pub valid_start: NaiveDate,
    pub valid_end: NaiveDate,
}

pub fn gather_companies(data: Vec<String>, exchange: String) -> CompaniesPriceData {
    let mut companies = CompaniesPriceData::new();
    for symbol in data {
        match symbol.as_str() {
            "all" => {
                companies.flush();
                fs::read_dir(format!("stock_market_data/{exchange}/csv"))
                    .unwrap()
                    .for_each(|entry| {
                        let entry = entry.unwrap();
                        let path = entry.path();
                        let symbol = path.file_stem().unwrap().to_str().unwrap();
                        companies.push(CompanyPriceData::new(symbol.to_string(), exchange.clone()));
                    });
                break;
            }
            _ => {
                companies.push(CompanyPriceData::new(symbol.to_string(), exchange.clone()));
            }
        }
    }
    return companies;
}
