pub mod config;
pub mod parse;

use std::{
    fs::File,
    hash::{Hash, Hasher},
};

use crate::ml_model::data_loader::WIDTH;
use chrono::NaiveDate;

#[derive(Debug, Clone, Copy)]
pub struct PriceDataItem {
    pub date: NaiveDate,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: i64,
    pub adjusted_close: f64,
}

impl Hash for PriceDataItem {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.date.hash(state);
    }
}

impl PriceDataItem {
    pub fn zero() -> Self {
        PriceDataItem {
            date: NaiveDate::MIN,
            open: f64::MIN,
            high: f64::MIN,
            low: f64::MIN,
            close: f64::MIN,
            volume: i64::MIN,
            adjusted_close: f64::MIN,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Company {
    pub symbol: String,
    pub exchange: String,
}

impl PartialEq for Company {
    fn eq(&self, other: &Self) -> bool {
        self.symbol == other.symbol && self.exchange == other.exchange
    }
}

#[derive(Debug, Clone)]
pub struct CompanyPriceData {
    pub symbol: String,
    pub exchange: String,
    pub price_data: Vec<PriceDataItem>,
}

impl PartialEq for CompanyPriceData {
    fn eq(&self, other: &Self) -> bool {
        self.symbol == other.symbol && self.exchange == other.exchange
    }
}

impl CompanyPriceData {
    pub fn new(symbol: String, exchange: String) -> CompanyPriceData {
        CompanyPriceData {
            symbol: symbol,
            exchange: exchange,
            price_data: Vec::new(),
        }
    }

    pub fn refresh_data(&mut self, start: NaiveDate, end: NaiveDate) {
        self.price_data =
            parse::fetch_company_price_data(self.get_file(), start, end, self.symbol.clone());
    }

    // This function is a little bit awkward because our argument is the current date, while the function utilizes the next day.
    // This allows for more readability, while also handling the edge case of the next day not having data.
    pub fn fetch_last_n_days(
        self,
        curr_date: NaiveDate,
        window_size: usize,
    ) -> Option<Vec<PriceDataItem>> {
        let next_day_price_data = self.price_data.iter().position(|x| x.date == curr_date);

        match next_day_price_data {
            // we are predicting prices for the next day
            // therefore, we need to check if tomorrow has price data available,
            // if not, we will return None.
            None => return None,
            Some(curr_idx) => {
                let next_idx = curr_idx + 1;
                if next_idx <= window_size || next_idx >= self.price_data.len() {
                    return None;
                }
                let end_idx = next_idx - 1;
                let start_idx = end_idx - window_size;
                let price_data = self.price_data[start_idx..end_idx].to_vec();
                return Some(price_data);
            }
        }
    }

    pub fn company(&self) -> Company {
        Company {
            symbol: self.symbol.clone(),
            exchange: self.exchange.clone(),
        }
    }

    fn get_file(&self) -> File {
        let path = format!(
            "stock_market_data/{exchange}/csv/{symbol}.csv",
            exchange = self.exchange,
            symbol = self.symbol
        );

        let file = File::open(path.clone());

        match file {
            Ok(file) => file,
            Err(e) => {
                panic!(
                    "Error opening file: {path}, error: {error}",
                    path = path,
                    error = e
                );
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompaniesPriceData {
    pub companies: Vec<CompanyPriceData>,
}

pub type ClosePrice = f32;

impl CompaniesPriceData {
    pub fn new() -> CompaniesPriceData {
        CompaniesPriceData {
            companies: Vec::new(),
        }
    }

    pub fn flush(&mut self) {
        self.companies = Vec::new();
    }

    pub fn extend(&mut self, other: CompaniesPriceData) {
        self.companies.extend(other.companies);
    }

    pub fn push(&mut self, company: CompanyPriceData) {
        if company.symbol.is_empty() {
            return;
        }
        self.companies.push(company);
    }

    pub fn append(&mut self, mut other: CompaniesPriceData) {
        self.companies.append(&mut other.companies);
    }

    pub fn refresh_data(&mut self, start: NaiveDate, end: NaiveDate) {
        let total = self.companies.len();
        for (idx, company) in &mut self.companies.iter_mut().enumerate() {
            if idx % 10 == 0 {
                print!("{}[2J", 27 as char);
                println!("Fetching data... company {idx} of {total}", idx = idx,);
            }

            company.refresh_data(start, end);
        }
        print!("{}[2J", 27 as char);
        println!("Company data fetch complete");
    }

    pub fn iter(&self) -> std::slice::Iter<CompanyPriceData> {
        self.companies.iter()
    }

    pub fn fetch_validate_data_set(&self, curr_date: NaiveDate) -> Vec<SearchableCompany> {
        let set = self.fetch_last_n_days(curr_date).clone();
        // this is a crutch to get the output to intigrate with the .batch() and .step() functions later on.
        return set;
    }

    fn fetch_last_n_days(&self, curr_date: NaiveDate) -> Vec<SearchableCompany> {
        let mut items = Vec::new();
        for company in self.companies.iter() {
            let data = company.to_owned().fetch_last_n_days(curr_date, WIDTH);

            match data {
                None => continue,
                Some(price_data) => {
                    if price_data.len() < WIDTH {
                        continue;
                    }
                    items.push(SearchableCompany::new(company.clone(), price_data))
                }
            }
        }

        return items;
    }
}

#[derive(Debug, Clone)]
pub struct SearchableCompany {
    pub company: CompanyPriceData,
    pub search_data: Vec<PriceDataItem>,
}

impl SearchableCompany {
    pub fn new(company: CompanyPriceData, search_data: Vec<PriceDataItem>) -> SearchableCompany {
        SearchableCompany {
            company,
            search_data,
        }
    }

    pub fn zero() -> Self {
        SearchableCompany {
            company: CompanyPriceData {
                symbol: "".to_string(),
                exchange: "".to_string(),
                price_data: Vec::new(),
            },
            search_data: Vec::new(),
        }
    }
}
