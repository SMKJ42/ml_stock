use std::sync::atomic::{AtomicUsize, Ordering};

use chrono::NaiveDate;

use crate::price_data::{CompaniesPriceData, Company};

pub struct Book {
    pub balance: f64,
    pub holdings: Vec<Holding>,
    pub history: Vec<Transaction>,
}

impl Book {
    pub fn new(balance: f64) -> Book {
        Book {
            balance,
            holdings: Vec::new(),
            history: Vec::new(),
        }
    }

    pub fn purchase(&mut self, mut holding: Holding, current_price: f64, date: NaiveDate) {
        if holding.count == 0 {
            return;
        }

        let value = holding.value(current_price);

        // if there are not enough funds to purchase the stock,
        // then purchase as many shares as possible.
        if self.balance < value {
            let new_share_count = (self.balance / current_price) as usize;
            holding.count = new_share_count;
            self.purchase(holding, current_price, date);
            return;
        }

        self.balance -= value;
        self.holdings.push(holding.clone());
        self.history.push(Transaction {
            side: Side::Buy,
            holding,
            date,
        });
    }

    pub fn sell(&mut self, mut holding: Holding, current_price: f64, date: NaiveDate) {
        holding.sale_price = Some(current_price);
        self.balance += holding.value(current_price);
        self.holdings.retain(|x| x.id != holding.id);
        self.history.push(Transaction {
            side: Side::Sell,
            holding,
            date,
        });
    }

    pub fn value(&self, companies_price_data: CompaniesPriceData, current_date: NaiveDate) -> f64 {
        let mut value = self.balance;
        for holding in &self.holdings {
            let company = companies_price_data
                .companies
                .iter()
                .find(|company| {
                    company.exchange == holding.company.exchange
                        && company.symbol == holding.company.symbol
                })
                .unwrap();
            let current_price = company
                .price_data
                .iter()
                .position(|price_data| price_data.date >= current_date);

            match current_price {
                Some(idx) => {
                    value += holding.value(company.price_data[idx].close);
                }
                None => {
                    if current_date > company.price_data.last().unwrap().date {
                        value += holding.value(company.price_data.last().unwrap().close);
                    } else {
                        panic!();
                    }
                }
            }
        }
        return value;
    }
}

static OBJECT_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone)]
pub struct Holding {
    pub id: usize,
    pub company: Company,
    pub purchase_date: NaiveDate,
    pub purchase_price: f64,
    pub sale_price: Option<f64>,
    pub count: usize,
}

impl Holding {
    pub fn new(
        company: Company,
        purchase_date: NaiveDate,
        purchase_price: f64,
        count: usize,
    ) -> Holding {
        Holding {
            id: OBJECT_COUNTER.fetch_add(1, Ordering::SeqCst),
            company,
            purchase_date,
            purchase_price,
            sale_price: None,
            count,
        }
    }

    pub fn value(&self, current_price: f64) -> f64 {
        return self.count as f64 * current_price;
    }

    pub fn purchase_value(&self) -> f64 {
        return self.count as f64 * self.purchase_price;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Debug, Clone)]
pub struct Transaction {
    pub side: Side,
    pub holding: Holding,
    pub date: NaiveDate,
}
