use burn::data::dataloader::batcher::Batcher;
use burn::tensor::backend::Backend;
use chrono::{Datelike, Days, NaiveDate, Weekday};

use crate::{
    ml_model::{
        data::PriceDataBatcher, data_loader::BurnPriceDataItem, model::Model,
        NormCompanyPriceDataBatch,
    },
    price_data::CompaniesPriceData,
};

use super::book::{Book, Holding};

pub struct StrategyEngine<'a> {
    pub end_date: NaiveDate,
    pub date: NaiveDate,
    pub book: Book,
    pub strategy: Strategy<'a>,
    pub balance_history: Vec<(NaiveDate, f64)>,
}

impl<'a> StrategyEngine<'a> {
    pub fn new(start_date: NaiveDate, end_date: NaiveDate, strategy: Strategy) -> StrategyEngine {
        let book = Book::new(strategy.start_balance);
        StrategyEngine {
            date: start_date,
            end_date,
            book,
            strategy,
            balance_history: Vec::new(),
        }
    }

    pub fn step_day<B: Backend>(&mut self, batcher: &PriceDataBatcher<B>, model: &Model<B>) {
        let mut batches: Vec<NormCompanyPriceDataBatch<B>> =
            self.batch_set(batcher, self.date, self.strategy.companies);

        batches
            .iter_mut()
            .for_each(|batch| batch.data = model.forward(batch.data.clone()));

        batches.sort_by(|batch1, batch2| {
            let val1 = batch1.norm_delta();
            let val2 = batch2.norm_delta();
            val1.partial_cmp(&val2).unwrap()
        });

        // this need something to adapt for larger datasets, but sqrt limits the selections too much.
        let purchase_weight = batches.len() / 2;

        let selections: Vec<&NormCompanyPriceDataBatch<B>> =
            batches.iter().take((purchase_weight).max(1)).collect();

        let selections_len = selections.len();

        for selection in selections {
            self.purchase(selection, selections_len);
        }
        self.checked_sell();

        let curr_value = self.book.value(self.strategy.companies.clone(), self.date);

        self.balance_history.push((self.date, curr_value));

        self.incr_date();
    }

    fn purchase<B: Backend>(&mut self, selection: &NormCompanyPriceDataBatch<B>, count: usize) {
        let current_price = self
            .strategy
            .companies
            .iter()
            .find(|company| *company == &selection.company)
            .unwrap()
            .price_data
            .iter()
            .find(|price_data| price_data.date == self.date)
            .unwrap()
            .close;

        let weight = 1.0 / count as f64;
        let share_count = (self.strategy.start_balance * weight) / current_price;

        let norm_delta = selection.norm_delta();

        if norm_delta < 0.2 {
            let holding = Holding::new(
                selection.company.company().clone(),
                self.date,
                current_price,
                share_count as usize,
            );

            self.book.purchase(holding, current_price, self.date);
        }
    }

    fn checked_sell(&mut self) {
        let holdings = self.book.holdings.clone();

        holdings.iter().for_each(|transaction| {
            let sell_date = transaction
                .purchase_date
                .checked_add_days(Days::new(self.strategy.hold_for as u64))
                .unwrap();

            if sell_date <= self.date {
                let company = self
                    .strategy
                    .companies
                    .iter()
                    .find(|company| company.symbol == transaction.company.symbol)
                    .unwrap();

                let curr_price_idx = company
                    .price_data
                    .iter()
                    .position(|price_data| price_data.date >= self.date)
                    .unwrap()
                    - 1;

                let sale_date = company.price_data[curr_price_idx].date;
                let current_price = company.price_data[curr_price_idx].close;

                self.book
                    .sell(transaction.clone(), current_price, sale_date);
            }
        });
    }

    fn incr_date(&mut self) {
        let mut new_date = self.date.succ_opt().unwrap();
        let dow = new_date.weekday();
        if dow == Weekday::Sat {
            new_date = new_date.succ_opt().unwrap().succ_opt().unwrap();
        }
        if dow == Weekday::Sun {
            new_date = new_date.succ_opt().unwrap();
        }
        assert_ne!(self.date, new_date);

        self.date = new_date;
    }

    fn batch_set<B: Backend>(
        &self,
        batcher: &PriceDataBatcher<B>,
        curr_date: NaiveDate,
        companies: &CompaniesPriceData,
    ) -> Vec<NormCompanyPriceDataBatch<B>> {
        let price_data = companies.fetch_validate_data_set(curr_date);

        let mut batches = Vec::new();

        for item in price_data.iter() {
            let close_data = item.search_data.iter().map(|x| x.close).collect();

            // To maintain the context of the last close price, we use target as the placeholder to hold the value.
            // because this value is not getting re-read into the model it should not polute the data.
            let target = item.search_data.last().unwrap().close;
            let data = BurnPriceDataItem::from_data_vec(close_data, target)
                .unwrap()
                .normalize();

            match data {
                Some(data) => {
                    let batch = batcher.batch(vec![data.clone()]);
                    batches.push(NormCompanyPriceDataBatch::new(
                        item.company.clone(),
                        batch,
                        data.min,
                        data.max,
                    ));
                }
                None => continue,
            }
        }

        return batches;
    }
}

pub struct Strategy<'a> {
    pub companies: &'a CompaniesPriceData,
    hold_for: usize,
    start_balance: f64,
}

impl Strategy<'_> {
    pub fn new<'a>(
        companies: &'a CompaniesPriceData,
        hold_for: usize,
        start_balance: f64,
    ) -> Strategy<'a> {
        Strategy {
            companies,
            hold_for,
            start_balance,
        }
    }
}
