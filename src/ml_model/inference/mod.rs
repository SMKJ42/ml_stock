mod book;
mod engine;
mod metric;

use burn::{record::CompactRecorder, tensor::backend::Backend};
use chrono::NaiveDate;
use engine::{Strategy, StrategyEngine};
use metric::{plot_company_bias, plot_model_output};

use crate::ml_model::data::PriceDataBatcher;
use crate::ml_model::training::TrainingConfig;
use crate::price_data::CompaniesPriceData;
use burn::prelude::*;
use burn::record::Recorder;

use super::HOLD_LENGTH;

pub fn infer<B: Backend>(
    artifact_dir: &str,
    companies_price_data: CompaniesPriceData,
    start_date: NaiveDate,
    end_date: NaiveDate,
    device: B::Device,
) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("training config could not be loaded");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("trained model could not be loaded");

    let model = config.model.init::<B>(&device).load_record(record);

    let batcher = PriceDataBatcher::<B>::new(device.clone());
    let strategy = Strategy::new(&companies_price_data, HOLD_LENGTH, 10000.0);
    let mut engine = StrategyEngine::new(start_date, end_date, strategy);

    let total_days = engine.end_date.signed_duration_since(start_date).num_days();
    while engine.date < engine.end_date {
        print!("{}[2J", 27 as char);
        println!(
            "Days validated: {}/{}",
            engine.date.signed_duration_since(start_date).num_days(),
            total_days
        );
        engine.step_day(&batcher, &model);
    }

    let mut transaction_timeline: Vec<i64> = engine
        .book
        .history
        .iter()
        .enumerate()
        .map(|(idx, transaction)| match transaction.side {
            book::Side::Buy => {
                let purchase_date = transaction.date;
                let sale_date =
                    engine.book.history.iter().skip(idx).find(|x| {
                        x.holding.id == transaction.holding.id && x.side == book::Side::Sell
                    });

                match sale_date {
                    Some(sale_date) => {
                        let diff = sale_date
                            .date
                            .signed_duration_since(purchase_date)
                            .num_days();
                        return diff;
                    }
                    None => 0,
                }
            }
            _ => return 0,
        })
        .collect();
    transaction_timeline.sort();
    transaction_timeline.reverse();

    plot_model_output(engine.balance_history.clone());
    plot_company_bias(&companies_price_data, start_date, end_date, &engine);
}
