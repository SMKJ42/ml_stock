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
    let model_config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("training config could not be loaded");

    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("trained model could not be loaded");

    let model = model_config.model.init::<B>(&device).load_record(record);

    let batcher = PriceDataBatcher::<B>::new(device.clone());
    let strategy = Strategy::new(&companies_price_data, HOLD_LENGTH, 10000.0);
    let mut engine = StrategyEngine::new(start_date, end_date, strategy);

    while engine.date < engine.end_date {
        engine.step_day(&batcher, &model);
    }

    plot_model_output(engine.value_history.clone());
    plot_company_bias(&companies_price_data, start_date, end_date, &engine);
}
