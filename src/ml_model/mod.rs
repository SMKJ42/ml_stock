pub mod data;
pub mod data_loader;
pub mod inference;
pub mod model;
pub mod training;

use crate::price_data::CompaniesPriceData;
use crate::price_data::CompanyPriceData;
use burn::backend::autodiff::Autodiff;
use burn::backend::libtorch::LibTorch;
use burn::backend::libtorch::LibTorchDevice;
use burn::optim::AdamConfig;
use burn::tensor::backend::Backend;
use burn::tensor::ElementConversion;
use burn::tensor::Tensor;
use chrono::NaiveDate;
use data::PriceDataBatch;
use training::TrainingConfig;

type MyBackend = LibTorch;
type MyAudodiffBackend = Autodiff<MyBackend>;

pub const HOLD_LENGTH: usize = 5;

pub struct StockPredictor {
    pub artifact_dir: String,
}

impl StockPredictor {
    pub fn new(artifact_dir: String) -> Self {
        StockPredictor { artifact_dir }
    }

    pub fn train_model(&self, companies: CompaniesPriceData, device: LibTorchDevice) {
        println!(
            "Training model with {} companies",
            companies.companies.len()
        );

        let model_config = model::ModelConfig::new(32, 64);
        let optimizer = AdamConfig::new();
        let learning_rate = 0.0001;

        let training_config =
            TrainingConfig::new(model_config, optimizer, learning_rate, HOLD_LENGTH)
                .with_num_epochs(10);

        training::train::<MyAudodiffBackend>(
            &self.artifact_dir,
            training_config,
            companies,
            device,
        );
    }

    pub fn validate_model(
        &self,
        companies: CompaniesPriceData,
        start_date: NaiveDate,
        end_date: NaiveDate,
        device: LibTorchDevice,
    ) {
        println!(
            "Validating model with {} companies",
            companies.companies.len()
        );

        inference::infer::<MyBackend>(&self.artifact_dir, companies, start_date, end_date, device);
    }
}

#[derive(Debug, Clone)]
pub struct NormCompanyPriceDataBatch<B: Backend> {
    pub company: CompanyPriceData,
    pub data: Tensor<B, 2>,
    pub targets: Tensor<B, 1>,
    pub min: f64,
    pub max: f64,
}

impl<B: Backend> NormCompanyPriceDataBatch<B> {
    pub fn new(company: CompanyPriceData, batch: PriceDataBatch<B>, min: f64, max: f64) -> Self {
        NormCompanyPriceDataBatch {
            company,
            data: batch.data,
            targets: batch.targets,
            min,
            max,
        }
    }

    pub fn denormalize(&self) -> Tensor<B, 2> {
        (self.data.clone() * (self.max - self.min)) + self.min
    }

    pub fn denormalize_last(&self) -> Tensor<B, 1> {
        (self.targets.clone() * (self.max - self.min)) + self.min
    }

    pub fn norm_delta(&self) -> f64 {
        let prediction: f64 = self.data.clone().flatten::<1>(0, 1).into_scalar().elem();
        let actual: f64 = self.targets.clone().into_scalar().elem();
        prediction - actual
    }

    pub fn denorm_delta(&self) -> f64 {
        let prediction = self.denormalize();
        let actual = self.denormalize_last();

        let prediction: f64 = prediction.flatten::<1>(0, 1).into_scalar().elem();
        let actual: f64 = actual.into_scalar().elem();

        prediction - actual
    }
}

pub struct CompanyPriceDataOutput {
    pub company: CompanyPriceData,
    pub prediction: f64,
}