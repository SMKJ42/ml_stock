use crate::price_data::CompaniesPriceData;

use super::{
    data::{PriceDataBatch, PriceDataBatcher},
    data_loader::TrainPriceDataSetConfig,
    model::{Model, ModelConfig},
};
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    nn::loss::{MseLoss, Reduction},
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::backend::{AutodiffBackend, Backend},
    tensor::Tensor,
    train::{
        metric::LossMetric, LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
    },
};

impl<B: Backend> Model<B> {
    pub fn forward_regression(
        &self,
        price_data: Tensor<B, 2>,
        targets: Tensor<B, 1>,
    ) -> RegressionOutput<B> {
        let targets = targets.unsqueeze();
        let output = self.forward(price_data.clone());

        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Reduction::Mean);

        RegressionOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<PriceDataBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: PriceDataBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(batch.data, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<PriceDataBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: PriceDataBatch<B>) -> RegressionOutput<B> {
        return self.forward_regression(batch.data, batch.targets);
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    pub learning_rate: f64,
    #[config(default = 0.9)]
    split_val: f32,
    #[doc = "Number of days to predict in the future"]
    prediction_interval: usize,
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    companies: CompaniesPriceData,
    device: B::Device,
) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Cannot save config");

    B::seed(config.seed);

    let batcher_train = PriceDataBatcher::<B>::new(device.clone());
    let batcher_valid = PriceDataBatcher::<B::InnerBackend>::new(device.clone());

    let (train, test) =
        TrainPriceDataSetConfig::new(config.split_val).init(companies, config.prediction_interval);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(train);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(test);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);
    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model could not be saved");
}
