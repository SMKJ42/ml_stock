use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, Data, Tensor},
};

use super::data_loader::NormBurnPriceDataItem;

#[derive(Clone)]
pub struct PriceDataBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> PriceDataBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct PriceDataBatch<B: Backend> {
    pub data: Tensor<B, 2>,
    pub targets: Tensor<B, 1>,
}

impl<B: Backend> Batcher<NormBurnPriceDataItem, PriceDataBatch<B>> for PriceDataBatcher<B> {
    fn batch(&self, items: Vec<NormBurnPriceDataItem>) -> PriceDataBatch<B> {
        let data = items
            .iter()
            .map(|row| Data::<f64, 1>::from(row.data))
            .map(|data| Tensor::<B, 1>::from_data(data.convert(), &self.device))
            .map(|data| data.reshape([1, 32]))
            .collect();

        let targets = items
            .iter()
            .map(|row| Data::<f64, 1>::from([row.target]))
            .map(|data| Tensor::<B, 1>::from_data(data.convert(), &self.device))
            .collect();

        let data = Tensor::cat(data, 0);
        let targets = Tensor::cat(targets, 0);

        return PriceDataBatch { data, targets };
    }
}
