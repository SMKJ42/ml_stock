use burn::data::dataset::{Dataset, InMemDataset};
use serde::{Deserialize, Serialize};

use crate::price_data::CompaniesPriceData;

use super::CHUNK_SIZE;

#[derive(Copy, Deserialize, Serialize, Debug, Clone)]
pub struct BurnPriceDataItem {
    pub data: [f64; 32],
    pub target: f64,
}

#[derive(Copy, Deserialize, Serialize, Debug, Clone)]
pub struct NormBurnPriceDataItem {
    pub data: [f64; 32],
    pub target: f64,
    pub min: f64,
    pub max: f64,
}

impl NormBurnPriceDataItem {
    pub fn from_data_vec(
        data: Vec<f64>,
        target: f64,
        min: f64,
        max: f64,
    ) -> Result<NormBurnPriceDataItem, ()> {
        if data.len() != 32 {
            return Err(());
        }
        let mut data_array = [0.0; 32];
        for i in 0..32 {
            data_array[i] = data[i];
        }
        Ok(NormBurnPriceDataItem {
            data: data_array,
            target,
            min,
            max,
        })
    }
}

impl BurnPriceDataItem {
    pub fn from_data_vec(data: Vec<f64>, target: f64) -> Result<BurnPriceDataItem, String> {
        if data.len() != 32 {
            return Err(format!(
                "Expected data length 32, instead found {}",
                data.len()
            ));
        }
        let mut data_array = [0.0; 32];
        for i in 0..32 {
            data_array[i] = data[i];
        }
        Ok(BurnPriceDataItem {
            data: data_array,
            target,
        })
    }

    pub fn normalize(&self) -> Option<NormBurnPriceDataItem> {
        let min = self.data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if max - min == 0.0 {
            return None;
        }

        let new_data = self.data.iter().map(|x| (x - min) / (max - min)).collect();
        let target = (self.target - min) / (max - min);

        return Some(NormBurnPriceDataItem::from_data_vec(new_data, target, min, max).unwrap());
    }
}

impl Dataset<NormBurnPriceDataItem> for TrainPriceDataSet {
    fn get(&self, idx: usize) -> Option<NormBurnPriceDataItem> {
        self.dataset.get(idx)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

pub struct TrainPriceDataSet {
    pub dataset: InMemDataset<NormBurnPriceDataItem>,
}

impl TrainPriceDataSet {
    fn new(
        companies: CompaniesPriceData,
        prediction_interval: usize,
        split_val: f32,
    ) -> (Self, Self) {
        let items = TrainPriceDataSet::chunk(companies, prediction_interval);

        let split = (items.len() as f32 * split_val) as usize;
        let train_dataset = items.iter().take(split).cloned().collect();
        let test_dataset = items.iter().skip(split).cloned().collect();

        let train = TrainPriceDataSet {
            dataset: InMemDataset::new(train_dataset),
        };
        let test = TrainPriceDataSet {
            dataset: InMemDataset::new(test_dataset),
        };

        return (train, test);
    }

    fn chunk(
        companies: CompaniesPriceData,
        prediction_interval: usize,
    ) -> Vec<NormBurnPriceDataItem> {
        let mut items = Vec::new();

        for company in companies.companies {
            let mut i = 0;

            let min_len = CHUNK_SIZE + prediction_interval + 1;
            if company.price_data.len() < min_len {
                continue;
            }

            while i < company.price_data.len() - CHUNK_SIZE - prediction_interval {
                let slice = &company.price_data[i..i + CHUNK_SIZE];
                let data = slice.iter().map(|x| x.close).collect();
                let target = company.price_data[i + CHUNK_SIZE + prediction_interval].close;

                let data = BurnPriceDataItem::from_data_vec(data, target)
                    .unwrap()
                    .normalize();

                match data {
                    Some(norm_data) => items.push(norm_data),
                    None => (),
                }

                i += 1;
            }
        }

        return items;
    }
}

pub struct TrainPriceDataSetConfig {
    split_val: f32,
}

impl TrainPriceDataSetConfig {
    pub fn new(split_val: f32) -> Self {
        Self { split_val }
    }

    pub fn init(
        &self,
        companies: CompaniesPriceData,
        prediction_interval: usize,
    ) -> (TrainPriceDataSet, TrainPriceDataSet) {
        return TrainPriceDataSet::new(companies, prediction_interval, self.split_val);
    }
}
