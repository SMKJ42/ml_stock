use burn::backend::libtorch::LibTorchDevice;
use price_data::config::DataConfig;
pub mod ml_model;
pub mod price_data;

fn main() {
    let data_config = DataConfig::new()
        .expect("Error reading from configuration file")
        .init();

    let model = ml_model::StockPredictor::new("tmp/stock_predictor".to_string());

    let device = LibTorchDevice::default();

    model.train_model(data_config.train_companies, device.clone());
    println!("Model trained");

    model.validate_model(
        data_config.validate_companies,
        data_config.validate_start,
        data_config.validate_end,
        device.clone(),
    );
}
