use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Lstm, LstmConfig, Relu},
    tensor::{backend::Backend, Tensor},
};

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let input_layer = LinearConfig::new(self.num_classes, self.hidden_size)
            .with_bias(true)
            .init(device);
        let output_layer = LinearConfig::new(self.hidden_size, 1)
            .with_bias(true)
            .init(device);
        let lstm = LstmConfig::new(self.hidden_size, self.hidden_size, true).init(device);

        return Model {
            input_layer,
            output_layer,
            activation: Relu::new(),
            lstm,
        };
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    input_layer: Linear<B>,
    output_layer: Linear<B>,
    activation: Relu,
    lstm: Lstm<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, price_data: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = price_data.detach();
        let x = x.unsqueeze();
        let x = self.input_layer.forward(x);
        let (x, _) = self.lstm.forward(x, None);
        let x = self.output_layer.forward(x);
        let x = x.squeeze(2);
        return x;
    }
}
