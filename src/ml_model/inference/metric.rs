use chrono::{Datelike, Months, NaiveDate};
use plotly::{common::Mode, layout::Annotation, HeatMap, Layout, Plot, Scatter};

use crate::price_data::{CompaniesPriceData, Company};

use super::{
    book::{Side, Transaction},
    engine::StrategyEngine,
};

pub fn plot_model_output(model_output: Vec<(NaiveDate, f64)>) {
    let value_history = Scatter::new(
        model_output.iter().map(|item| item.0).collect(),
        model_output.iter().map(|item| item.1).collect(),
    )
    .mode(Mode::Lines)
    .name("Value History");

    let mut profit_plot = Plot::new();
    profit_plot.add_trace(value_history);
    profit_plot.show();
}

pub struct BiasWindow {
    pub year: i32,
    pub month: u32,
    pub bias: f64,
}

pub struct CompanyBias {
    pub company: Company,
    pub windows: Vec<BiasWindow>,
}

impl CompanyBias {
    pub fn new(company: Company, start_date: NaiveDate, end_date: NaiveDate) -> CompanyBias {
        let mut windows = Vec::new();

        let mut curr_date = start_date;
        while curr_date < end_date {
            windows.push(BiasWindow {
                year: curr_date.year(),
                month: curr_date.month(),
                bias: 0.0,
            });
            curr_date = curr_date.checked_add_months(Months::new(1)).unwrap();
        }

        CompanyBias { company, windows }
    }

    pub fn windows(&self) -> Vec<f64> {
        self.windows.iter().map(|window| window.bias).collect()
    }
}

pub fn calculate_company_bias(
    companies: Vec<Company>,
    transactions: Vec<Transaction>,
    start_date: NaiveDate,
    end_date: NaiveDate,
) -> Vec<CompanyBias> {
    let mut company_biases: Vec<CompanyBias> = companies
        .iter()
        .map(|company| CompanyBias::new(company.clone(), start_date, end_date))
        .collect();

    for transaction in transactions {
        match transaction.side {
            Side::Buy => {
                let year = transaction.date.year();
                let month = transaction.date.month();

                let company_bias = company_biases
                    .iter_mut()
                    .find(|company| company.company == transaction.holding.company)
                    .unwrap();

                for window in company_bias.windows.iter_mut() {
                    if window.year == year && window.month == month {
                        window.bias += transaction.holding.purchase_value();
                    }
                }
            }
            // ignore sale transactions
            _ => {}
        }
    }

    company_biases
}

pub fn plot_company_bias(
    companies_price_data: &CompaniesPriceData,
    start_date: NaiveDate,
    end_date: NaiveDate,
    engine: &StrategyEngine,
) {
    let companies = companies_price_data
        .companies
        .iter()
        .map(|company| company.company())
        .collect();

    let company_biases =
        calculate_company_bias(companies, engine.book.history.clone(), start_date, end_date);

    let company_biases_out = company_biases
        .iter()
        .map(|company_bias| company_bias.windows())
        .collect();

    let company_biases_symbols: Vec<&str> = company_biases
        .iter()
        .map(|company_bias| company_bias.company.symbol.as_str())
        .collect();

    let bias_trace = HeatMap::new_z(company_biases_out);

    let y_annotations: Vec<Annotation> = company_biases_symbols
        .iter()
        .enumerate()
        .map(|(i, symbol)| {
            Annotation::new()
                .x(-1.0)
                .y(i as f64)
                .text(symbol)
                .show_arrow(false)
        })
        .collect();

    let mut curr_date = start_date.clone();
    let mut x_annotations: Vec<Annotation> = Vec::new();
    let mut i = 0;

    while curr_date < end_date {
        let interval = 6;
        let x_annotation = Annotation::new()
            .x(0.5 + (i * interval) as f64)
            .y(-1.0)
            .text(format!(
                "{month}{year}",
                month = month_num_to_str(curr_date.month0()),
                year = curr_date.year()
            ))
            .text_angle(45.0)
            .show_arrow(false);

        x_annotations.push(x_annotation);
        curr_date = curr_date.checked_add_months(Months::new(interval)).unwrap();
        i += 1;
    }

    let mut layout = Layout::new().title("Distrubution of monthly purchases");

    for annotation in y_annotations.iter() {
        layout.add_annotation(annotation.to_owned());
    }

    for annotation in x_annotations.iter() {
        layout.add_annotation(annotation.to_owned());
    }

    let mut bias_plot = Plot::new();
    bias_plot.add_trace(bias_trace);
    bias_plot.set_layout(layout);
    bias_plot.use_local_plotly();
    bias_plot.show();
}

pub fn month_num_to_str(month: u32) -> String {
    match month {
        0 => "Jan".to_string(),
        1 => "Feb".to_string(),
        2 => "Mar".to_string(),
        3 => "Apr".to_string(),
        4 => "May".to_string(),
        5 => "Jun".to_string(),
        6 => "Jul".to_string(),
        7 => "Aug".to_string(),
        8 => "Sep".to_string(),
        9 => "Oct".to_string(),
        10 => "Nov".to_string(),
        11 => "Dec".to_string(),
        _ => "Invalid month".to_string(),
    }
}
