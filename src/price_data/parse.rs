use std::fs::File;

use chrono::NaiveDate;

use super::PriceDataItem;

pub fn fetch_company_price_data(
    file: File,
    start: NaiveDate,
    end: NaiveDate,
    symbol: String,
) -> Vec<PriceDataItem> {
    let mut rdr = csv::Reader::from_reader(file);
    let mut records: Vec<PriceDataItem> = Vec::new();

    for (num, result) in rdr.records().enumerate() {
        // Some records in the data have malformed lengths.
        // Because we cannot know the fields that these records belong to are for,
        // and we cannot know the correct values for these fields, we skip these records.
        if result.is_err() {
            let err = result.err().unwrap();
            match err.kind() {
                csv::ErrorKind::UnequalLengths {
                    pos: _pos,
                    expected_len: _expected_len,
                    len: _len,
                } => {
                    println!(
                        "SKIPPING RECORD, error in price data csv record row:{num} {:?}, SYMBOL: {symbol}",
                        err
                    );
                    continue;
                }
                _ => {
                    panic!("{err}");
                }
            }
        }
        let record =
            result.expect(format!("error in price data csv record, SYMBOL: {symbol}",).as_str());

        // when the stock did not trade on this day, continue to the next record
        if record[1].is_empty() {
            continue;
        }

        // This handles malformed dates in the csv. As these do not conform to the standard date format of the data,
        // therefore the date string cannot be reliably parsed and the record is skipped.
        if record[0].len() != 10 {
            println!("SKIPPING RECORD, malformed date, SYMBOL: {symbol}",);
            continue;
        }

        // This handles the edge case where the volume is a whole number but the csv file has a decimal value of 0.
        // Sometimes rust cannot safely parse this value, and errors out.
        // We know that all volumes are whole numbers, so we can safely parse the volume as an integer by dropping the decimal.
        let volume: i64 = record[3]
            .split(".")
            .next()
            .unwrap()
            .parse()
            .expect(format!("Could not parse volume: {:?}", record).as_str());

        let record = PriceDataItem {
            date: NaiveDate::parse_from_str(&record[0], "%d-%m-%Y")
                .expect(format!("Could not parse date: {:?}, SYMBOL: {symbol}", record).as_str()),
            low: record[1]
                .parse()
                .expect(format!("Could not parse low: {:?} SYMBOL: {symbol}", record).as_str()),
            open: record[2]
                .parse()
                .expect(format!("Could not parse open: {:?} SYMBOL: {symbol}", record).as_str()),
            volume: volume,
            high: record[4]
                .parse()
                .expect(format!("Could not parse high: {:?}  SYMBOL: {symbol}", record).as_str()),
            close: record[5]
                .parse()
                .expect(format!("Could not parse close: {:?}  SYMBOL: {symbol}", record).as_str()),
            adjusted_close: record[6].parse().expect(
                format!(
                    "Could not parse adjusted_close: {:?}  SYMBOL: {symbol}",
                    record
                )
                .as_str(),
            ),
        };

        if record.date < start || record.date > end {
            continue;
        }

        records.push(record);
    }
    return records;
}
