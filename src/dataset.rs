//! Sudoku Dataset Module using direct CSV download
//!
//! This module provides functionality to download and process Sudoku puzzles
//! from HuggingFace Hub by downloading CSV files directly, similar to the PyTorch version.

use burn::data::dataloader::batcher::Batcher;
use burn::tensor::{Int, Tensor, backend::Backend};
use serde::{Deserialize, Serialize};

/// Configuration for Sudoku dataset loading
#[derive(Debug, Clone)]
pub struct SudokuDatasetConfig {
    pub repo: String,
    pub split: String,
    pub cache_dir: Option<String>,
    pub subsample_size: Option<usize>,
    pub min_difficulty: Option<u32>,
}

impl Default for SudokuDatasetConfig {
    fn default() -> Self {
        Self {
            repo: "sapientinc/sudoku-extreme".to_string(),
            split: "train".to_string(),
            cache_dir: None,
            subsample_size: None,
            min_difficulty: None,
        }
    }
}

/// Raw Sudoku item from CSV
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SudokuItem {
    pub source: String,
    pub puzzle: String,
    pub solution: String,
    pub rating: u32,
}

/// Processed Sudoku data for training
#[derive(Debug, Clone)]
pub struct SudokuData {
    pub input: Vec<u8>,  // 81-element vector (9x9 flattened), values 1-10
    pub target: Vec<u8>, // 81-element vector (9x9 flattened), values 1-10
    pub puzzle_id: usize,
}

/// Training batch for Sudoku puzzles
#[derive(Debug, Clone)]
pub struct SudokuBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,             // [batch_size, 81]
    pub targets: Tensor<B, 2, Int>,            // [batch_size, 81]
    pub puzzle_identifiers: Tensor<B, 1, Int>, // [batch_size] - all zeros for blank identifier
}

/// Simple dataset wrapper for Sudoku data
pub struct SudokuDataset {
    pub items: Vec<SudokuData>,
}

impl SudokuDataset {
    /// Create a new Sudoku dataset by downloading and processing CSV files
    pub fn new(
        config: SudokuDatasetConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let csv_data =
            download_csv_with_cache(&config.repo, &config.split, config.cache_dir.as_deref())?;
        let items = process_csv_data(csv_data, &config)?;
        Ok(Self { items })
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn get(&self, index: usize) -> Option<&SudokuData> {
        self.items.get(index)
    }
}

impl burn::data::dataset::Dataset<SudokuData> for SudokuDataset {
    fn get(&self, index: usize) -> Option<SudokuData> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

/// Batcher for converting SudokuData items into batches
#[derive(Clone, Default)]
pub struct SudokuBatcher<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> SudokuBatcher<B> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Batcher<B, SudokuData, SudokuBatch<B>> for SudokuBatcher<B> {
    fn batch(&self, items: Vec<SudokuData>, device: &B::Device) -> SudokuBatch<B> {
        let batch_size = items.len();
        let seq_len = 81; // 9x9 Sudoku grid

        // Collect data from all items
        let mut inputs_data = Vec::with_capacity(batch_size * seq_len);
        let mut targets_data = Vec::with_capacity(batch_size * seq_len);
        let mut puzzle_ids = Vec::with_capacity(batch_size);

        for item in items {
            inputs_data.extend(item.input.iter().map(|&x| x as i32));
            targets_data.extend(item.target.iter().map(|&x| x as i32));
            puzzle_ids.push(0i32); // All puzzles use blank identifier (0)
        }

        // Create tensors
        let inputs = Tensor::<B, 1, Int>::from_ints(inputs_data.as_slice(), device)
            .reshape([batch_size, seq_len]);
        let targets = Tensor::<B, 1, Int>::from_ints(targets_data.as_slice(), device)
            .reshape([batch_size, seq_len]);
        let puzzle_identifiers = Tensor::<B, 1, Int>::from_ints(puzzle_ids.as_slice(), device);

        SudokuBatch {
            inputs,
            targets,
            puzzle_identifiers,
        }
    }
}

/// Download CSV file directly from HuggingFace Hub with caching
fn download_csv_with_cache(
    repo: &str,
    split: &str,
    cache_dir: Option<&str>,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    use std::fs;
    use std::process::Command;

    // Create cache directory
    let cache_dir_path = match cache_dir {
        Some(dir) => format!("{}/hrm_datasets", dir),
        None => format!(
            "{}/.cache/hrm_datasets",
            std::env::var("HOME").unwrap_or("/tmp".to_string())
        ),
    };
    fs::create_dir_all(&cache_dir_path)?;

    // Cache file path
    let cache_file = format!(
        "{}/{}_{}.csv",
        cache_dir_path,
        repo.replace("/", "_"),
        split
    );

    // Check if cached file exists and is recent (less than 1 day old)
    if let Ok(metadata) = fs::metadata(&cache_file) {
        if let Ok(modified) = metadata.modified() {
            if let Ok(elapsed) = modified.elapsed() {
                if elapsed.as_secs() < 86400 {
                    // 1 day
                    println!("Using cached {} data from {}", split, cache_file);
                    return Ok(fs::read_to_string(&cache_file)?);
                }
            }
        }
    }

    // Download if not cached or cache is old
    let url = format!(
        "https://huggingface.co/datasets/{}/resolve/main/{}.csv",
        repo, split
    );

    println!("Downloading {} data from {} to cache", split, url);

    // Use curl to download directly to cache file
    let output = Command::new("curl")
        .arg("-L") // Follow redirects
        .arg("-o")
        .arg(&cache_file)
        .arg(&url)
        .output()?;

    if !output.status.success() {
        return Err(format!(
            "Failed to download CSV: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    // Read the cached file
    let content = fs::read_to_string(&cache_file)?;

    println!(
        "Downloaded and cached {} lines of CSV data",
        content.lines().count()
    );
    Ok(content)
}

/// Process CSV data into SudokuData items
fn process_csv_data(
    csv_content: String,
    config: &SudokuDatasetConfig,
) -> Result<Vec<SudokuData>, Box<dyn std::error::Error + Send + Sync>> {
    let mut items = Vec::new();
    let mut csv_reader = csv::Reader::from_reader(csv_content.as_bytes());

    for (index, record) in csv_reader.records().enumerate() {
        let record = record?;

        if record.len() < 4 {
            continue; // Skip malformed records
        }

        let _source = record[0].to_string();
        let puzzle = record[1].to_string();
        let solution = record[2].to_string();
        let rating: u32 = record[3].parse().unwrap_or(0);

        // Apply difficulty filter if specified
        if let Some(min_diff) = config.min_difficulty {
            if rating < min_diff {
                continue;
            }
        }

        // Validate puzzle and solution length
        if puzzle.len() != 81 || solution.len() != 81 {
            continue;
        }

        // Parse puzzle and solution
        let input = parse_sudoku_string(&puzzle);
        let target = parse_sudoku_string(&solution);

        items.push(SudokuData {
            input,
            target,
            puzzle_id: index,
        });

        // Apply subsample limit if specified for training
        if config.split == "train" {
            if let Some(subsample_size) = config.subsample_size {
                if items.len() >= subsample_size {
                    break;
                }
            }
        }
    }

    println!("Processed {} puzzles from CSV", items.len());
    Ok(items)
}

/// Parse a Sudoku string (81 characters) into a vector of u8
/// '.' and '0' become 1 (empty cells), digits 1-9 become 2-10 (shifted by +1 for PAD token)
fn parse_sudoku_string(s: &str) -> Vec<u8> {
    s.chars()
        .take(81)
        .map(|c| {
            if c == '.' || c == '0' {
                1 // Empty cells become 1 (0 is reserved for PAD token)
            } else {
                c.to_digit(10).unwrap_or(1) as u8 + 1 // Digits 1-9 become 2-10
            }
        })
        .collect()
}

/// Dataset metadata for Sudoku puzzles
#[derive(Debug, Clone)]
pub struct SudokuDatasetMetadata {
    pub vocab_size: usize,              // 11 (PAD(0) + empty(1) + digits(2-10))
    pub seq_len: usize,                 // 81 (9x9 grid)
    pub num_puzzle_identifiers: usize,  // 1 (just blank identifier)
    pub pad_id: usize,                  // 0
    pub ignore_label_id: Option<usize>, // Some(1) for empty cells in loss computation
    pub blank_identifier_id: usize,     // 0
}

impl Default for SudokuDatasetMetadata {
    fn default() -> Self {
        Self {
            vocab_size: 11, // PAD(0) + empty(1) + digits(2-10)
            seq_len: 81,    // 9x9 grid
            num_puzzle_identifiers: 1,
            pad_id: 0,
            ignore_label_id: Some(1), // Ignore empty cells in loss
            blank_identifier_id: 0,
        }
    }
}

/// Create a Sudoku dataset with direct CSV download
pub fn create_sudoku_dataset(
    config: SudokuDatasetConfig,
) -> Result<SudokuDataset, Box<dyn std::error::Error + Send + Sync>> {
    SudokuDataset::new(config)
}
