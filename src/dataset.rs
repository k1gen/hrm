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
    pub num_aug: usize, // Number of augmentations per puzzle
}

impl Default for SudokuDatasetConfig {
    fn default() -> Self {
        Self {
            repo: "sapientinc/sudoku-extreme".to_string(),
            split: "train".to_string(),
            cache_dir: None,
            subsample_size: None,
            min_difficulty: None,
            num_aug: 0,
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
    pub inputs: Tensor<B, 2, Int>,  // [batch_size, 81]
    pub targets: Tensor<B, 2, Int>, // [batch_size, 81]
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

        for item in items {
            inputs_data.extend(item.input.iter().map(|&x| x as i32));
            targets_data.extend(item.target.iter().map(|&x| x as i32));
        }

        // Create tensors
        let inputs = Tensor::<B, 1, Int>::from_ints(inputs_data.as_slice(), device)
            .reshape([batch_size, seq_len]);
        let targets = Tensor::<B, 1, Int>::from_ints(targets_data.as_slice(), device)
            .reshape([batch_size, seq_len]);

        SudokuBatch { inputs, targets }
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
    if let Ok(metadata) = fs::metadata(&cache_file)
        && let Ok(modified) = metadata.modified()
            && let Ok(elapsed) = modified.elapsed()
                && elapsed.as_secs() < 86400 {
                    // 1 day
                    println!("Using cached {} data from {}", split, cache_file);
                    return Ok(fs::read_to_string(&cache_file)?);
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
        if let Some(min_diff) = config.min_difficulty
            && rating < min_diff {
                continue;
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

        // Apply subsample limit if specified
        if let Some(subsample_size) = config.subsample_size
            && items.len() >= subsample_size {
                break;
            }
    }

    println!("Processed {} puzzles from CSV", items.len());
    // Apply augmentation for training data
    if config.split == "train" && config.num_aug > 0 {
        let original_items = items.clone();
        for original_item in original_items {
            for aug_id in 1..=config.num_aug {
                let (aug_input, aug_target) =
                    apply_sudoku_augmentation(&original_item.input, &original_item.target, aug_id);
                items.push(SudokuData {
                    input: aug_input,
                    target: aug_target,
                    puzzle_id: original_item.puzzle_id * 1000 + aug_id, // Unique ID for augmented puzzles
                });
            }
        }
    }

    println!(
        "Processed {} puzzles from CSV (including {} augmentations)",
        items.len(),
        if config.split == "train" {
            config.num_aug
        } else {
            0
        }
    );
    Ok(items)
}

/// Apply PyTorch-style Sudoku augmentation with digit permutation and advanced position mapping
fn apply_sudoku_augmentation(input: &[u8], target: &[u8], aug_id: usize) -> (Vec<u8>, Vec<u8>) {
    use nanorand::WyRand;

    // Use aug_id as seed for reproducible augmentations
    let mut rng = WyRand::new_seed(aug_id as u64);

    // Convert input back to 0-9 range (subtract 1 since our data is 1-10)
    let mut input_grid: [[u8; 9]; 9] = [[0; 9]; 9];
    let mut target_grid: [[u8; 9]; 9] = [[0; 9]; 9];

    for i in 0..9 {
        for j in 0..9 {
            input_grid[i][j] = input[i * 9 + j].saturating_sub(1);
            target_grid[i][j] = target[i * 9 + j].saturating_sub(1);
        }
    }

    // Apply PyTorch-style shuffle_sudoku transformation
    let (aug_input_grid, aug_target_grid) = shuffle_sudoku_rust(input_grid, target_grid, &mut rng);

    // Convert back to flat arrays with +1 offset
    let mut aug_input = vec![0u8; 81];
    let mut aug_target = vec![0u8; 81];

    for i in 0..9 {
        for j in 0..9 {
            aug_input[i * 9 + j] = aug_input_grid[i][j] + 1;
            aug_target[i * 9 + j] = aug_target_grid[i][j] + 1;
        }
    }

    (aug_input, aug_target)
}

/// Rust implementation of PyTorch's shuffle_sudoku function
fn shuffle_sudoku_rust(
    input_board: [[u8; 9]; 9],
    target_board: [[u8; 9]; 9],
    rng: &mut nanorand::WyRand,
) -> ([[u8; 9]; 9], [[u8; 9]; 9]) {
    use nanorand::Rng;
    // Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    let mut digits: Vec<u8> = (1..=9).collect();
    shuffle_vec(&mut digits, rng);
    let mut digit_map = [0u8; 10]; // digit_map[0] = 0 (blank stays blank)
    for (i, &digit) in digits.iter().enumerate() {
        digit_map[i + 1] = digit;
    }

    // Randomly decide whether to transpose
    let transpose_flag = rng.generate::<f32>() < 0.5;

    // Generate valid row permutation: shuffle 3 bands, then shuffle rows within each band
    let mut bands: Vec<usize> = (0..3).collect();
    shuffle_vec(&mut bands, rng);

    let mut row_perm = Vec::with_capacity(9);
    for &band in &bands {
        let mut band_rows: Vec<usize> = (0..3).map(|i| band * 3 + i).collect();
        shuffle_vec(&mut band_rows, rng);
        row_perm.extend(band_rows);
    }

    // Similarly for columns (stacks)
    let mut stacks: Vec<usize> = (0..3).collect();
    shuffle_vec(&mut stacks, rng);

    let mut col_perm = Vec::with_capacity(9);
    for &stack in &stacks {
        let mut stack_cols: Vec<usize> = (0..3).map(|i| stack * 3 + i).collect();
        shuffle_vec(&mut stack_cols, rng);
        col_perm.extend(stack_cols);
    }

    // Build 81->81 mapping
    let mut mapping = [0usize; 81];
    for i in 0..81 {
        let old_row = row_perm[i / 9];
        let old_col = col_perm[i % 9];
        mapping[i] = old_row * 9 + old_col;
    }

    // Apply transformation to both boards
    let apply_transformation = |board: [[u8; 9]; 9]| -> [[u8; 9]; 9] {
        let mut result = board;

        // Apply transpose if flag is set
        if transpose_flag {
            let mut transposed = [[0u8; 9]; 9];
            for i in 0..9 {
                for j in 0..9 {
                    transposed[j][i] = result[i][j];
                }
            }
            result = transposed;
        }

        // Apply position mapping
        let flat: Vec<u8> = result.iter().flatten().copied().collect();
        let mut new_flat = [0u8; 81];
        for i in 0..81 {
            new_flat[i] = flat[mapping[i]];
        }

        // Reshape back to 9x9
        let mut new_board = [[0u8; 9]; 9];
        for i in 0..9 {
            for j in 0..9 {
                new_board[i][j] = new_flat[i * 9 + j];
            }
        }

        // Apply digit mapping
        for i in 0..9 {
            for j in 0..9 {
                new_board[i][j] = digit_map[new_board[i][j] as usize];
            }
        }

        new_board
    };

    (
        apply_transformation(input_board),
        apply_transformation(target_board),
    )
}

/// Simple Fisher-Yates shuffle for nanorand
fn shuffle_vec<T>(slice: &mut [T], rng: &mut nanorand::WyRand) {
    use nanorand::Rng;
    for i in (1..slice.len()).rev() {
        let j = rng.generate_range(0..=i);
        slice.swap(i, j);
    }
}

/// Print the processed data statistics
#[allow(dead_code)]
fn print_data_stats(items: &[SudokuData], config: &SudokuDatasetConfig) {
    let original_count = if config.split == "train" && config.num_aug > 0 {
        items.len() / (config.num_aug + 1)
    } else {
        items.len()
    };

    println!("Dataset statistics for {}:", config.split);
    println!("  - Original puzzles: {}", original_count);
    if config.split == "train" && config.num_aug > 0 {
        println!("  - Augmentations per puzzle: {}", config.num_aug);
        println!("  - Total training examples: {}", items.len());
    }
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
    pub vocab_size: usize, // 9 (digits 0-8, representing Sudoku digits 1-9)
    pub seq_len: usize,    // 81 (9x9 grid)
}

impl Default for SudokuDatasetMetadata {
    fn default() -> Self {
        Self {
            vocab_size: 9, // Digits 1-9 (classes 0-8) - no pad token needed
            seq_len: 81,   // 9x9 grid
        }
    }
}

/// Create a Sudoku dataset with direct CSV download
pub fn create_sudoku_dataset(
    config: SudokuDatasetConfig,
) -> Result<SudokuDataset, Box<dyn std::error::Error + Send + Sync>> {
    SudokuDataset::new(config)
}
