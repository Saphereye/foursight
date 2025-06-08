use rand::{random, random_range};
use std::collections::HashMap;
use std::fmt::Display;
use std::io::{self, BufRead};

const ROWS: usize = 6;
const COLS: usize = 7;

// Bitboard representation for faster operations
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct BitBoard {
    player1: u64,
    player2: u64,
    heights: [u8; COLS], // Height of each column
}

impl Display for BitBoard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in (0..ROWS).rev() {
            write!(f, "|")?;
            for col in 0..COLS {
                let pos = col * (ROWS + 1) + row;
                let mask = 1u64 << pos;
                let char = if self.player1 & mask != 0 {
                    'X'
                } else if self.player2 & mask != 0 {
                    'O'
                } else {
                    '.'
                };
                write!(f, "{}|", char)?;
            }
            writeln!(f)?;
        }
        writeln!(f, "|1|2|3|4|5|6|7|")?;
        Ok(())
    }
}

impl BitBoard {
    fn new() -> Self {
        BitBoard {
            player1: 0,
            player2: 0,
            heights: [0; COLS],
        }
    }

    fn make_move(&mut self, col: usize, is_player1: bool) -> bool {
        if col >= COLS || self.heights[col] >= ROWS as u8 {
            return false;
        }

        let position = col * (ROWS + 1) + self.heights[col] as usize;
        let mask = 1u64 << position;

        if is_player1 {
            self.player1 |= mask;
        } else {
            self.player2 |= mask;
        }

        self.heights[col] += 1;
        true
    }

    fn can_move(&self, col: usize) -> bool {
        col < COLS && self.heights[col] < ROWS as u8
    }

    fn get_legal_moves(&self) -> Vec<usize> {
        (0..COLS).filter(|&col| self.can_move(col)).collect()
    }

    fn get_legal_moves_with_threat_pruning(&self, is_player1_turn: bool) -> Vec<usize> {
        let legal_moves = self.get_legal_moves();

        // Check if opponent has winning moves on their next turn
        let opponent_threats: Vec<usize> = legal_moves
            .iter()
            .filter(|&&col| self.is_winning_move(col, !is_player1_turn))
            .copied()
            .collect();

        // If opponent has threats, we must address them
        if !opponent_threats.is_empty() {
            // If opponent has multiple threats, the position is lost
            // But we still return the blocking moves for completeness
            return opponent_threats;
        }

        // Check if we have immediate winning moves
        let winning_moves: Vec<usize> = legal_moves
            .iter()
            .filter(|&&col| self.is_winning_move(col, is_player1_turn))
            .copied()
            .collect();

        if !winning_moves.is_empty() {
            return winning_moves;
        }

        // No immediate threats or wins, return all legal moves
        legal_moves
    }

    fn count_threats(&self, is_player1: bool) -> usize {
        self.get_legal_moves()
            .iter()
            .filter(|&&col| self.is_winning_move(col, is_player1))
            .count()
    }

    fn is_winning_move(&self, col: usize, is_player1: bool) -> bool {
        if !self.can_move(col) {
            return false;
        }

        let mut temp_board = *self;
        temp_board.make_move(col, is_player1);
        temp_board.check_winner().is_some()
    }

    fn check_winner(&self) -> Option<bool> {
        if self.is_winning_position(self.player1) {
            Some(true) // Player 1 wins
        } else if self.is_winning_position(self.player2) {
            Some(false) // Player 2 wins
        } else {
            None // No winner yet
        }
    }

    fn is_winning_position(&self, player_board: u64) -> bool {
        // Horizontal check
        let mut temp = player_board & (player_board >> (ROWS + 1));
        if temp & (temp >> (2 * (ROWS + 1))) != 0 {
            return true;
        }

        // Vertical check
        temp = player_board & (player_board >> 1);
        if temp & (temp >> 2) != 0 {
            return true;
        }

        // Diagonal / check
        temp = player_board & (player_board >> ROWS);
        if temp & (temp >> (2 * ROWS)) != 0 {
            return true;
        }

        // Diagonal \ check
        temp = player_board & (player_board >> (ROWS + 2));
        if temp & (temp >> (2 * (ROWS + 2))) != 0 {
            return true;
        }

        false
    }

    fn is_full(&self) -> bool {
        self.heights.iter().all(|&h| h >= ROWS as u8)
    }
}

#[derive(Clone)]
struct MCTSNode {
    board: BitBoard,
    is_player1_turn: bool,
    visits: u32,
    wins: f64,
    children: HashMap<usize, Box<MCTSNode>>,
    untried_moves: Vec<usize>,
}

impl MCTSNode {
    fn new(board: BitBoard, is_player1_turn: bool) -> Self {
        let untried_moves = board.get_legal_moves_with_threat_pruning(is_player1_turn);
        MCTSNode {
            board,
            is_player1_turn,
            visits: 0,
            wins: 0.0,
            children: HashMap::new(),
            untried_moves,
        }
    }

    fn ucb1_value(&self, parent_visits: u32, exploration_constant: f64) -> f64 {
        if self.visits == 0 {
            f64::INFINITY
        } else {
            let exploitation = self.wins / self.visits as f64;
            let exploration =
                exploration_constant * ((parent_visits as f64).ln() / self.visits as f64).sqrt();
            exploitation + exploration
        }
    }

    fn select_best_child(&self, exploration_constant: f64) -> Option<usize> {
        self.children
            .iter()
            .map(|(&mov, child)| (mov, child.ucb1_value(self.visits, exploration_constant)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(mov, _)| mov)
    }

    fn expand(&mut self, mov: usize) -> &mut MCTSNode {
        let mut new_board = self.board;
        new_board.make_move(mov, self.is_player1_turn);

        let new_node = Box::new(MCTSNode::new(new_board, !self.is_player1_turn));
        self.children.insert(mov, new_node);
        self.untried_moves.retain(|&x| x != mov);

        self.children.get_mut(&mov).unwrap()
    }

    fn backpropagate(&mut self, result: f64) {
        self.visits += 1;
        self.wins += result;
    }

    fn get_best_move(&self) -> Option<usize> {
        // Use visits for robustness, but also consider win rate
        self.children
            .iter()
            .map(|(&mov, child)| {
                let win_rate = if child.visits > 0 {
                    child.wins / child.visits as f64
                } else {
                    0.0
                };
                let robustness_score = child.visits as f64 + win_rate * 100.0;
                (mov, robustness_score)
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(mov, _)| mov)
    }
}

struct MCTSEngine {
    iterations: u32,
    exploration_constant: f64,
}

impl MCTSEngine {
    fn new() -> Self {
        MCTSEngine {
            iterations: u32::MAX, // Increased default iterations
            exploration_constant: 1.414,
        }
    }

    fn search(&self, board: BitBoard, is_player1_turn: bool, move_time_ms: u32) -> usize {
        // Enhanced immediate tactical analysis
        let legal_moves = board.get_legal_moves();
        let time_limit =
            std::time::Instant::now() + std::time::Duration::from_millis(move_time_ms as u64);
        let start_time = std::time::Instant::now();

        // 1. Check for immediate winning moves
        for &col in &legal_moves {
            if board.is_winning_move(col, is_player1_turn) {
                return col;
            }
        }

        // 2. Check for necessary blocking moves (opponent threats)
        let opponent_threats: Vec<usize> = legal_moves
            .iter()
            .filter(|&&col| board.is_winning_move(col, !is_player1_turn))
            .copied()
            .collect();

        // If opponent has multiple threats, position is likely lost, but block one anyway
        if opponent_threats.len() >= 2 {
            return opponent_threats[0]; // Block one threat (position likely lost anyway)
        } else if opponent_threats.len() == 1 {
            // Single threat - must block it, but also check if blocking creates a counter-threat
            let blocking_move = opponent_threats[0];
            let mut temp_board = board;
            temp_board.make_move(blocking_move, is_player1_turn);

            // If blocking creates our own winning threat, it's definitely the right move
            let our_threats_after_block = temp_board.count_threats(is_player1_turn);
            if our_threats_after_block > 0 {
                return blocking_move;
            }

            // Otherwise, still must block (forced move)
            return blocking_move;
        }

        // 3. No immediate threats, proceed with MCTS search
        let mut root = MCTSNode::new(board, is_player1_turn);

        for _ in 0..self.iterations {
            // Check if we should stop early based on time limit
            if std::time::Instant::now() >= time_limit {
                break;
            }

            // Log out the best move in uci style every 1000 iterations
            if root.visits % 100_000 == 0 {
                println!(
                    "info string iteration {} score {:.2} nodes {} nps {:.0} pv {:?}",
                    root.visits,
                    if root.visits > 0 {
                        (root.wins / root.visits as f64) * 2.0 - 1.0
                    } else {
                        0.0
                    },
                    root.visits,
                    root.visits as f64
                        / (std::time::Instant::now() - start_time)
                            .as_secs_f64()
                            .max(1.0),
                    root.get_best_move().unwrap_or(3) + 1
                );
            }
            self.mcts_iteration(&mut root);
        }

        // Return the most robust move
        root.get_best_move().unwrap_or(3) // Default to center if no moves found
    }

    fn evaluate_position(
        &self,
        board: BitBoard,
        is_player1_turn: bool,
        move_time_ms: u32,
        iterations: u32,
    ) -> (f64, HashMap<usize, (f64, u32)>) {
        let time_limit =
            std::time::Instant::now() + std::time::Duration::from_millis(move_time_ms as u64);
        // Quick check for terminal positions
        if let Some(winner) = board.check_winner() {
            let score = if winner == is_player1_turn { 1.0 } else { -1.0 };
            return (score, HashMap::new());
        }

        if board.is_full() {
            return (0.0, HashMap::new());
        }

        // Check for immediate winning moves first
        let legal_moves = board.get_legal_moves();
        for &mov in &legal_moves {
            if board.is_winning_move(mov, is_player1_turn) {
                // If we have a winning move, position is winning
                let mut move_evals = HashMap::new();
                move_evals.insert(mov, (1.0, 1)); // Winning move gets perfect score
                return (1.0, move_evals);
            }
        }

        // Check if opponent has winning moves (we must block)
        let opponent_threats: Vec<usize> = legal_moves
            .iter()
            .filter(|&&col| board.is_winning_move(col, !is_player1_turn))
            .copied()
            .collect();

        if opponent_threats.len() >= 2 {
            // Multiple opponent threats = we lose
            let mut move_evals = HashMap::new();
            for &mov in &opponent_threats {
                move_evals.insert(mov, (-0.9, 1)); // Blocking moves, but position is still losing
            }
            return (-1.0, move_evals);
        }

        // Perform MCTS search for evaluation
        let mut root = MCTSNode::new(board, is_player1_turn);

        for _ in 0..iterations {
            if std::time::Instant::now() >= time_limit {
                break; // Stop if we hit the time limit
            }
            self.mcts_iteration(&mut root);
        }

        // Calculate overall position score
        let position_score = if root.visits > 0 {
            // MCTS wins are from current player's perspective
            // High win rate = good for current player
            let win_rate = root.wins / root.visits as f64;
            win_rate * 2.0 - 1.0 // Convert [0,1] to [-1,1]
        } else {
            0.0
        };

        // Collect move evaluations
        let mut move_evaluations = HashMap::new();
        for (&mov, child) in &root.children {
            if child.visits > 0 {
                // Child's win rate is from the child's perspective (opponent's turn)
                // So we need to invert it to get our perspective
                let child_win_rate = child.wins / child.visits as f64;
                let our_win_rate = 1.0 - child_win_rate; // Invert because it's opponent's turn
                let move_score = our_win_rate * 2.0 - 1.0; // Convert to [-1,1]
                move_evaluations.insert(mov, (move_score, child.visits));
            }
        }

        (position_score, move_evaluations)
    }

    fn mcts_iteration(&self, node: &mut MCTSNode) -> f64 {
        // Check if terminal
        if let Some(winner) = node.board.check_winner() {
            let result = if winner == node.is_player1_turn {
                0.0
            } else {
                1.0
            };
            node.backpropagate(result);
            return result;
        }

        if node.board.is_full() {
            node.backpropagate(0.5);
            return 0.5;
        }

        // Early pruning: if opponent has multiple threats next turn, this position is likely lost
        let opponent_threats = node.board.count_threats(!node.is_player1_turn);
        if opponent_threats >= 2 {
            // Position is lost - opponent has multiple winning moves
            let result = 0.0; // Loss for current player
            node.backpropagate(result);
            return result;
        }

        // Expansion phase
        if !node.untried_moves.is_empty() {
            let move_idx = random_range(0..node.untried_moves.len());
            let mov = node.untried_moves[move_idx];
            let child = node.expand(mov);

            let result = self.simulate(child.board, child.is_player1_turn);
            let score = if result == child.is_player1_turn {
                0.0
            } else {
                1.0
            };

            child.backpropagate(score);
            let parent_score = 1.0 - score;
            node.backpropagate(parent_score);
            return parent_score;
        }

        // Selection phase - but only if we have children to select from
        if !node.children.is_empty() {
            if let Some(best_move) = node.select_best_child(self.exploration_constant) {
                let child = node.children.get_mut(&best_move).unwrap();
                let result = self.mcts_iteration(child);
                let score = 1.0 - result;
                node.backpropagate(score);
                return score;
            }
        }

        // Leaf node - simulate
        let result = self.simulate(node.board, node.is_player1_turn);
        let score = if result == node.is_player1_turn {
            0.0
        } else {
            1.0
        };
        node.backpropagate(score);
        score
    }

    fn simulate(&self, mut board: BitBoard, mut is_player1_turn: bool) -> bool {
        let mut moves_played = 0;
        const MAX_SIMULATION_MOVES: usize = 42; // Prevent infinite loops

        while moves_played < MAX_SIMULATION_MOVES {
            // Check for winner
            if let Some(winner) = board.check_winner() {
                return winner;
            }

            if board.is_full() {
                return random::<bool>(); // Random winner for draw in simulation
            }

            let legal_moves = board.get_legal_moves();
            if legal_moves.is_empty() {
                break;
            }

            // Improved simulation policy
            let mut chosen_move = None;

            // 1. Check for winning moves
            for &mov in &legal_moves {
                if board.is_winning_move(mov, is_player1_turn) {
                    chosen_move = Some(mov);
                    break;
                }
            }

            // 2. Check for blocking moves
            if chosen_move.is_none() {
                for &mov in &legal_moves {
                    if board.is_winning_move(mov, !is_player1_turn) {
                        chosen_move = Some(mov);
                        break;
                    }
                }
            }

            // 3. Weighted random selection favoring center
            if chosen_move.is_none() {
                let weights: Vec<f32> = legal_moves
                    .iter()
                    .map(|&col| {
                        match col {
                            3 => 4.0,     // Center
                            2 | 4 => 3.0, // Near center
                            1 | 5 => 2.0, // Outer
                            _ => 1.0,     // Edges
                        }
                    })
                    .collect();

                let total_weight: f32 = weights.iter().sum();
                let mut random_val = random::<f32>() * total_weight;

                for (i, &mov) in legal_moves.iter().enumerate() {
                    random_val -= weights[i];
                    if random_val <= 0.0 {
                        chosen_move = Some(mov);
                        break;
                    }
                }
            }

            let mov = chosen_move.unwrap_or(legal_moves[0]);
            board.make_move(mov, is_player1_turn);
            is_player1_turn = !is_player1_turn;
            moves_played += 1;
        }

        // If simulation didn't end naturally, return random
        random::<bool>()
    }
}

fn main() {
    let mut engine = MCTSEngine::new();
    let mut current_board = BitBoard::new();
    let mut is_player1_turn = true;
    let stdin = io::stdin();

    for line in stdin.lock().lines() {
        let line = line.unwrap().trim().to_string();
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "uci" => {
                println!(
                    "id name {} {}",
                    env!("CARGO_PKG_NAME"),
                    env!("CARGO_PKG_VERSION")
                );
                println!("id author {}", env!("CARGO_PKG_AUTHORS"));
                println!("option name Iterations type spin");
                println!("option name ExplorationConstant type string default 1.414");
                println!("uciok");
            }
            "isready" => {
                println!("readyok");
            }
            "setoption" => {
                if parts.len() >= 5 && parts[1] == "name" {
                    match parts[2] {
                        "Iterations" if parts[3] == "value" => {
                            if let Ok(iter) = parts[4].parse::<u32>() {
                                engine.iterations = iter;
                            }
                        }
                        "ExplorationConstant" if parts[3] == "value" => {
                            if let Ok(exp) = parts[4].parse::<f64>() {
                                engine.exploration_constant = exp;
                            }
                        }
                        _ => {}
                    }
                }
            }
            "position" => {
                if parts.len() >= 2 && parts[1] == "startpos" {
                    current_board = BitBoard::new();
                    is_player1_turn = true;

                    if parts.len() >= 4 && parts[2] == "moves" {
                        for part in parts.iter().skip(3) {
                            if let Ok(col) = part.parse::<usize>() {
                                if (1..=7).contains(&col) && current_board.can_move(col - 1) {
                                    current_board.make_move(col - 1, is_player1_turn);
                                    is_player1_turn = !is_player1_turn;
                                }
                            }
                        }
                    }
                }
            }
            "go" => {
                // Parse search parameters
                let mut search_iterations = engine.iterations;
                let mut move_time_ms = u32::MAX;

                for i in 1..parts.len() {
                    match parts[i] {
                        "movetime" if i + 1 < parts.len() => {
                            if let Ok(time_ms) = parts[i + 1].parse::<u32>() {
                                // Rough estimation: 1000 iterations per 10ms
                                move_time_ms = time_ms;
                            }
                        }
                        "iterations" if i + 1 < parts.len() => {
                            if let Ok(iter) = parts[i + 1].parse::<u32>() {
                                search_iterations = iter;
                            }
                        }
                        _ => {}
                    }
                }

                let original_iterations = engine.iterations;
                engine.iterations = search_iterations;

                let best_move = engine.search(current_board, is_player1_turn, move_time_ms);
                println!("bestmove {}", best_move + 1); // Convert back to 1-indexed

                engine.iterations = original_iterations;
            }
            "quit" => {
                break;
            }
            "eval" => {
                let legal_moves = current_board.get_legal_moves();
                let p1_threats = current_board.count_threats(true);
                let p2_threats = current_board.count_threats(false);
                let winner = current_board.check_winner();
                let mut move_time_ms = u32::MAX;
                let mut iterations = engine.iterations;
                for i in 1..parts.len() {
                    match parts[i] {
                        "movetime" if i + 1 < parts.len() => {
                            if let Ok(time_ms) = parts[i + 1].parse::<u32>() {
                                move_time_ms = time_ms;
                            }
                        }
                        "iterations" if i + 1 < parts.len() => {
                            if let Ok(iter) = parts[i + 1].parse::<u32>() {
                                iterations = iter;
                            }
                        }
                        _ => {}
                    }
                }

                println!("info string === Position Evaluation ===");

                // Check for immediate tactical situations
                if let Some(w) = winner {
                    println!(
                        "info string Game Over: {} wins",
                        if w { "Player 1 (X)" } else { "Player 2 (O)" }
                    );
                    continue;
                }

                if current_board.is_full() {
                    println!("info string Game Over: Draw");
                    continue;
                }

                // Immediate tactical analysis
                let winning_moves: Vec<usize> = legal_moves
                    .iter()
                    .filter(|&&col| current_board.is_winning_move(col, is_player1_turn))
                    .copied()
                    .collect();

                let blocking_moves: Vec<usize> = legal_moves
                    .iter()
                    .filter(|&&col| current_board.is_winning_move(col, !is_player1_turn))
                    .copied()
                    .collect();

                if !winning_moves.is_empty() {
                    println!(
                        "info string IMMEDIATE WIN available: {:?}",
                        winning_moves.iter().map(|&m| m + 1).collect::<Vec<_>>()
                    );
                }

                if blocking_moves.len() >= 2 {
                    println!(
                        "info string POSITION LOST: Opponent has multiple threats: {:?}",
                        blocking_moves.iter().map(|&m| m + 1).collect::<Vec<_>>()
                    );
                } else if !blocking_moves.is_empty() {
                    println!(
                        "info string MUST BLOCK opponent threat: {:?}",
                        blocking_moves.iter().map(|&m| m + 1).collect::<Vec<_>>()
                    );
                }

                // Perform MCTS evaluation
                if winning_moves.is_empty() && blocking_moves.len() < 2 {
                    println!("info string Running MCTS evaluation...");
                }

                let (position_score, move_evals) = engine.evaluate_position(
                    current_board,
                    is_player1_turn,
                    move_time_ms,
                    iterations,
                );

                println!(
                    "info string Current player: {}",
                    if is_player1_turn {
                        "Player 1 (X)"
                    } else {
                        "Player 2 (O)"
                    }
                );
                println!(
                    "info string Position score: {:.3} (positive favors current player)",
                    position_score
                );

                // Interpret the score
                let interpretation = if position_score >= 0.99 {
                    "Winning (forced win available)"
                } else if position_score <= -0.99 {
                    "Losing (opponent has forced win)"
                } else if position_score > 0.3 {
                    "Strong advantage"
                } else if position_score > 0.1 {
                    "Slight advantage"
                } else if position_score > -0.1 {
                    "Roughly equal"
                } else if position_score > -0.3 {
                    "Slight disadvantage"
                } else {
                    "Strong disadvantage"
                };
                println!("info string Assessment: {}", interpretation);

                println!(
                    "info string Player 1 threats: {}, Player 2 threats: {}",
                    p1_threats, p2_threats
                );

                // Show detailed move analysis
                println!("info string === Move Analysis ===");
                let mut move_data: Vec<(usize, f64, u32, String)> = move_evals
                    .iter()
                    .map(|(&mov, &(score, visits))| {
                        let mut flags = Vec::new();
                        if winning_moves.contains(&mov) {
                            flags.push("WIN".to_string());
                        }
                        if blocking_moves.contains(&mov) {
                            flags.push("BLOCK".to_string());
                        }
                        if mov == 3 {
                            flags.push("CENTER".to_string());
                        }
                        let flag_str = if flags.is_empty() {
                            "".to_string()
                        } else {
                            format!(" [{}]", flags.join(","))
                        };
                        (mov, score, visits, flag_str)
                    })
                    .collect();

                // Sort by score (best moves first)
                move_data.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                for (mov, score, visits, flags) in move_data {
                    println!(
                        "info string   Move {}: {score:+.3} ({visits} simulations){flags}",
                        mov + 1
                    );
                }

                // Show moves not yet explored (only if not a tactical position)
                if winning_moves.is_empty() && blocking_moves.len() <= 1 {
                    let unexplored: Vec<usize> = legal_moves
                        .iter()
                        .filter(|&&mov| !move_evals.contains_key(&mov))
                        .copied()
                        .collect();

                    if !unexplored.is_empty() {
                        println!(
                            "info string Unexplored moves: {:?}",
                            unexplored.iter().map(|&m| m + 1).collect::<Vec<_>>()
                        );
                    }
                }
            }
            "d" => {
                // Debug command to display current board
                print!("{}", current_board);
            }
            _ => {
                // Unknown command, ignore silently
            }
        }
    }
}
