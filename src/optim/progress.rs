//! Terminal progress reporting without additional dependencies.

use std::io::{self, IsTerminal, Write};
use std::time::{Duration, Instant};

/// A training progress bar with elapsed time, ETA and a user-supplied metric.
/// Terminal output is throttled and disabled when stderr is redirected.
/// `render()` remains available for log files and notebook frontends.
pub struct ProgressBar {
    total: usize,
    position: usize,
    started: Instant,
    last_draw: Instant,
    message: String,
    terminal: bool,
}

impl ProgressBar {
    pub fn new(total: usize) -> Self {
        let now = Instant::now();
        Self {
            total,
            position: 0,
            started: now,
            last_draw: now,
            message: String::new(),
            terminal: io::stderr().is_terminal(),
        }
    }

    pub fn set_message(&mut self, message: impl Into<String>) {
        self.message = message.into();
    }

    pub fn inc(&mut self, steps: usize) -> io::Result<()> {
        self.position = self.position.saturating_add(steps).min(self.total);
        if self.terminal && (self.last_draw.elapsed() >= Duration::from_millis(100) || self.position == self.total) {
            self.draw(false)?;
        }
        Ok(())
    }

    pub fn finish(&mut self) -> io::Result<()> {
        self.position = self.total;
        if self.terminal {
            self.draw(true)?;
        }
        Ok(())
    }

    pub fn render(&self) -> String {
        let ratio = if self.total == 0 { 1.0 } else { self.position as f64 / self.total as f64 };
        let filled = (ratio * 24.0) as usize;
        let elapsed = self.started.elapsed().as_secs_f64();
        let eta = if self.position == 0 { "--".into() } else { format!("{:.1}s", elapsed * (1.0 - ratio) / ratio) };
        format!(
            "[{}{}] {}/{} {:>3.0}% elapsed {:.1}s ETA {} {}",
            "=".repeat(filled),
            " ".repeat(24 - filled),
            self.position,
            self.total,
            ratio * 100.0,
            elapsed,
            eta,
            self.message
        )
    }

    fn draw(&mut self, newline: bool) -> io::Result<()> {
        let mut stderr = io::stderr().lock();
        write!(stderr, "\r\x1b[2K{}{}", self.render(), if newline { "\n" } else { "" })?;
        stderr.flush()?;
        self.last_draw = Instant::now();
        Ok(())
    }
}
