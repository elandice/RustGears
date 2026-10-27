//! 외부 의존성 없이 표준 라이브러리만으로 동작하는 터미널 진행도 표시(Progress Bar) 모듈.

use std::io::{self, IsTerminal, Write};
use std::time::{Duration, Instant};

/// 경과 시간, 예상 남은 시간(ETA), 사용자 지정 지표 메시지를 출력하는 학습 진행 표시줄.
/// 출력이 파일 등으로 리다이렉트된 경우 터미널 제어 문자를 끄며, `render()`를 통해 로그용 문자열을 얻을 수 있습니다.
pub struct ProgressBar {
    total: usize,
    position: usize,
    started: Instant,
    last_draw: Instant,
    message: String,
    terminal: bool,
}

impl ProgressBar {
    /// 총 단계 수(`total`)로 새 프로그레스 바를 생성합니다.
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

    /// 진행 바 뒤에 표시할 부가 메시지(손실값, 정확도 등)를 설정합니다.
    pub fn set_message(&mut self, message: impl Into<String>) {
        self.message = message.into();
    }

    /// 진행 단계를 `steps`만큼 증가시키고 화면을 갱신합니다.
    pub fn inc(&mut self, steps: usize) -> io::Result<()> {
        self.position = self.position.saturating_add(steps).min(self.total);
        if self.terminal && (self.last_draw.elapsed() >= Duration::from_millis(100) || self.position == self.total) {
            self.draw(false)?;
        }
        Ok(())
    }

    /// 진행을 100%로 완료하고 줄바꿈을 출력합니다.
    pub fn finish(&mut self) -> io::Result<()> {
        self.position = self.total;
        if self.terminal {
            self.draw(true)?;
        }
        Ok(())
    }

    /// 현재 진행 상태를 서식화된 문자열로 렌더링합니다.
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

    /// 표준 에러(stderr)에 현재 진행률을 그립니다.
    fn draw(&mut self, newline: bool) -> io::Result<()> {
        let mut stderr = io::stderr().lock();
        write!(stderr, "\r\x1b[2K{}{}", self.render(), if newline { "\n" } else { "" })?;
        stderr.flush()?;
        self.last_draw = Instant::now();
        Ok(())
    }
}
