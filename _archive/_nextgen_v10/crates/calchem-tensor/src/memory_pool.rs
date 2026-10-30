use std::sync::atomic::{AtomicUsize, Ordering};

pub struct MemoryWatermarkGuard {
    max_bytes: usize,
    allocated_bytes: AtomicUsize,
}

impl MemoryWatermarkGuard {
    pub fn new(max_gigabytes: f64) -> Self {
        Self {
            max_bytes: (max_gigabytes * 1024.0 * 1024.0 * 1024.0) as usize,
            allocated_bytes: AtomicUsize::new(0),
        }
    }

    pub fn acquire(&self, bytes: usize) -> Result<(), String> {
        let current = self.allocated_bytes.fetch_add(bytes, Ordering::SeqCst);
        if current + bytes > self.max_bytes {
            self.allocated_bytes.fetch_sub(bytes, Ordering::SeqCst);
            return Err(format!(
                "Memory limit exceeded! Requested: {:.2} MB, Max allowed: {:.2} GB",
                bytes as f64 / 1e6,
                self.max_bytes as f64 / 1e9
            ));
        }
        Ok(())
    }

    pub fn release(&self, bytes: usize) {
        self.allocated_bytes.fetch_sub(bytes, Ordering::SeqCst);
    }
}
